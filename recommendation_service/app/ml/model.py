import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjection(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super(MLPProjection, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)

class UserTower(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(UserTower, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.aggregation = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [Batch, Seq, 128]
        # 1. 0벡터(패딩)를 동적으로 감지하여 마스킹 처리 (L2 Norm이 1e-5 미만인 경우 패딩으로 판단)
        norms = torch.norm(x, dim=-1) # [Batch, Seq]
        mask = (norms < 1e-5) # [Batch, Seq] - 패딩된 위치는 True
        
        # 모든 히스토리가 빈 경우(Cold-Start)를 대비해 안전장치 마련
        all_padded = mask.all(dim=-1, keepdim=True)
        safe_mask = mask.clone()
        safe_mask[all_padded.expand_as(mask)] = False # 전체가 패딩이면 NaN 방지를 위해 마스킹을 임시로 끔
        
        # 2. 어텐션 레이어가 진짜 행동 정보에만 100% 집중하도록 패딩 마스크 적용
        attn_output, _ = self.attention(x, x, x, key_padding_mask=safe_mask)
        
        # 3. GRU를 통한 단기 선호도 추출
        _, h_n = self.aggregation(attn_output)
        gru_vec = h_n.squeeze(0) # [Batch, 128]
        
        # 4. 진짜 행동들에 대해서만 평균(Mean Pool)을 구하여 희석 방지
        active_mask = (~mask).float().unsqueeze(-1) # [Batch, Seq, 1]
        active_count = active_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = (x * active_mask).sum(dim=1) / active_count.squeeze(-1) # [Batch, 128]
        
        # 5. 최종 결합 및 레이어 정규화 (LayerNorm)
        # 만약 완전히 빈 이력이라면 노이즈 폭발을 막기 위해 최종 유저 벡터를 0벡터로 밀어버립니다.
        user_vec = self.norm(self.fc(gru_vec + mean_pool))
        normed_user_vec = F.normalize(user_vec, p=2, dim=1)
        
        return normed_user_vec * (~all_padded).float()

class UnifiedDiscoveryModel(nn.Module):
    def __init__(self):
        super(UnifiedDiscoveryModel, self).__init__()
        
        # 검색어·캡션·해시태그 모두 같은 텍스트 레이어 공유
        # (해시태그도 텍스트이므로 같은 의미 공간에 투영)
        self.text_proj = MLPProjection(768, 128)
        self.image_proj = MLPProjection(512, 128)
        
        # Fusion MLP (256 -> 128): text + image 두 모달
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
            
        self.user_tower = UserTower()

    def get_multimodal_item_embedding(self, caption_vec, hashtag_vec, image_vec):
        """
        캡션·해시태그 → text_proj (공유, L2 norm 후 평균)
        이미지      → image_proj
        fusion_mlp([text_fused, i_emb]) + 50% text 정체성 보존
        """
        def safe_proj(proj_layer, vec):
            mask = (torch.norm(vec, dim=-1, keepdim=True) > 1e-6).float()
            projected = proj_layer(vec)
            return F.normalize(projected * mask, p=2, dim=-1) * mask

        c_emb = safe_proj(self.text_proj, caption_vec)    # 캡션
        h_emb = safe_proj(self.text_proj, hashtag_vec)    # 해시태그 (같은 레이어)
        i_emb = safe_proj(self.image_proj, image_vec)     # 이미지

        # 캡션·해시태그 평균 → 단일 텍스트 표현
        text_fused = F.normalize(c_emb + h_emb, p=2, dim=-1)

        combined = torch.cat([text_fused, i_emb], dim=-1)  # [B, 256]
        fused = F.normalize(self.fusion_mlp(combined), p=2, dim=-1)

        # 텍스트 정체성(50%) + 멀티모달 문맥(50%)
        final_emb = 0.5 * fused + 0.5 * c_emb
        return F.normalize(final_emb, p=2, dim=-1)

    def get_item_embedding(self, caption_vec, hashtag_vec=None, image_vec=None):
        batch_size = caption_vec.size(0)
        device = caption_vec.device
        if hashtag_vec is None: hashtag_vec = torch.zeros((batch_size, 768), device=device)
        if image_vec is None: image_vec = torch.zeros((batch_size, 512), device=device)
        return self.get_multimodal_item_embedding(caption_vec, hashtag_vec, image_vec)

    def get_query_embedding(self, raw_query_vec):
        """검색어는 text_proj를 거쳐 128차원으로 변환 (캡션과 동일 레이어)"""
        return F.normalize(self.text_proj(raw_query_vec), p=2, dim=-1)

    def get_user_embedding(self, history_vecs):
        return self.user_tower(history_vecs)

    def discovery(self, query_vec, user_vec):
        """
        Interaction-based Fusion with Dynamic Gating: v = q + u_gated + (q * u_gated)
        단순 가중합(alpha)에서 발생하는 정보 손실을 막고, 검색어가 명확할 때는 유저 취향의 개입을 차단합니다.
        """
        # 검색어(query_vec)의 정보량(norm) 계산
        query_intensity = torch.norm(query_vec, dim=-1, keepdim=True)
        
        # 검색어 강도가 높을수록 유저 벡터의 영향력을 감소시킴 (0에 수렴)
        gated_user_vec = user_vec * (1.0 - torch.tanh(query_intensity))
        
        interaction = query_vec * gated_user_vec
        combined_vec = query_vec + gated_user_vec + interaction
        return F.normalize(combined_vec, p=2, dim=-1)
