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

    def forward(self, x, mask=None):
        # x: [Batch, Seq, 128]
        # 1. Attention for feature importance
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # 2. Short-term preference (GRU)
        _, h_n = self.aggregation(attn_output)
        gru_vec = h_n.squeeze(0) # [Batch, 128]
        
        # 3. Long-term preference (Residual Mean Pool)
        # 정보 손실 방지를 위해 전체 시퀀스의 평균을 GRU 결과에 더해줌
        mean_pool = torch.mean(x, dim=1) # [Batch, 128]
        
        # 4. Final Aggregation
        user_vec = self.norm(self.fc(gru_vec + mean_pool))
        return F.normalize(user_vec, p=2, dim=1)

class UnifiedDiscoveryModel(nn.Module):
    def __init__(self):
        super(UnifiedDiscoveryModel, self).__init__()
        
        self.caption_proj = MLPProjection(768, 128)
        self.hashtag_proj = MLPProjection(768, 128)
        self.image_proj = MLPProjection(512, 128)
        self.query_proj = MLPProjection(768, 128)
        
        # Fusion MLP (384 -> 128)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
            
        self.user_tower = UserTower()

    def get_multimodal_item_embedding(self, caption_vec, hashtag_vec, image_vec):
        """
        Modality Normalization: 0 벡터는 투영하지 않고 0으로 유지하여 노이즈 방지
        """
        def safe_proj(proj_layer, vec):
            # 벡터가 0이면 결과도 0, 아니면 투영 후 L2 Normalize
            mask = (torch.norm(vec, dim=-1, keepdim=True) > 1e-6).float()
            projected = proj_layer(vec)
            # mask를 곱해 bias와 LayerNorm 효과를 0으로 만듦
            return F.normalize(projected * mask, p=2, dim=-1) * mask

        c_emb = safe_proj(self.caption_proj, caption_vec)
        h_emb = safe_proj(self.hashtag_proj, hashtag_vec)
        i_emb = safe_proj(self.image_proj, image_vec)
        
        # Scale-normalized Concat
        combined = torch.cat([c_emb, h_emb, i_emb], dim=-1)
        
        # LayerNorm(128)을 통과한 fused는 벡터 크기(Norm)가 약 11.3으로 매우 커집니다.
        # 반면 c_emb는 L2 정규화되어 크기가 딱 1.0입니다.
        # 이대로 더하면 텍스트 정체성이 10배 이상 압도당하므로, fused도 크기를 1로 맞춰줍니다.
        fused = F.normalize(self.fusion_mlp(combined), p=2, dim=-1)
        
        # 텍스트 정체성(50%)과 멀티모달 문맥(50%)을 1:1로 결합
        final_emb = 0.5 * fused + 0.5 * c_emb 
        
        # ANN 품질을 위해 최종 L2 Normalize
        return F.normalize(final_emb, p=2, dim=-1)

    def get_item_embedding(self, caption_vec, hashtag_vec=None, image_vec=None):
        batch_size = caption_vec.size(0)
        device = caption_vec.device
        if hashtag_vec is None: hashtag_vec = torch.zeros((batch_size, 768), device=device)
        if image_vec is None: image_vec = torch.zeros((batch_size, 512), device=device)
        return self.get_multimodal_item_embedding(caption_vec, hashtag_vec, image_vec)

    def get_query_embedding(self, raw_query_vec):
        """
        검색어는 전용 투영 레이어(query_proj)를 거쳐 128차원으로 변환됩니다.
        """
        projected = self.query_proj(raw_query_vec)
        return F.normalize(projected, p=2, dim=-1)

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
