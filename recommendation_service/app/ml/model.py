import torch
import torch.nn as nn
import torch.nn.functional as F

UNIFIED_DIM = 512  # Native CLIP ViT-B/32 dimension — text & image natively aligned


class UserTower(nn.Module):
    """
    Aggregates a sequence of 512-dim item embeddings (user history) into a
    single 512-dim user preference vector via Multihead Attention + GRU.
    """
    def __init__(self, embed_dim: int = UNIFIED_DIM, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.aggregation = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, 512]  — zero-padded where history is shorter than seq_len
        norms = torch.norm(x, dim=-1)                       # [B, seq_len]
        mask = norms < 1e-5                                 # True = padding slot
        all_padded = mask.all(dim=-1, keepdim=True)

        safe_mask = mask.clone()
        safe_mask[all_padded.expand_as(mask)] = False       # avoid NaN on all-zero input

        attn_out, _ = self.attention(x, x, x, key_padding_mask=safe_mask)
        _, h_n = self.aggregation(attn_out)
        gru_vec = h_n.squeeze(0)                            # [B, 512]

        active = (~mask).float().unsqueeze(-1)              # [B, seq_len, 1]
        count = active.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = (x * active).sum(dim=1) / count.squeeze(-1)  # [B, 512]

        user_vec = self.norm(self.fc(gru_vec + mean_pool))
        normed = F.normalize(user_vec, p=2, dim=1)
        return normed * (~all_padded).float()               # cold-start → zero vec


class UnifiedDiscoveryModel(nn.Module):
    """
    CLIP-unified two-tower model.

    Item Tower  : normalize( CLIP_text(mood_text) + CLIP_image(image) )  — no training needed
    User Tower  : UserTower(Attention + GRU) over item history            — trained (Phase 2)
    Query Tower : normalize( CLIP_text(query) )                           — shares CLIP space
    """
    def __init__(self):
        super().__init__()
        self.user_tower = UserTower()

    def get_item_embedding(
        self,
        clip_text_vec: torch.Tensor,          # [B, 512]  CLIP text
        clip_image_vec: torch.Tensor = None,  # [B, 512]  CLIP image  (optional)
    ) -> torch.Tensor:
        """Fuse text + image in CLIP's native space — no learned projection."""
        if clip_image_vec is not None:
            mask_t = (torch.norm(clip_text_vec,  dim=-1, keepdim=True) > 1e-6).float()
            mask_i = (torch.norm(clip_image_vec, dim=-1, keepdim=True) > 1e-6).float()
            fused = clip_text_vec * mask_t + clip_image_vec * mask_i
            return F.normalize(fused, p=2, dim=-1)
        return F.normalize(clip_text_vec, p=2, dim=-1)

    def get_query_embedding(self, clip_text_vec: torch.Tensor) -> torch.Tensor:
        """Query is already in CLIP text space — just normalize."""
        return F.normalize(clip_text_vec, p=2, dim=-1)

    def get_user_embedding(self, history_vecs: torch.Tensor) -> torch.Tensor:
        """history_vecs: [B, seq_len, 512]"""
        return self.user_tower(history_vecs)

    def discovery(self, query_vec: torch.Tensor, user_vec: torch.Tensor) -> torch.Tensor:
        """
        Interaction-based fusion with dynamic gating.
        Strong query → user signal suppressed; pure recommendation → user_vec only.
        """
        query_intensity = torch.norm(query_vec, dim=-1, keepdim=True)
        gated_user = user_vec * (1.0 - torch.tanh(query_intensity))
        combined = query_vec + gated_user + query_vec * gated_user
        return F.normalize(combined, p=2, dim=-1)
