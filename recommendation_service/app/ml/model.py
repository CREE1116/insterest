import torch
import torch.nn as nn
import torch.nn.functional as F

UNIFIED_DIM = 512  # Native CLIP ViT-B/32 dimension — text & image natively aligned


class ItemFusionGate(nn.Module):
    """
    Learns the optimal text/image blend ratio per item.
    Input : concat(text_vec [B,512], image_vec [B,512]) → 1024-dim
    Output: alpha [B,1]  →  fused = alpha*text + (1-alpha)*image
    Trained jointly with UserTower via InfoNCE.
    """
    def __init__(self, dim: int = UNIFIED_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_vec: torch.Tensor, image_vec: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([text_vec, image_vec], dim=-1))  # [B, 1]
        return alpha * text_vec + (1.0 - alpha) * image_vec


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

    Item Tower  : ItemFusionGate( CLIP_text + hashtag(0.5w), CLIP_image ) — trained (gate)
    User Tower  : UserTower(Attention + GRU) over item history            — trained
    Query Tower : normalize( CLIP_text(query) )                           — shares CLIP space
    """
    def __init__(self):
        super().__init__()
        self.user_tower = UserTower()
        self.item_gate = ItemFusionGate()

    def get_item_embedding(
        self,
        clip_text_vec: torch.Tensor,           # [B, 512]  CLIP text (mood)
        clip_image_vec: torch.Tensor = None,   # [B, 512]  CLIP image  (optional)
        clip_hashtag_vec: torch.Tensor = None, # [B, 512]  CLIP hashtag (optional, 0.5 weight)
    ) -> torch.Tensor:
        """
        Fuse text (+ hashtag) and image in CLIP's native 512-dim space.
        - Hashtag: fixed 0.5-weight addition to text (Option B)
        - Text/Image balance: learned gate (Option A)
        """
        # Option B: incorporate hashtag with fixed 0.5 weight
        if clip_hashtag_vec is not None:
            mask_h = (torch.norm(clip_hashtag_vec, dim=-1, keepdim=True) > 1e-6).float()
            text_enriched = clip_text_vec + 0.5 * clip_hashtag_vec * mask_h
            text_in = F.normalize(text_enriched, p=2, dim=-1)
        else:
            text_in = clip_text_vec

        if clip_image_vec is None:
            return F.normalize(text_in, p=2, dim=-1)

        mask_t = (torch.norm(text_in,        dim=-1, keepdim=True) > 1e-6).float()
        mask_i = (torch.norm(clip_image_vec, dim=-1, keepdim=True) > 1e-6).float()
        t = text_in * mask_t
        i = clip_image_vec * mask_i

        # Option A: learned gate (text vs. image balance per item)
        has_image = mask_i.squeeze(-1) > 0  # [B]
        gated = self.item_gate(t, i)         # [B, 512]
        # Fall back to text-only for items without an image
        fused = torch.where(has_image.unsqueeze(-1), gated, t)
        return F.normalize(fused, p=2, dim=-1)

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
