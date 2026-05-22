import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECTED_DIM = 128

class ItemFusionGate(nn.Module):
    """
    Kept for compatibility, but we now use fusion_mlp for projected spaces.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_vec: torch.Tensor, image_vec: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([text_vec, image_vec], dim=-1))
        return alpha * text_vec + (1.0 - alpha) * image_vec


class UserTower(nn.Module):
    """
    Stateless Exponential Time-Decay Centroid Pooling user tower.
    No learnable parameters.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, 128]
        # u = normalize( sum_{i=1}^N 0.5^{N-i} * E_i )
        norms = torch.norm(x, dim=-1) # [B, seq_len]
        mask = norms > 1e-5 # [B, seq_len]
        
        N = mask.sum(dim=1, keepdim=True) # [B, 1]
        idx = torch.cumsum(mask.long(), dim=1) # [B, seq_len]
        
        # Calculate weights: 0.5^(N - idx) for active items, 0.0 for padding
        weight = torch.pow(0.5, (N - idx).float()) * mask.float() # [B, seq_len]
        
        user_vec = (x * weight.unsqueeze(-1)).sum(dim=1) # [B, 128]
        
        # Avoid NaN on division
        user_norm = torch.norm(user_vec, dim=-1, keepdim=True)
        user_vec = torch.where(user_norm > 1e-5, user_vec / user_norm.clamp(min=1e-5), torch.zeros_like(user_vec))
        return user_vec


class UnifiedDiscoveryModel(nn.Module):
    """
    Unified discovery model projected to 128-dim.
    """
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(768, 128)
        self.image_proj = nn.Linear(512, 128)
        self.fusion_mlp = nn.Linear(256, 128)
        self.user_tower = UserTower()

    def get_item_embedding(
        self,
        text_vec: torch.Tensor,                # [B, 768] SBERT text
        image_vec: torch.Tensor = None,        # [B, 512] CLIP image (optional)
        hashtag_vec: torch.Tensor = None,      # [B, 768] SBERT hashtag (optional)
        num_hashtags: int = 0,                 # number of hashtags for dynamic weighting
    ) -> torch.Tensor:
        # Enrich text with hashtag using dynamic weight based on hashtag count
        # weight = 0.3 + 0.1 * min(num_hashtags, 5) → range [0.3, 0.8]
        if hashtag_vec is not None:
            mask_h = (torch.norm(hashtag_vec, dim=-1, keepdim=True) > 1e-6).float()
            hashtag_weight = 0.3 + 0.1 * min(num_hashtags, 5)
            text_enriched = text_vec + hashtag_weight * hashtag_vec * mask_h
            text_in = F.normalize(text_enriched, p=2, dim=-1)
        else:
            text_in = text_vec

        # Project text to 128d
        proj_text = self.text_proj(text_in)
        proj_text = F.normalize(proj_text, p=2, dim=-1)

        if image_vec is None:
            return proj_text

        # Project image to 128d
        proj_image = self.image_proj(image_vec)
        proj_image = F.normalize(proj_image, p=2, dim=-1)

        mask_t = (torch.norm(text_in, dim=-1, keepdim=True) > 1e-6).float()
        mask_i = (torch.norm(image_vec, dim=-1, keepdim=True) > 1e-6).float()
        
        t = proj_text * mask_t
        i = proj_image * mask_i
        
        # Fuse projected text and image via fusion_mlp
        concat_vec = torch.cat([t, i], dim=-1) # [B, 256]
        fused = self.fusion_mlp(concat_vec) # [B, 128]
        
        has_image = (mask_i.squeeze(-1) > 0) # [B]
        out_vec = torch.where(has_image.unsqueeze(-1), fused, t)
        return F.normalize(out_vec, p=2, dim=-1)

    def get_query_embedding(self, text_vec: torch.Tensor) -> torch.Tensor:
        """Query is SBERT text space (768-dim) -> project and normalize."""
        proj_text = self.text_proj(text_vec)
        return F.normalize(proj_text, p=2, dim=-1)

    def get_user_embedding(self, history_vecs: torch.Tensor) -> torch.Tensor:
        """history_vecs: [B, seq_len, 128]"""
        return self.user_tower(history_vecs)

    def discovery(self, query_vec: torch.Tensor, user_vec: torch.Tensor) -> torch.Tensor:
        query_intensity = torch.norm(query_vec, dim=-1, keepdim=True)
        gated_user = user_vec * (1.0 - torch.tanh(query_intensity))
        combined = query_vec + gated_user + query_vec * gated_user
        return F.normalize(combined, p=2, dim=-1)
