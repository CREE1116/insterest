import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from app.ml.model import UnifiedDiscoveryModel

logger = logging.getLogger(__name__)


class UnifiedDiscoveryTrainer:
    """
    Single-phase trainer: UserTower only.
    Item representations are fixed CLIP vectors — no projection training needed.

    Phase 2 (InfoNCE): given user history of item vecs → predict next item vec.
    """
    def __init__(self, model: UnifiedDiscoveryModel, learning_rate: float = 1e-4,
                 device: str = "cpu", temperature: float = 0.07):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            list(model.user_tower.parameters()) + list(model.item_gate.parameters()),
            lr=learning_rate, weight_decay=1e-5
        )

    def info_nce_loss(self, query_vecs: torch.Tensor, item_vecs: torch.Tensor) -> torch.Tensor:
        """InfoNCE: query i must be nearest item i; all other rows are negatives.

        item_vecs may contain extra hard negatives appended after the B positives.
        Labels still point to diagonal (0..B-1).
        """
        logits = torch.matmul(query_vecs, item_vecs.T) / self.temperature
        labels = torch.arange(query_vecs.size(0), device=query_vecs.device)
        return self.criterion(logits, labels)

    def train_step(
        self,
        user_histories: torch.Tensor,            # [B, 10, 512]
        target_item_vecs: torch.Tensor,          # [B, 512]
        query_clip_vecs: torch.Tensor,           # [B, 512]
        extra_negatives: torch.Tensor = None,    # [M, 512] hard negatives from full item pool
    ) -> float:
        """
        UserTower InfoNCE with optional hard negatives.

        Hard negatives (extra_negatives) are appended to the key matrix so the model
        must distinguish the true positive from pool-level lookalikes — not just
        random in-batch items.
        50% query masking forces the user tower to work without an explicit query.
        """
        self.model.train()
        self.optimizer.zero_grad()

        user_vecs = self.model.get_user_embedding(user_histories)           # [B, 512]

        # 50% query masking: forces user tower to work independently of query
        mask = (torch.rand(user_histories.size(0), 1, device=self.device) > 0.5).float()
        masked_query = query_clip_vecs * mask
        query_vecs = self.model.get_query_embedding(masked_query)           # [B, 512]

        discovery_vecs = self.model.discovery(query_vecs, user_vecs)       # [B, 512]

        # Append hard negatives to the key matrix (labels still point to diagonal)
        if extra_negatives is not None and extra_negatives.size(0) > 0:
            key_vecs = torch.cat([target_item_vecs, extra_negatives.detach()], dim=0)
        else:
            key_vecs = target_item_vecs

        loss = self.info_nce_loss(discovery_vecs, key_vecs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def train_user_query_step(self, user_histories, target_item_vecs, caption_vecs_for_query,
                              extra_negatives=None):
        return self.train_step(user_histories, target_item_vecs, caption_vecs_for_query,
                               extra_negatives)

    def save_model(self, path: str):
        import os
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"💾 Model saved to {path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

    def load_model(self, path: str):
        import os
        if not os.path.exists(path):
            logger.warning(f"⚠️ Model file not found at {path}, skipping load.")
            return
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
            logger.info(f"✅ Model loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load model from {path}: {e}")
