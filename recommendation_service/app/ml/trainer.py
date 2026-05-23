import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from app.ml.model import UnifiedDiscoveryModel

logger = logging.getLogger(__name__)

class UnifiedDiscoveryTrainer:
    """
    SBERT-Guided Bidirectional Soft CLIP Loss Trainer.
    Optimizes projection layers (text_proj, image_proj, fusion_mlp).
    """
    def __init__(self, model: UnifiedDiscoveryModel, learning_rate: float = 1e-4,
                 device: str = "cpu", temperature: float = 0.07):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)

        # Optimize trainable parameters in the model (projection layers)
        self.optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate, weight_decay=1e-5
        )

    def train_step(
        self,
        user_histories: torch.Tensor,            # [B, 10, 128]
        target_item_vecs: torch.Tensor,          # [B, 128]
        query_sbert_vecs: torch.Tensor,          # [B, 768] SBERT query
        extra_negatives: torch.Tensor = None,    # [M, 128] hard negatives
        target_image_vecs: torch.Tensor = None,  # [B, 512] CLIP image (optional)
    ) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        user_vecs = self.model.get_user_embedding(user_histories)           # [B, 128]

        # 50% query masking: forces user tower to work independently of query
        mask = (torch.rand(user_histories.size(0), 1, device=self.device) > 0.5).float()
        masked_query = query_sbert_vecs * mask
        query_vecs = self.model.get_query_embedding(masked_query)           # [B, 128]

        discovery_vecs = self.model.discovery(query_vecs, user_vecs)       # [B, 128]

        if extra_negatives is not None and extra_negatives.size(0) > 0:
            key_vecs = torch.cat([target_item_vecs, extra_negatives.detach()], dim=0)
        else:
            key_vecs = target_item_vecs

        # SBERT-Guided Bidirectional Soft CLIP Loss
        # 1. Compute SBERT similarity matrix for query batch
        norm_sbert = F.normalize(query_sbert_vecs, p=2, dim=-1) # [B, 768]
        sbert_sim = torch.matmul(norm_sbert, norm_sbert.T) # [B, B]
        
        # Target distribution (soft labels)
        soft_labels = F.softmax(sbert_sim / self.temperature, dim=-1) # [B, B]

        # 2. Logits for u2i (user to item)
        logits_u2i = torch.matmul(discovery_vecs, key_vecs.T) / self.temperature # [B, B + M]
        
        if key_vecs.size(0) > discovery_vecs.size(0):
            num_negs = key_vecs.size(0) - discovery_vecs.size(0)
            zero_padding = torch.zeros(discovery_vecs.size(0), num_negs, device=discovery_vecs.device)
            soft_labels_u2i = torch.cat([soft_labels, zero_padding], dim=-1)
        else:
            soft_labels_u2i = soft_labels

        log_prob_u2i = F.log_softmax(logits_u2i, dim=-1)
        loss_u2i = F.kl_div(log_prob_u2i, soft_labels_u2i, reduction='batchmean')

        # 3. Logits for i2u (item to user) - only on positive batch elements
        logits_i2u = torch.matmul(target_item_vecs, discovery_vecs.T) / self.temperature # [B, B]
        log_prob_i2u = F.log_softmax(logits_i2u, dim=-1)
        loss_i2u = F.kl_div(log_prob_i2u, soft_labels, reduction='batchmean')

        loss = (loss_u2i + loss_i2u) / 2.0

        # Auxiliary Contrastive InfoNCE Loss to align text and image spaces
        if target_image_vecs is not None:
            image_norms = torch.norm(target_image_vecs, dim=-1)
            valid_mask = image_norms > 1e-5
            if valid_mask.sum() > 1:
                valid_txt = query_sbert_vecs[valid_mask]
                valid_img = target_image_vecs[valid_mask]
                
                proj_text = self.model.text_proj(F.normalize(valid_txt, p=2, dim=-1))
                proj_text = F.normalize(proj_text, p=2, dim=-1)
                
                proj_image = self.model.image_proj(valid_img)
                proj_image = F.normalize(proj_image, p=2, dim=-1)
                
                logits_txt_img = torch.matmul(proj_text, proj_image.T) / self.temperature
                logits_img_txt = logits_txt_img.T
                
                labels = torch.arange(logits_txt_img.size(0), device=self.device)
                loss_txt_img = F.cross_entropy(logits_txt_img, labels)
                loss_img_txt = F.cross_entropy(logits_img_txt, labels)
                
                loss_infonce = (loss_txt_img + loss_img_txt) / 2.0
                loss = loss + 0.1 * loss_infonce

        loss.backward()

        # Scale text projection update gradients by 0.2 to prevent semantic drift
        if hasattr(self.model, "text_proj") and self.model.text_proj.weight.grad is not None:
            self.model.text_proj.weight.grad.data.mul_(0.2)
        if hasattr(self.model, "text_proj") and self.model.text_proj.bias.grad is not None:
            self.model.text_proj.bias.grad.data.mul_(0.2)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def train_user_query_step(self, user_histories, target_item_vecs, caption_vecs_for_query,
                              extra_negatives=None, target_image_vecs=None):
        return self.train_step(user_histories, target_item_vecs, caption_vecs_for_query,
                               extra_negatives, target_image_vecs)

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
        loaded = False
        
        # 1. Primary path loading
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
                logger.info(f"✅ Model loaded from {path}")
                loaded = True
            except Exception as e:
                logger.warning(f"Could not load model from primary path {path}: {e}")

        # 2. Fallback paths loading
        if not loaded:
            fallback_paths = [
                "./discovery_engine.pth",
                "./app/ml/models/discovery_engine.pth",
                "../discovery_engine.pth",
                "/app/ml/models/discovery_engine.pth",
                "/workspace/recommendation_service/discovery_engine.pth"
            ]
            for fallback in fallback_paths:
                if os.path.exists(fallback):
                    try:
                        self.model.load_state_dict(torch.load(fallback, map_location=self.device), strict=False)
                        logger.info(f"✅ Model loaded from fallback path: {fallback}")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Could not load model from fallback {fallback}: {e}")

        if not loaded:
            logger.warning(f"⚠️ Model file not found or readable at {path} or fallbacks, serving with randomized/initial weights.")
