import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from app.ml.model import UnifiedDiscoveryModel

logger = logging.getLogger(__name__)

class UnifiedDiscoveryTrainer:
    """
    Advanced Trainer featuring Query Simulation and Multi-modal Alignment (CLIP-style)
    """
    def __init__(self, model: UnifiedDiscoveryModel, lr=1e-4, temperature=0.05): # 온도를 0.05로 낮춤 (Harder Matching)
        self.model = model
        # weight_decay 추가로 공간 붕괴 방지
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def info_nce_loss(self, query_user_vecs: torch.Tensor, item_vecs: torch.Tensor):
        """
        InfoNCE Loss: Aligning the Discovery (Query+User) vector with the target Item
        """
        logits = torch.matmul(query_user_vecs, item_vecs.T) / self.temperature
        batch_size = query_user_vecs.size(0)
        labels = torch.arange(batch_size, device=query_user_vecs.device)
        return self.criterion(logits, labels)

    def similarity_preservation_loss(self, projected_vecs: torch.Tensor, raw_vecs: torch.Tensor):
        """
        원본 공간(SBERT)의 유사도 구조를 128차원 공간에서도 유지하도록 함.
        (Cat-Hedgehog 유사도 0.4가 0.97로 붕괴하는 것을 방지)
        """
        if projected_vecs.size(0) < 2: return torch.tensor(0.0).to(projected_vecs.device)
        
        # 1. 원본 공간 유사도 행렬 (Teacher)
        raw_norm = F.normalize(raw_vecs, p=2, dim=-1)
        teacher_sim = torch.matmul(raw_norm, raw_norm.T)
        
        # 2. 투영 공간 유사도 행렬 (Student)
        student_sim = torch.matmul(projected_vecs, projected_vecs.T)
        
        # 3. MSE Loss between similarity matrices
        return F.mse_loss(student_sim, teacher_sim.detach())

    def multimodal_alignment_loss(self, text_embs: torch.Tensor, image_embs: torch.Tensor):
        """
        Aligns image projections with text projections.
        """
        teacher_logits = torch.matmul(text_embs.detach(), text_embs.detach().T) / self.temperature
        student_logits = torch.matmul(image_embs, text_embs.detach().T) / self.temperature
        
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    def train_discovery_step(self, user_histories: torch.Tensor, target_items: Dict[str, torch.Tensor], item_metadata_vecs: torch.Tensor):
        """
        Discovery Training Step with Multi-modal Item support and Query Masking
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = user_histories.size(0)
        
        # 1. Project target items
        target_item_vecs = self.model.get_item_embedding(
            target_items.get("caption"), 
            target_items.get("hashtag"), 
            target_items.get("image")
        )
        
        # 2. Extract User Vector
        user_vecs = self.model.get_user_embedding(user_histories)
        
        # 3. Dynamic Query Masking
        mask = torch.rand(batch_size, 1, device=user_histories.device) > 0.5
        masked_query_raw = item_metadata_vecs * mask.float()
        query_vecs = self.model.get_query_embedding(masked_query_raw)
        
        # 4. Discovery Fusion
        discovery_vecs = self.model.discovery(query_vecs, user_vecs)
        
        # 5. Core Loss: InfoNCE
        loss_discovery = self.info_nce_loss(discovery_vecs, target_item_vecs)
        
        # 6. Structural Loss: Similarity Preservation (공간 붕괴 방지 핵심)
        # 캡션 투영 레이어가 원본 텍스트의 의미 구조를 깨지 않도록 함
        projected_caps = F.normalize(self.model.caption_proj(target_items["caption"]), p=2, dim=-1)
        loss_structure = self.similarity_preservation_loss(projected_caps, target_items["caption"])
        
        # 7. Alignment Loss: Multi-modal (Text-Image)
        with torch.set_grad_enabled(True):
            c_emb = F.normalize(self.model.caption_proj(target_items["caption"]), p=2, dim=-1)
            i_emb = F.normalize(self.model.image_proj(target_items["image"]), p=2, dim=-1)
            loss_alignment = self.multimodal_alignment_loss(c_emb, i_emb)
            
        # 8. Direct Query-Item & Query-Image Alignment (순수 벡터 검색 정확도 향상용)
        # 마스킹되지 않은 진짜 Query가 있을 때만 해당 Loss 추가
        valid_query_mask = mask.squeeze(-1)
        if valid_query_mask.sum() > 0:
            valid_query_vecs = query_vecs[valid_query_mask]
            # 8-1. 순수 검색어와 최종 타겟 아이템 벡터의 직접 일치
            valid_target_vecs = target_item_vecs[valid_query_mask]
            loss_query_item = self.info_nce_loss(valid_query_vecs, valid_target_vecs)
            
            # 8-2. 순수 검색어와 이미지 벡터 간의 멀티모달 직접 일치 
            # 텍스트가 "절대 기준(Anchor)"이 되도록 valid_query_vecs를 detach()하여 
            # 이미지가 텍스트 공간으로 끌려오도록(정렬되도록) 강제합니다.
            valid_image_embs = i_emb[valid_query_mask]
            loss_query_image = self.info_nce_loss(valid_query_vecs.detach(), valid_image_embs)
        else:
            loss_query_item = torch.tensor(0.0).to(user_histories.device)
            loss_query_image = torch.tensor(0.0).to(user_histories.device)
        
        # Total Loss (텍스트를 Anchor로 삼아 이미지를 강력하게 정렬)
        # loss_alignment 가중치를 높여 이미지가 텍스트 공간을 엄격하게 따르게 함
        total_loss = loss_discovery + 1.0 * loss_structure + 2.0 * loss_alignment + 1.0 * loss_query_item + 2.0 * loss_query_image
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def save_model(self, path: str):
        import os
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"💾 Discovery Model saved to {path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model to {path}: {e}")

    def load_model(self, path: str):
        import os
        if not os.path.exists(path):
            logger.warning(f"⚠️ Model file not found at {path}, skipping load.")
            return
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"✅ Discovery Model loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load discovery model from {path}: {e}")
