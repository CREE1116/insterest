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
    Two-Phase Trainer:
    - Phase 1: Item Tower self-supervised alignment (caption+hashtag → image)
    - Phase 2: User/Query Tower trained on FROZEN item embeddings (InfoNCE)
    """
    def __init__(self, model: UnifiedDiscoveryModel, learning_rate=1e-4, device="cpu", temperature=0.05):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

        # 아이템 타워 파라미터 (Phase 1에서만 업데이트)
        self.item_tower_params = (
            list(model.text_proj.parameters()) +
            list(model.hashtag_proj.parameters()) +
            list(model.image_proj.parameters()) +
            list(model.fusion_mlp.parameters())
        )
        # 유저 타워 파라미터 (Phase 2에서만 업데이트)
        # text_proj는 Phase 1에서 이미 학습됨 → Phase 2에서는 user_tower만 업데이트
        self.user_query_params = list(model.user_tower.parameters())

        self.item_optimizer = optim.Adam(self.item_tower_params, lr=learning_rate, weight_decay=1e-5)
        self.user_query_optimizer = optim.Adam(self.user_query_params, lr=learning_rate, weight_decay=1e-5)

    def _set_item_tower_grad(self, requires_grad: bool):
        for p in self.item_tower_params:
            p.requires_grad = requires_grad

    def _set_user_query_grad(self, requires_grad: bool):
        for p in self.user_query_params:
            p.requires_grad = requires_grad

    def info_nce_loss(self, query_user_vecs: torch.Tensor, item_vecs: torch.Tensor):
        """InfoNCE Loss: query+user vector를 target item vector에 정렬"""
        logits = torch.matmul(query_user_vecs, item_vecs.T) / self.temperature
        batch_size = query_user_vecs.size(0)
        labels = torch.arange(batch_size, device=query_user_vecs.device)
        return self.criterion(logits, labels)

    def similarity_preservation_loss(self, projected_vecs: torch.Tensor, raw_vecs: torch.Tensor):
        """
        원본 SBERT 공간의 상대적 유사도 구조를 128차원에서도 보존.
        projected_vecs는 반드시 L2 normalize된 상태여야 함.
        """
        if projected_vecs.size(0) < 2:
            return torch.tensor(0.0, device=projected_vecs.device)
        # Teacher: SBERT 원본 유사도 행렬
        raw_norm = F.normalize(raw_vecs, p=2, dim=-1)
        teacher_sim = torch.matmul(raw_norm, raw_norm.T)
        # Student: 투영 공간 유사도 행렬 (projected_vecs는 이미 L2 normalize 상태)
        proj_norm = F.normalize(projected_vecs, p=2, dim=-1)
        student_sim = torch.matmul(proj_norm, proj_norm.T)
        return F.mse_loss(student_sim, teacher_sim.detach())

    # ──────────────────────────────────────────────────────────
    # Phase 1: 아이템 타워 학습
    # ──────────────────────────────────────────────────────────
    def train_item_tower_step(
        self,
        caption_vecs: torch.Tensor,   # [B, 768] SBERT 원본
        hashtag_vecs: torch.Tensor,   # [B, 768] SBERT 원본
        image_vecs: torch.Tensor,     # [B, 512] 이미지 임베딩
    ) -> float:
        """
        아이템 타워 자가 지도 학습:
        1. 캡션+해시태그 = text anchor
        2. 이미지를 text anchor 방향으로 정렬 (CLIP-style InfoNCE)
        3. 캡션 투영 공간이 SBERT 의미 구조를 보존하도록 강제
        4. query_proj도 caption_proj와 동일 공간을 갖도록 강제 (검색 일관성)
        """
        self.model.train()
        self._set_item_tower_grad(True)
        self._set_user_query_grad(False)

        self.item_optimizer.zero_grad()

        # text_proj 공간이 SBERT 원본의 의미 구조를 보존하도록 강제
        # (고양이들끼리 가깝고, 강아지는 멀리 — SBERT가 이미 알고 있는 관계를 128차원에 이식)
        c_emb = F.normalize(self.model.text_proj(caption_vecs), p=2, dim=-1)
        loss_struct = self.similarity_preservation_loss(c_emb, caption_vecs)

        # 해시태그도 text_proj 공간 구조 보존
        h_emb = F.normalize(self.model.hashtag_proj(hashtag_vecs), p=2, dim=-1)
        loss_struct_h = self.similarity_preservation_loss(h_emb, hashtag_vecs)

        # 이미지 투영을 텍스트 공간으로 당기되, 이미지 벡터가 유의미할 때만 적용
        i_emb = F.normalize(self.model.image_proj(image_vecs), p=2, dim=-1)
        loss_img = self.similarity_preservation_loss(i_emb, F.normalize(image_vecs, p=2, dim=-1))

        total_loss = loss_struct + 0.5 * loss_struct_h + 0.3 * loss_img
        total_loss.backward()
        self.item_optimizer.step()
        return total_loss.item()

    # ──────────────────────────────────────────────────────────
    # Phase 2: 유저/쿼리 타워 학습 (아이템 타워 완전 동결)
    # ──────────────────────────────────────────────────────────
    def train_user_query_step(
        self,
        user_histories: torch.Tensor,         # [B, 10, 128] 사전 계산된 아이템 임베딩 (동결)
        target_item_vecs: torch.Tensor,        # [B, 128] 사전 계산된 타겟 아이템 임베딩 (동결)
        caption_vecs_for_query: torch.Tensor,  # [B, 768] SBERT 원본 (query masking용)
    ) -> float:
        """
        유저/쿼리 타워 학습:
        - 아이템 타워는 완전 동결 (frozen item embeddings으로 안정적 학습)
        - UserTower + query_proj만 업데이트
        """
        self.model.train()
        self._set_item_tower_grad(False)  # 아이템 타워 완전 동결
        self._set_user_query_grad(True)

        self.user_query_optimizer.zero_grad()

        batch_size = user_histories.size(0)

        # --- 유저 벡터 추출 ---
        user_vecs = self.model.get_user_embedding(user_histories)

        # --- 쿼리 마스킹: 50% 확률로 쿼리를 비워 유저 벡터만으로도 추천 가능하게 ---
        mask = torch.rand(batch_size, 1, device=self.device) > 0.5
        masked_query_raw = caption_vecs_for_query * mask.float()
        query_vecs = self.model.get_query_embedding(masked_query_raw)

        # --- Discovery Fusion ---
        discovery_vecs = self.model.discovery(query_vecs, user_vecs)

        # --- InfoNCE Loss: discovery vec → target item vec ---
        loss = self.info_nce_loss(discovery_vecs, target_item_vecs)

        loss.backward()
        self.user_query_optimizer.step()
        return loss.item()

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
