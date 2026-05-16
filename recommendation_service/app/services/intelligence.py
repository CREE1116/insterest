import asyncio
import httpx
import uuid
import random
import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select

from app.core.config import settings
from app.entities.models import PostVector
from app.ml.model import UnifiedDiscoveryModel
from app.ml.trainer import UnifiedDiscoveryTrainer
from app.ml.nlp import nlp_embedder
import logging

logger = logging.getLogger(__name__)

class UnifiedIntelligenceService:
    def __init__(self):
        # CPU/GPU Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = UnifiedDiscoveryModel()
        self.model.to(self.device)
        self.trainer = UnifiedDiscoveryTrainer(self.model, device=self.device)
        
        # Load weights if exist
        self.trainer.load_model(settings.MODEL_SAVE_PATH)

    async def discover(self, db: AsyncSession, query_text: str = None, user_id: uuid.UUID = None,
                       limit: int = 20, skip: int = 0, use_personalization: bool = True,
                       exclude_history_ids: List[uuid.UUID] = None,
                       history_ids: List[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        검색 + 개인화 추천:
        - 아이템 벡터는 Redis에 미리 저장된 128차원 벡터를 사용 (모델 재계산 없음)
        - 쿼리/유저 벡터만 실시간 계산 후 Redis ANN 검색
        """
        from app.ml.vector_store import vector_store
        try:
            # 1. 검색어 벡터 계산
            query_vec_128 = None
            if query_text:
                raw_vec = nlp_embedder.embed_text(query_text).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    query_vec_128 = self.model.get_query_embedding(raw_vec).squeeze(0).cpu().numpy()

            # 2. 유저 히스토리 → 유저 벡터 계산
            user_vec = None
            if (user_id or history_ids) and use_personalization:
                if not history_ids and user_id:
                    stmt = text("SELECT post_id FROM interaction.likes WHERE user_id = :uid ORDER BY created_at ASC LIMIT 10")
                    res = await db.execute(stmt, {"uid": user_id})
                    history_ids = [row[0] for row in res.all()]

                if history_ids:
                    res = await db.execute(select(PostVector).where(PostVector.post_id.in_(history_ids)))
                    p_vectors = {v.post_id: v for v in res.scalars().all()}

                    hist_embs = []
                    with torch.no_grad():
                        for h_id in history_ids:
                            if h_id in p_vectors:
                                v = p_vectors[h_id]
                                c = torch.from_numpy(np.frombuffer(v.caption_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                t = torch.from_numpy(np.frombuffer(v.hashtag_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                img = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                hist_embs.append(self.model.get_item_embedding(c, t, img).squeeze(0))

                        if hist_embs:
                            # 단순 평균 풀링: 좋아요한 아이템 임베딩의 centroid
                            # UserTower(랜덤 가중치)보다 훨씬 신뢰할 수 있는 유저 벡터
                            stacked = torch.stack(hist_embs)  # [N, 128]
                            user_vec_t = stacked.mean(dim=0)  # centroid
                            user_vec = torch.nn.functional.normalize(user_vec_t, p=2, dim=-1).cpu().numpy()

            # 3. 검색 vs 추천 분리
            if query_vec_128 is not None and user_vec is None:
                # ── 순수 검색: 퓨전 없이 쿼리 벡터 그대로 ANN 검색 ──
                search_vec = query_vec_128

            elif query_vec_128 is not None or user_vec is not None:
                # ── 개인화 추천 (or 검색+추천 혼합): Discovery Fusion ──
                with torch.no_grad():
                    q_t = torch.from_numpy(query_vec_128).to(self.device) if query_vec_128 is not None \
                          else torch.zeros(128, device=self.device)
                    u_t = torch.from_numpy(user_vec).to(self.device) if user_vec is not None \
                          else torch.zeros(128, device=self.device)
                    search_vec = self.model.discovery(q_t.unsqueeze(0), u_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            else:
                search_vec = None

            if search_vec is not None:
                # Redis HNSW ANN 검색
                result_ids = vector_store.search_knn(search_vec, k=limit + len(exclude_history_ids or []) + skip)
                exclude_set = set(str(i) for i in (exclude_history_ids or []))
                filtered = [str(pid) for pid in result_ids if str(pid) not in exclude_set]
                page = filtered[skip: skip + limit]
                return [{"id": pid, "score": 1.0} for pid in page]

            # Fallback: 최신순
            stmt = text("SELECT id FROM upload.post WHERE is_deleted = FALSE ORDER BY created_at DESC LIMIT :l OFFSET :s")
            res = await db.execute(stmt, {"l": limit, "s": skip})
            return [{"id": str(row[0]), "score": 0.0} for row in res.all()]

        except Exception as e:
            logger.error(f"❌ Error in discovery: {e}", exc_info=True)
            return []

    async def index_post(self, db: AsyncSession, post_id: uuid.UUID, 
                         caption_vec: np.ndarray, hashtag_vec: np.ndarray, image_vec: np.ndarray,
                         metadata: Dict[str, Any]):
        """
        포스트 벡터를 DB와 Redis에 동시 인덱싱합니다.
        """
        try:
            from app.ml.vector_store import vector_store
            
            # 1. DB (PostgreSQL) 저장
            pv = PostVector(
                post_id=post_id,
                caption_vector=caption_vec.tobytes(),
                hashtag_vector=hashtag_vec.tobytes(),
                image_vector=image_vec.tobytes(),
                content_text=metadata
            )
            db.add(pv)
            await db.commit()
            
            # 2. Redis Vector Store 저장 (실시간 검색용)
            # 오직 128차원 투영 퓨전 벡터만 Redis에 전송
            with torch.no_grad():
                c = torch.from_numpy(caption_vec).to(self.device).unsqueeze(0)
                t = torch.from_numpy(hashtag_vec).to(self.device).unsqueeze(0)
                img = torch.from_numpy(image_vec).to(self.device).unsqueeze(0)
                p_vec_128 = self.model.get_item_embedding(c, t, img).squeeze(0).cpu().numpy()
            
            vector_store.upsert_vector(str(post_id), p_vec_128, metadata)
            
        except Exception as e:
            logger.error(f"❌ Indexing failed for {post_id}: {e}")

    async def backfill_all_posts(self, db: AsyncSession):
        """
        DB의 모든 포스트를 Redis Vector Store에 재동기화합니다.
        """
        try:
            from app.ml.vector_store import vector_store
            res = await db.execute(select(PostVector))
            posts = res.scalars().all()
            for p in posts:
                with torch.no_grad():
                    c = torch.from_numpy(np.frombuffer(p.caption_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                    t = torch.from_numpy(np.frombuffer(p.hashtag_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                    img = torch.from_numpy(np.frombuffer(p.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                    p_vec_128 = self.model.get_item_embedding(c, t, img).squeeze(0).cpu().numpy()
                    
                vector_store.upsert_vector(str(p.post_id), p_vec_128, p.content_text)
            logger.info(f"✅ Backfilled {len(posts)} posts to Redis.")
        except Exception as e:
            logger.error(f"❌ Backfill failed: {e}")

    async def train_daily_async(self, db: AsyncSession):
        """
        비동기적으로 일일 학습 태스크를 실행합니다.
        """
        asyncio.create_task(self.train_discovery(db))

    async def train_discovery(self, db: AsyncSession):
        """
        2-Phase 학습:
        Phase 1 - 아이템 타워: 캡션+해시태그(anchor) → 이미지 정렬 (CLIP-style)
        Phase 2 - 유저/쿼리 타워: 동결된 아이템 임베딩으로 InfoNCE 학습
        """
        try:
            total_start = time.time()

            # ── 데이터 로드 ──────────────────────────────
            res = await db.execute(text("SELECT user_id, post_id FROM interaction.likes ORDER BY created_at ASC LIMIT 1000"))
            rows = res.all()
            user_sequences: Dict[Any, List] = {}
            for uid, pid in rows:
                user_sequences.setdefault(uid, []).append(pid)

            res = await db.execute(select(PostVector))
            all_vectors = res.scalars().all()
            if not all_vectors:
                logger.warning("⚠️ 포스트 데이터 없음")
                return
            p_vectors = {v.post_id: v for v in all_vectors}

            # ── Phase 1: 아이템 타워 학습 (50 에폭) ─────
            logger.info("🔵 [Phase 1] Item Tower training (caption+hashtag → image alignment)...")
            PHASE1_EPOCHS = 50
            BATCH = 32

            # 전체 포스트 텐서 사전 로드
            all_caps = torch.stack([
                torch.from_numpy(np.frombuffer(v.caption_vector, dtype=np.float32).copy())
                for v in all_vectors
            ]).to(self.device)
            all_tags = torch.stack([
                torch.from_numpy(np.frombuffer(v.hashtag_vector, dtype=np.float32).copy())
                for v in all_vectors
            ]).to(self.device)
            all_imgs = torch.stack([
                torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy())
                for v in all_vectors
            ]).to(self.device)

            n = len(all_vectors)
            for epoch in range(PHASE1_EPOCHS):
                idx = torch.randperm(n)
                total_loss = 0.0
                for i in range(0, n, BATCH):
                    b = idx[i:i+BATCH]
                    loss = self.trainer.train_item_tower_step(
                        all_caps[b], all_tags[b], all_imgs[b]
                    )
                    total_loss += loss
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  [Phase1] Epoch {epoch+1}/{PHASE1_EPOCHS} loss={total_loss:.4f}")

            logger.info("✅ [Phase 1] Item Tower training complete.")

            # ── Phase 2: 유저/쿼리 타워 학습 (좋아요 데이터 있을 때만) ──
            if len(user_sequences) >= 2:
                logger.info("🟢 [Phase 2] User/Query Tower training on FROZEN item embeddings...")

                # 아이템 타워 완전 동결 후 임베딩 한 번에 사전 계산
                item_emb_lookup: Dict[Any, torch.Tensor] = {}
                with torch.no_grad():
                    for idx_v, v in enumerate(all_vectors):
                        emb = self.model.get_item_embedding(
                            all_caps[idx_v].unsqueeze(0),
                            all_tags[idx_v].unsqueeze(0),
                            all_imgs[idx_v].unsqueeze(0)
                        ).squeeze(0).detach()
                        item_emb_lookup[v.post_id] = emb

                train_data = []
                for uid, pids in user_sequences.items():
                    for i in range(1, len(pids)):
                        hist = pids[:i][-10:]
                        target = pids[i]
                        if target in p_vectors and all(h in p_vectors for h in hist):
                            train_data.append((hist, target))

                PHASE2_EPOCHS = 50
                for epoch in range(PHASE2_EPOCHS):
                    random.shuffle(train_data)
                    total_loss = 0.0
                    for i in range(0, len(train_data), BATCH):
                        batch = train_data[i:i+BATCH]
                        hist_embs, target_vecs, query_caps = [], [], []
                        for hist, target in batch:
                            h_embs = [item_emb_lookup[h_id] for h_id in hist]
                            while len(h_embs) < 10:
                                h_embs.insert(0, torch.zeros(128, device=self.device))
                            hist_embs.append(torch.stack(h_embs))
                            target_vecs.append(item_emb_lookup[target])
                            pv_t = p_vectors[target]
                            query_caps.append(torch.from_numpy(
                                np.frombuffer(pv_t.caption_vector, dtype=np.float32).copy()
                            ).to(self.device))

                        loss = self.trainer.train_user_query_step(
                            torch.stack(hist_embs),
                            torch.stack(target_vecs),
                            torch.stack(query_caps),
                        )
                        total_loss += loss
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"  [Phase2] Epoch {epoch+1}/{PHASE2_EPOCHS} loss={total_loss:.4f}")

                logger.info("✅ [Phase 2] User/Query Tower training complete.")
            else:
                logger.info("ℹ️ [Phase 2] Skipped (not enough likes data — search only mode)")

            # ── 모델 저장 + Redis 동기화 ────────────────
            self.trainer.save_model(settings.MODEL_SAVE_PATH)
            logger.info("🔄 [Sync] Backfilling Redis vectors with newly trained item tower...")
            await self.backfill_all_posts(db)
            logger.info(f"✅ Total Training Pipeline: {time.time()-total_start:.2f}s")

        except Exception as e:
            logger.error(f"❌ Training failure: {e}", exc_info=True)

    async def evaluate_offline(self, db: AsyncSession):
        """
        자동화된 오프라인 성능 측정 (NDCG, Recall)
        """
        try:
            logger.info("📊 Starting offline evaluation (NDCG/Recall)...")
            
            # 1. 상호작용 데이터 로드
            res = await db.execute(text("SELECT user_id, post_id FROM interaction.likes ORDER BY created_at"))
            rows = res.all()
            
            user_history = {}
            for uid, pid in rows:
                if uid not in user_history: user_history[uid] = []
                user_history[uid].append(pid)
            
            test_users = {uid: pids for uid, pids in user_history.items() if len(pids) >= 5}
            
            k_list = [5, 10, 20]
            metrics = {k: {"recall": [], "ndcg": []} for k in k_list}
            
            # 2. 추천 정확도 측정
            if test_users:
                for user_id, pids in test_users.items():
                    history = pids[:-1]  # 정답 제외
                    target_id = pids[-1] # 정답
                    
                    raw_reco = await self.discover(
                        db, 
                        user_id=user_id, 
                        limit=50, 
                        history_ids=history[-10:], 
                        exclude_history_ids=history
                    )
                    reco_ids = [str(r["id"]) for r in raw_reco]
                    
                    for k in k_list:
                        top_k = reco_ids[:k]
                        hit = 1.0 if str(target_id) in top_k else 0.0
                        metrics[k]["recall"].append(hit)
                        if str(target_id) in top_k:
                            rank = top_k.index(str(target_id)) + 1
                            metrics[k]["ndcg"].append(1.0 / np.log2(rank + 1))
                        else:
                            metrics[k]["ndcg"].append(0.0)

            # 3. 검색 신뢰도 측정 (Self-Retrieval)
            search_metrics = {k: {"recall": [], "ndcg": []} for k in k_list}
            res = await db.execute(select(PostVector).limit(50))
            for post in res.scalars().all():
                caption = (post.content_text or {}).get("caption", "")
                if not caption: continue
                results = await self.discover(db, query_text=caption, limit=50, use_personalization=False)
                top_ids = [str(r["id"]) for r in results]
                for k in k_list:
                    top_k = top_ids[:k]
                    hit = 1.0 if str(post.post_id) in top_k else 0.0
                    search_metrics[k]["recall"].append(hit)
                    if str(post.post_id) in top_k:
                        rank = top_k.index(str(post.post_id)) + 1
                        search_metrics[k]["ndcg"].append(1.0 / np.log2(rank + 1))
                    else:
                        search_metrics[k]["ndcg"].append(0.0)

            # 4. 요약 리포트 구성
            final_summary = {"recommendation_quality": {}, "search_fidelity": {}}
            for k in k_list:
                final_summary["recommendation_quality"][f"NDCG@{k}"] = float(np.mean(metrics[k]["ndcg"])) if metrics[k]["ndcg"] else 0.0
                final_summary["recommendation_quality"][f"Recall@{k}"] = float(np.mean(metrics[k]["recall"])) if metrics[k]["recall"] else 0.0
                final_summary["search_fidelity"][f"NDCG@{k}"] = float(np.mean(search_metrics[k]["ndcg"])) if search_metrics[k]["ndcg"] else 0.0
                final_summary["search_fidelity"][f"Recall@{k}"] = float(np.mean(search_metrics[k]["recall"])) if search_metrics[k]["recall"] else 0.0
                
            return {
                "status": "success",
                "sample_size_users": len(test_users),
                "sample_size_search": len(search_metrics[k_list[0]]["ndcg"]),
                "metrics": final_summary
            }
        except Exception as e:
            logger.error(f"❌ Evaluation error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


intel_service = UnifiedIntelligenceService()
