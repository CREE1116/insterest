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
                       exclude_history_ids: List[uuid.UUID] = None) -> List[Dict[str, Any]]:
        try:
            final_ids = []
            
            # 1. 텍스트 임베딩 (검색어 있는 경우)
            target_vec_np = None
            if query_text:
                target_vec_np = nlp_embedder.embed_text(query_text).cpu().numpy()

            # 2. 개인화 추천 (유저 ID 있는 경우)
            user_vec = None
            if user_id and use_personalization:
                # 유저 히스토리 로드
                stmt = text("SELECT post_id FROM interaction.likes WHERE user_id = :uid ORDER BY created_at DESC LIMIT 10")
                res = await db.execute(stmt, {"uid": user_id})
                history_ids = [row[0] for row in res.all()]
                
                if history_ids:
                    # 히스토리 임베딩 구성
                    hist_embs = []
                    res = await db.execute(select(PostVector).where(PostVector.post_id.in_(history_ids)))
                    p_vectors = {v.post_id: v for v in res.scalars().all()}
                    
                    with torch.no_grad():
                        for h_id in history_ids:
                            if h_id in p_vectors:
                                v = p_vectors[h_id]
                                c = torch.from_numpy(np.frombuffer(v.caption_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                t = torch.from_numpy(np.frombuffer(v.hashtag_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                img = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                hist_embs.append(self.model.get_item_embedding(c, t, img).squeeze(0))
                        
                        if hist_embs:
                            while len(hist_embs) < 10: hist_embs.insert(0, torch.zeros(128).to(self.device))
                            u_hist = torch.stack(hist_embs).unsqueeze(0)
                            user_vec = self.model.get_user_embedding(u_hist).squeeze(0).cpu().numpy()

            # 3. 벡터 검색 실행 (Fusion Search)
            # 검색어 벡터(target_vec_np)와 유저 취향 벡터(user_vec)를 결합
            query_vec = None
            if target_vec_np is not None and user_vec is not None:
                query_vec = 0.5 * target_vec_np + 0.5 * user_vec # Fusion
            elif target_vec_np is not None:
                query_vec = target_vec_np
            elif user_vec is not None:
                query_vec = user_vec

            if query_vec is not None:
                res = await db.execute(select(PostVector))
                all_posts = res.scalars().all()
                
                scored = []
                for p in all_posts:
                    if exclude_history_ids and p.post_id in exclude_history_ids: continue
                    p_vec = np.frombuffer(p.caption_vector, dtype=np.float32)
                    # 코사인 유사도
                    score = np.dot(query_vec, p_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(p_vec) + 1e-9)
                    scored.append({"id": p.post_id, "score": float(score), "caption": p.content_text.get("caption", "")})
                
                scored.sort(key=lambda x: x["score"], reverse=True)
                return scored[skip:skip+limit]

            # Fallback: 최신순
            stmt = text("SELECT id, caption FROM upload.post WHERE is_deleted = FALSE ORDER BY created_at DESC LIMIT :l OFFSET :s")
            res = await db.execute(stmt, {"l": limit, "s": skip})
            return [{"id": row[0], "score": 0.0, "caption": row[1]} for row in res.all()]

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
            # 텍스트 벡터를 기본 인덱스로 사용
            vector_store.upsert_vector(str(post_id), caption_vec, metadata)
            
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
                vec = np.frombuffer(p.caption_vector, dtype=np.float32)
                vector_store.upsert_vector(str(p.post_id), vec, p.content_text)
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
        성능 최적화 버전: 에폭당 1회 사전 계산 + 메모리 관리
        """
        try:
            total_start = time.time()
            logger.info("🏋️ [Sequence Learning] Starting optimized training pipeline...")
            
            # 1. 상호작용 이력 로드
            t0 = time.time()
            res = await db.execute(text("SELECT user_id, post_id FROM interaction.likes LIMIT 1000"))
            rows = res.all()
            logger.info(f"⏱️ Step 1 (DB Load): {time.time()-t0:.2f}s")
            
            user_sequences = {}
            for uid, pid in rows:
                if uid not in user_sequences: user_sequences[uid] = []
                user_sequences[uid].append(pid)
            
            # 2. 아이템 로드
            res = await db.execute(select(PostVector))
            all_vectors = res.scalars().all()
            p_vectors = {v.post_id: v for v in all_vectors}
            
            if len(user_sequences) < 2:
                logger.warning("⚠️ 학습할 데이터 부족")
                return

            # 3. 데이터셋 구성
            train_data = []
            for uid, pids in user_sequences.items():
                for i in range(1, len(pids)):
                    hist = pids[:i][-10:]
                    target = pids[i]
                    if target in p_vectors and all(h in p_vectors for h in hist):
                        train_data.append((hist, target))

            # 4. 학습 루프
            t2 = time.time()
            epochs = 5
            batch_size = 32
            
            for epoch in range(epochs):
                # 에폭별 임베딩 갱신 (detach로 메모리 보호)
                item_emb_lookup = {}
                with torch.no_grad():
                    for v in all_vectors:
                        c = torch.from_numpy(np.frombuffer(v.caption_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                        t = torch.from_numpy(np.frombuffer(v.hashtag_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                        img = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                        emb = self.model.get_item_embedding(c, t, img).squeeze(0).detach()
                        item_emb_lookup[v.post_id] = emb

                total_loss = 0.0
                random.shuffle(train_data)
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i : i + batch_size]
                    hist_embs, target_caps, target_tags, target_imgs = [], [], [], []
                    
                    for hist, target in batch:
                        h_embs = [item_emb_lookup[h_id] for h_id in hist]
                        while len(h_embs) < 10: h_embs.insert(0, torch.zeros(128).to(self.device))
                        hist_embs.append(torch.stack(h_embs))
                        
                        pv_t = p_vectors[target]
                        target_caps.append(torch.from_numpy(np.frombuffer(pv_t.caption_vector, dtype=np.float32).copy()).to(self.device))
                        target_tags.append(torch.from_numpy(np.frombuffer(pv_t.hashtag_vector, dtype=np.float32).copy()).to(self.device))
                        target_imgs.append(torch.from_numpy(np.frombuffer(pv_t.image_vector, dtype=np.float32).copy()).to(self.device))
                    
                    loss = self.trainer.train_discovery_step(
                        torch.stack(hist_embs), 
                        {"caption": torch.stack(target_caps), "hashtag": torch.stack(target_tags), "image": torch.stack(target_imgs)},
                        torch.stack(target_caps)
                    )
                    total_loss += loss
            
            logger.info(f"⏱️ Step 3 (Training): {time.time()-t2:.2f}s")
            self.trainer.save_model(settings.MODEL_SAVE_PATH)
            logger.info(f"✅ Total Pipeline: {time.time()-total_start:.2f}s")
            
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
                    target_id = pids[-1]
                    raw_reco = await self.discover(db, user_id=user_id, limit=50, exclude_history_ids=[target_id])
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
                caption = post.content_text.get("caption", "")
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
