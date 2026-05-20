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
                            # UserTower (Attention + GRU) 통과 — 훈련과 동일한 경로
                            # history_ids는 ASC(오래된 순) → 최신이 뒤쪽 슬롯에 오도록 배치
                            seq_len = 10
                            padded = torch.zeros(1, seq_len, 128, device=self.device)
                            for i, emb in enumerate(hist_embs[-seq_len:]):
                                padded[0, i] = emb
                            user_vec = self.model.get_user_embedding(padded).squeeze(0).cpu().numpy()

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

    async def discover_by_image(self, db: AsyncSession, image_bytes: bytes, user_id: uuid.UUID = None,
                                limit: int = 20, skip: int = 0, use_personalization: bool = True,
                                exclude_history_ids: List[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        이미지 검색 + 개인화 추천 (Redis 고속 512차원 검색 최적화 버전):
        1. 이미지를 CLIP 모델을 통해 512차원 벡터로 변환 및 L2 정규화
        2. Redis HNSW 인덱스의 @image_vector 필드를 통해 512차원 고속 근사 이웃 검색 (Candidate Generation)
        3. 반환된 후보 포스트들을 기반으로 DB에서 상세 벡터를 조회하여 재정렬 (Re-ranking)
        4. 개인화 추천(user_id)이 활성화되어 있다면 사용자의 시각 선호도와 결합하여 정렬
        """
        import torch.nn.functional as F
        from app.ml.vector_store import vector_store
        try:
            # 1. 이미지 CLIP 512차원 벡터 추출 및 정규화
            raw_img_vec = nlp_embedder.embed_image(image_bytes).to(self.device)
            if torch.norm(raw_img_vec) < 1e-6:
                raise ValueError("Generated raw image vector is zero")
            
            raw_img_vec_norm = F.normalize(raw_img_vec, p=2, dim=-1).cpu().numpy()

            # 2. Redis 512차원 HNSW 고속 검색 (후보군 생성)
            k_search = limit + len(exclude_history_ids or []) + skip + 50
            candidate_ids = vector_store.search_knn(raw_img_vec_norm, k=k_search, vector_field="image_vector")
            if not candidate_ids:
                return []

            # 3. 후보군 포스트 데이터만 DB에서 조회
            res = await db.execute(select(PostVector).where(PostVector.post_id.in_(candidate_ids)))
            posts = res.scalars().all()
            if not posts:
                return []

            # 4. 사용자 행동 이력(좋아요 누른 이미지 벡터) 확인 (개인화용)
            user_hist_img_norms = []
            if user_id and use_personalization:
                stmt = text("SELECT post_id FROM interaction.likes WHERE user_id = :uid ORDER BY created_at ASC LIMIT 10")
                res_likes = await db.execute(stmt, {"uid": user_id})
                history_ids = [row[0] for row in res_likes.all()]
                if history_ids:
                    res_hist = await db.execute(select(PostVector).where(PostVector.post_id.in_(history_ids)))
                    hist_posts = res_hist.scalars().all()
                    for hp in hist_posts:
                        hp_vec = torch.from_numpy(np.frombuffer(hp.image_vector, dtype=np.float32).copy()).to(self.device)
                        if torch.norm(hp_vec) > 1e-6:
                            user_hist_img_norms.append(F.normalize(hp_vec, p=2, dim=-1))

            # 5. 코사인 유사도 계산 및 개인 선호도 결합 (재정렬)
            exclude_set = set(str(i) for i in (exclude_history_ids or []))
            scored_posts = []

            for p in posts:
                if str(p.post_id) in exclude_set:
                    continue

                p_img_vec = torch.from_numpy(np.frombuffer(p.image_vector, dtype=np.float32).copy()).to(self.device)
                if torch.norm(p_img_vec) < 1e-6:
                    continue

                p_img_vec_norm = F.normalize(p_img_vec, p=2, dim=-1)

                # (A) 비주얼 유사도
                visual_sim = torch.dot(torch.from_numpy(raw_img_vec_norm).to(self.device), p_img_vec_norm).item()

                # (B) 유저 선호도
                user_pref_sim = 0.0
                if user_hist_img_norms:
                    prefs = [torch.dot(uh_norm, p_img_vec_norm).item() for uh_norm in user_hist_img_norms]
                    user_pref_sim = max(prefs)

                # 최종 점수 계산 (검색 70%, 개인화 취향 30%)
                final_score = 0.7 * visual_sim + 0.3 * user_pref_sim if user_hist_img_norms else visual_sim
                scored_posts.append((str(p.post_id), final_score))

            # 6. 정렬 및 페이징
            scored_posts.sort(key=lambda x: x[1], reverse=True)
            page = scored_posts[skip: skip + limit]
            return [{"id": pid, "score": score} for pid, score in page]

        except Exception as e:
            logger.error(f"❌ Error in discover_by_image: {e}", exc_info=True)
            # Fallback: 최신순
            stmt = text("SELECT id FROM upload.post WHERE is_deleted = FALSE ORDER BY created_at DESC LIMIT :l OFFSET :s")
            res = await db.execute(stmt, {"l": limit, "s": skip})
            return [{"id": str(row[0]), "score": 0.0} for row in res.all()]

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
            # 128차원 투영 퓨전 벡터와 512차원 이미지 벡터 저장
            with torch.no_grad():
                c = torch.from_numpy(caption_vec).to(self.device).unsqueeze(0)
                t = torch.from_numpy(hashtag_vec).to(self.device).unsqueeze(0)
                img = torch.from_numpy(image_vec).to(self.device).unsqueeze(0)
                p_vec_128 = self.model.get_item_embedding(c, t, img).squeeze(0).cpu().numpy()
            
            vector_store.upsert_vector(str(post_id), p_vec_128, image_vector=image_vec, metadata=metadata)
            
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
                    
                p_img_512 = np.frombuffer(p.image_vector, dtype=np.float32).copy()
                vector_store.upsert_vector(str(p.post_id), p_vec_128, image_vector=p_img_512, metadata=p.content_text)
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
            res = await db.execute(select(PostVector))
            search_posts = [p for p in res.scalars().all() if (p.content_text or {}).get("caption")]
            for post in search_posts:
                caption = (post.content_text or {}).get("caption", "")
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
