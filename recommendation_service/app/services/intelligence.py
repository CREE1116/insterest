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
        - 아이템 벡터는 Redis HNSW 512-dim 인덱스 사용 (모델 재계산 없음)
        - 쿼리/유저 벡터만 실시간 계산 후 ANN 검색
        - 콜드스타트: 인기도 × 시간감쇠 트렌딩 폴백
        """
        from app.ml.vector_store import vector_store
        try:
            # 1. 검색어 벡터 계산 (CLIP text → 512-dim)
            query_vec = None
            if query_text:
                with torch.no_grad():
                    raw_vec = nlp_embedder.embed_text_clip(query_text).unsqueeze(0)
                    query_vec = self.model.get_query_embedding(raw_vec).squeeze(0).cpu().numpy()

            # 2. 유저 히스토리 → UserTower → 512-dim user_vec
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
                                cap_bytes = v.caption_vector
                                # Handle legacy SBERT 768-dim
                                if len(cap_bytes) == 768 * 4:
                                    meta = v.content_text or {}
                                    mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                    if not mood:
                                        continue
                                    c = nlp_embedder.embed_text_clip(mood).unsqueeze(0).to(self.device)
                                else:
                                    c = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                img = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                hist_embs.append(self.model.get_item_embedding(c, img).squeeze(0))

                        if hist_embs:
                            seq_len = 10
                            padded = torch.zeros(1, seq_len, 512, device=self.device)
                            for i, emb in enumerate(hist_embs[-seq_len:]):
                                padded[0, i] = emb
                            user_vec = self.model.get_user_embedding(padded).squeeze(0).cpu().numpy()

            # 3. 검색 / 추천 / 혼합 분기
            if query_vec is not None and user_vec is None:
                search_vec = query_vec
            elif query_vec is not None or user_vec is not None:
                with torch.no_grad():
                    q_t = torch.from_numpy(query_vec).to(self.device) if query_vec is not None \
                          else torch.zeros(512, device=self.device)
                    u_t = torch.from_numpy(user_vec).to(self.device) if user_vec is not None \
                          else torch.zeros(512, device=self.device)
                    search_vec = self.model.discovery(q_t.unsqueeze(0), u_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            else:
                search_vec = None

            if search_vec is not None:
                # Redis HNSW ANN 검색 — 충분한 후보 확보를 위해 2배 오버페치
                k_fetch = (limit + len(exclude_history_ids or []) + skip) * 2 + 20
                result_ids = vector_store.search_knn(search_vec, k=k_fetch)
                exclude_set = set(str(i) for i in (exclude_history_ids or []))
                filtered = [str(pid) for pid in result_ids if str(pid) not in exclude_set]
                page = filtered[skip: skip + limit]
                return [{"id": pid, "score": 1.0} for pid in page]

            # Fallback: 인기도 × 시간감쇠 트렌딩
            stmt = text("""
                SELECT id FROM upload.post
                WHERE is_deleted = FALSE
                ORDER BY (like_count * 0.7 + view_count * 0.3)
                         / POWER(1.0 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0, 1.2) DESC
                LIMIT :l OFFSET :s
            """)
            res = await db.execute(stmt, {"l": limit, "s": skip})
            return [{"id": str(row[0]), "score": 0.0} for row in res.all()]

        except Exception as e:
            logger.error(f"❌ Error in discovery: {e}", exc_info=True)
            return []

    async def discover_by_image(self, db: AsyncSession, image_bytes: bytes, user_id: uuid.UUID = None,
                                limit: int = 20, skip: int = 0, use_personalization: bool = True,
                                exclude_history_ids: List[uuid.UUID] = None) -> List[Dict[str, Any]]:
        """
        이미지 검색: CLIP image → 512-dim → 통합 item 벡터 인덱스 검색.
        item 벡터 = normalize(CLIP_text + CLIP_image) 이므로 이미지 쿼리가 자연스럽게 정렬됨.
        """
        import torch.nn.functional as F
        from app.ml.vector_store import vector_store
        try:
            raw_img_vec = nlp_embedder.embed_image(image_bytes).to(self.device)
            if torch.norm(raw_img_vec) < 1e-6:
                raise ValueError("Image vector is zero")

            # 개인화 있으면 discovery fusion, 없으면 raw image vec
            if user_id and use_personalization:
                results = await self.discover(
                    db, query_text=None, user_id=user_id,
                    limit=limit, skip=skip, use_personalization=True,
                    exclude_history_ids=exclude_history_ids,
                )
                # image 쿼리와 discovery 결과를 블렌딩: image 70% + user 30%
                img_norm = F.normalize(raw_img_vec, p=2, dim=-1).cpu().numpy()
                k_search = (limit + len(exclude_history_ids or []) + skip) * 2 + 20
                img_ids = vector_store.search_knn(img_norm, k=k_search)
                exclude_set = set(str(i) for i in (exclude_history_ids or []))
                img_ids_str = [str(i) for i in img_ids if str(i) not in exclude_set]
                # rank-based blend
                img_rank = {pid: r for r, pid in enumerate(img_ids_str)}
                disc_rank = {r["id"]: r2 for r2, r in enumerate(results)}
                all_ids = list(dict.fromkeys(img_ids_str + [r["id"] for r in results]))
                scored = []
                for pid in all_ids:
                    s_img  = 1.0 / (1 + img_rank.get(pid, 999))
                    s_disc = 1.0 / (1 + disc_rank.get(pid, 999))
                    scored.append((pid, 0.7 * s_img + 0.3 * s_disc))
                scored.sort(key=lambda x: x[1], reverse=True)
                page = scored[skip: skip + limit]
                return [{"id": pid, "score": s} for pid, s in page]
            else:
                img_norm = F.normalize(raw_img_vec, p=2, dim=-1).cpu().numpy()
                k_search = (limit + len(exclude_history_ids or []) + skip) * 2 + 20
                result_ids = vector_store.search_knn(img_norm, k=k_search)
                exclude_set = set(str(i) for i in (exclude_history_ids or []))
                filtered = [str(i) for i in result_ids if str(i) not in exclude_set]
                page = filtered[skip: skip + limit]
                return [{"id": pid, "score": 1.0} for pid in page]

        except Exception as e:
            logger.error(f"❌ Error in discover_by_image: {e}", exc_info=True)
            stmt = text("SELECT id FROM upload.post WHERE is_deleted = FALSE ORDER BY created_at DESC LIMIT :l OFFSET :s")
            res = await db.execute(stmt, {"l": limit, "s": skip})
            return [{"id": str(row[0]), "score": 0.0} for row in res.all()]

    async def index_post(self, db: AsyncSession, post_id: uuid.UUID,
                         caption_vec: np.ndarray, hashtag_vec: np.ndarray, image_vec: np.ndarray,
                         metadata: Dict[str, Any]):
        """
        포스트 벡터를 DB와 Redis에 동시 인덱싱합니다.
        caption_vec: CLIP text 512-dim (mood = caption + image_prompt + music_prompt)
        image_vec  : CLIP image 512-dim
        hashtag_vec: kept for DB schema compat, not used for search
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
            # 512-dim CLIP 통합 아이템 벡터: normalize(CLIP_text + CLIP_image)
            with torch.no_grad():
                c = torch.from_numpy(caption_vec).to(self.device).unsqueeze(0)
                img = torch.from_numpy(image_vec).to(self.device).unsqueeze(0)
                p_vec = self.model.get_item_embedding(c, img).squeeze(0).cpu().numpy()

            vector_store.upsert_vector(str(post_id), p_vec, metadata=metadata)

        except Exception as e:
            logger.error(f"❌ Indexing failed for {post_id}: {e}")

    async def backfill_all_posts(self, db: AsyncSession):
        """
        DB의 모든 포스트를 Redis Vector Store에 재동기화합니다.
        레거시 SBERT 768-dim caption_vector 감지 시 metadata에서 CLIP 재인코딩.
        """
        try:
            from app.ml.vector_store import vector_store
            from sqlalchemy import update as sa_update
            res = await db.execute(select(PostVector))
            posts = res.scalars().all()
            count = 0
            for p in posts:
                try:
                    cap_bytes = p.caption_vector
                    img_bytes = p.image_vector

                    with torch.no_grad():
                        # 768*4=3072 bytes → old SBERT vector → re-encode from metadata
                        if len(cap_bytes) == 768 * 4:
                            meta = p.content_text or {}
                            caption = meta.get("caption", "")
                            image_prompt = meta.get("image_prompt", "")
                            music_prompt = meta.get("music_prompt", "")
                            mood_text = f"{caption} {image_prompt} {music_prompt}".strip()
                            if not mood_text:
                                logger.warning(f"⚠️ No text for {p.post_id}, skipping backfill")
                                continue
                            c_t = nlp_embedder.embed_text_clip(mood_text).unsqueeze(0).to(self.device)
                            # Update DB record to new 512-dim vector
                            new_cap_bytes = c_t.squeeze(0).cpu().numpy().astype(np.float32).tobytes()
                            await db.execute(
                                sa_update(PostVector)
                                .where(PostVector.post_id == p.post_id)
                                .values(caption_vector=new_cap_bytes)
                            )
                        else:
                            c_t = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)

                        img_t = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                        p_vec = self.model.get_item_embedding(c_t, img_t).squeeze(0).cpu().numpy()

                    vector_store.upsert_vector(str(p.post_id), p_vec, metadata=p.content_text)
                    count += 1
                except Exception as e:
                    logger.error(f"❌ Backfill failed for {p.post_id}: {e}")

            await db.commit()
            logger.info(f"✅ Backfilled {count}/{len(posts)} posts to Redis.")
        except Exception as e:
            logger.error(f"❌ Backfill failed: {e}")

    async def remove_post(self, post_id: uuid.UUID):
        """Redis에서 포스트 벡터 삭제."""
        from app.ml.vector_store import vector_store
        try:
            vector_store.r.delete(f"post:{post_id}")
        except Exception as e:
            logger.warning(f"Failed to remove {post_id} from Redis: {e}")

    async def train_daily_async(self, db: AsyncSession):
        """
        비동기적으로 일일 학습 태스크를 실행합니다.
        """
        asyncio.create_task(self.train_discovery(db))

    async def train_discovery(self, db: AsyncSession):
        """
        UserTower InfoNCE 학습 (Single Phase).
        아이템 벡터는 CLIP 고정값 — projection 학습 불필요.
        """
        try:
            total_start = time.time()
            BATCH = 32

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

            if len(user_sequences) < 2:
                logger.info("ℹ️ Skipped (not enough likes data — search only mode)")
                return

            logger.info("🟢 [UserTower] InfoNCE training on CLIP item embeddings...")

            # 아이템 임베딩 사전 계산 (CLIP 고정, 학습 불필요)
            item_emb_lookup: Dict[Any, torch.Tensor] = {}
            with torch.no_grad():
                for v in all_vectors:
                    cap_bytes = v.caption_vector
                    # Handle legacy SBERT 768-dim vectors
                    if len(cap_bytes) == 768 * 4:
                        meta = v.content_text or {}
                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                        c_t = nlp_embedder.embed_text_clip(mood).unsqueeze(0).to(self.device) if mood else \
                              torch.zeros(1, 512, device=self.device)
                    else:
                        c_t = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                    img_t = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                    item_emb_lookup[v.post_id] = self.model.get_item_embedding(c_t, img_t).squeeze(0).detach()

            train_data = []
            for uid, pids in user_sequences.items():
                for i in range(1, len(pids)):
                    hist = pids[:i][-10:]
                    target = pids[i]
                    if target in p_vectors and all(h in p_vectors for h in hist):
                        train_data.append((hist, target))

            EPOCHS = 50
            for epoch in range(EPOCHS):
                random.shuffle(train_data)
                total_loss = 0.0
                for i in range(0, len(train_data), BATCH):
                    batch = train_data[i:i+BATCH]
                    hist_embs, target_vecs, query_caps = [], [], []
                    for hist, target in batch:
                        h_embs = [item_emb_lookup[h_id] for h_id in hist]
                        while len(h_embs) < 10:
                            h_embs.insert(0, torch.zeros(512, device=self.device))
                        hist_embs.append(torch.stack(h_embs))
                        target_vecs.append(item_emb_lookup[target])
                        # query = CLIP text of target item
                        pv_t = p_vectors[target]
                        cap_bytes = pv_t.caption_vector
                        if len(cap_bytes) == 768 * 4:
                            meta = pv_t.content_text or {}
                            mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                            with torch.no_grad():
                                qv = nlp_embedder.embed_text_clip(mood).to(self.device) if mood else \
                                     torch.zeros(512, device=self.device)
                        else:
                            qv = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device)
                        query_caps.append(qv)

                    # Hard negatives: 전체 아이템 풀에서 랜덤 샘플링 (배치 외부 negatives)
                    pool_keys = list(item_emb_lookup.keys())
                    target_set = {target for _, target in batch}
                    neg_candidates = [item_emb_lookup[k] for k in pool_keys if k not in target_set]
                    n_hard = min(64, len(neg_candidates))
                    extra_negs = torch.stack(random.sample(neg_candidates, n_hard)).to(self.device) \
                                 if n_hard > 0 else None

                    loss = self.trainer.train_user_query_step(
                        torch.stack(hist_embs),
                        torch.stack(target_vecs),
                        torch.stack(query_caps),
                        extra_negs,
                    )
                    total_loss += loss
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{EPOCHS} loss={total_loss:.4f}")

            logger.info("✅ [UserTower] Training complete.")
            self.trainer.save_model(settings.MODEL_SAVE_PATH)
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
