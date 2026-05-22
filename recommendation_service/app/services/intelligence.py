import asyncio
import logging
import math
import time
import uuid
import random
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.entities.models import PostVector
from app.ml.nlp import nlp_embedder
from app.ml.model import UnifiedDiscoveryModel
from app.ml.trainer import UnifiedDiscoveryTrainer

logger = logging.getLogger(__name__)

class UnifiedIntelligenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UnifiedDiscoveryModel()
        self.model.to(self.device)
        self.trainer = UnifiedDiscoveryTrainer(self.model, device=self.device)
        self.trainer.load_model(settings.MODEL_SAVE_PATH)
        
        self._cf_item_users = None
        self._cf_cache_ts = 0.0
        self._last_loss_history = []

    async def discover(self, db: AsyncSession, query_text: str = None, user_id: uuid.UUID = None,
                       limit: int = 20, skip: int = 0, use_personalization: bool = True,
                       exclude_history_ids: List[uuid.UUID] = None,
                       history_ids: List[uuid.UUID] = None) -> List[Dict[str, Any]]:
        from app.ml.vector_store import vector_store
        try:
            exclude_set = set(str(i) for i in (exclude_history_ids or []))

            # 1. Pure Query Search Bypass
            if query_text:
                sbert_vec = await asyncio.to_thread(nlp_embedder.embed_text, query_text)
                sbert_vec_np = sbert_vec.cpu().numpy()
                text_ids = vector_store.search_knn(sbert_vec_np, k=limit+skip+50, vector_field="text_vector")
                
                results = []
                for pid in text_ids:
                    pid_str = str(pid)
                    if pid_str not in exclude_set:
                        results.append({"id": pid_str, "score": 1.0})
                return results[skip: skip + limit]

            # If no query_text, perform personalization
            rank_lists = []
            k_search = max(200, limit * 3 + skip)

            # 2. User preference vector assembly via stateless exponential time-decay pooling
            self.model.eval()
            user_vec = None
            if (user_id or history_ids) and use_personalization:
                if not history_ids and user_id:
                    stmt = text("SELECT post_id FROM interaction.likes WHERE user_id = :uid ORDER BY created_at ASC LIMIT 10")
                    res = await db.execute(stmt, {"uid": user_id})
                    history_ids = [row[0] for row in res.all()]

                if history_ids:
                    res = await db.execute(select(PostVector).where(PostVector.post_id.in_(history_ids)))
                    p_vectors = {v.post_id: v for v in res.scalars().all()}

                    def _get_user_vector():
                        hist_embs = []
                        with torch.no_grad():
                            for h_id in history_ids:
                                if h_id in p_vectors:
                                    v = p_vectors[h_id]
                                    cap_bytes = v.caption_vector
                                    # caption_vector in DB is SBERT 768-dim
                                    if len(cap_bytes) == 768 * 4:
                                        c = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        meta = v.content_text or {}
                                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                        if mood:
                                            c = nlp_embedder.embed_text(mood).to(self.device).unsqueeze(0)
                                        else:
                                            c = torch.zeros(1, 768, device=self.device)
                                            
                                    img_bytes = v.image_vector
                                    if len(img_bytes) == 512 * 4:
                                        img = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        img = torch.zeros(1, 512, device=self.device)
                                        
                                    h_bytes = v.hashtag_vector
                                    h = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)                                         if h_bytes and len(h_bytes) == 768 * 4 else None
                                    
                                    hist_embs.append(self.model.get_item_embedding(c, img, h).squeeze(0))

                            if hist_embs:
                                seq_len = len(hist_embs)
                                user_vec = torch.zeros(128, device=self.device)
                                for idx, emb in enumerate(hist_embs):
                                    weight = 0.5 ** (seq_len - 1 - idx)
                                    user_vec += weight * emb
                                norm = torch.norm(user_vec)
                                if norm > 1e-5:
                                    user_vec = user_vec / norm
                                return user_vec.cpu().numpy()
                        return None

                    user_vec = await asyncio.to_thread(_get_user_vector)

            if user_vec is not None:
                # E. ef_runtime=400 for personalization (high-recall mode)
                user_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="vector", ef_runtime=400)
                rank_lists.append([str(pid) for pid in user_ids])

            # 3. Item-item Jaccard CF
            if history_ids:
                cf_ids = await self._get_cf_candidates(db, history_ids, k=k_search)
                if cf_ids:
                    rank_lists.append(cf_ids)

            # A. Adaptive RRF merge
            # When user has history (personalization), double-weight personalization lists.
            # When no personalization available, all lists contribute equally.
            if rank_lists:
                rrf_scores: Dict[str, float] = {}
                k_rrf = 60
                # rank_lists layout: [user_vec_list, cf_list] (no query_text in this branch)
                # Boost personalization signal when history is rich
                personalization_boost = 2.0 if user_vec is not None else 1.0
                for list_idx, r_list in enumerate(rank_lists):
                    # list_idx==0 → user-tower search (personalization)
                    # list_idx==1 → CF candidates
                    weight = personalization_boost if list_idx == 0 else 1.0
                    for rank, pid in enumerate(r_list):
                        if pid in exclude_set:
                            continue
                        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + weight / (k_rrf + (rank + 1))

                sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

                needed = skip + limit - len(sorted_items)
                if needed > 0:
                    existing_ids = {pid for pid, _ in sorted_items} | exclude_set
                    # B. Trending fallback: popularity × exp(-λ × hours_since_upload)
                    pad_stmt = text("""
                        SELECT p.id FROM upload.post p
                        LEFT JOIN (
                            SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                        ) lc ON lc.post_id = p.id
                        WHERE p.is_deleted = FALSE
                        ORDER BY
                            COALESCE(lc.cnt, 0) * EXP(-0.01 *
                                EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 3600.0
                            ) DESC,
                            p.created_at DESC
                        LIMIT :lim
                    """)
                    pad_res = await db.execute(pad_stmt, {"lim": needed + len(existing_ids) + 20})
                    for row in pad_res.all():
                        pid = str(row[0])
                        if pid not in existing_ids:
                            sorted_items.append((pid, 0.0))
                            existing_ids.add(pid)

                page = sorted_items[skip: skip + limit]
                return [{"id": pid, "score": float(score)} for pid, score in page]

            # B. Cold-start fallback trending: popularity × exp(-λ × hours_since_upload)
            stmt = text("""
                SELECT p.id FROM upload.post p
                LEFT JOIN (
                    SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                ) lc ON lc.post_id = p.id
                WHERE p.is_deleted = FALSE
                ORDER BY
                    COALESCE(lc.cnt, 0) * EXP(-0.01 *
                        EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 3600.0
                    ) DESC,
                    p.created_at DESC
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
        from app.ml.vector_store import vector_store
        try:
            def _embed_image():
                raw_img_vec = nlp_embedder.embed_image(image_bytes).to(self.device)
                if torch.norm(raw_img_vec) < 1e-6:
                    raise ValueError("Image vector is zero")
                # Project the 512-dim CLIP image vector to 128-dim using the model's image_proj
                proj_img_vec = self.model.image_proj(raw_img_vec)
                return F.normalize(proj_img_vec, p=2, dim=-1).squeeze(0).cpu().numpy()
            
            img_norm = await asyncio.to_thread(_embed_image)
            k_search = max(200, limit * 3 + skip)

            rank_lists = []

            # 1. 이미지 검색 랭킹 (128-dim projected image_vector)
            img_ids = vector_store.search_knn(img_norm, k=k_search, vector_field="image_vector")
            rank_lists.append([str(pid) for pid in img_ids])

            # 2. 개인화 랭킹 (UserTower)
            user_vec = None
            if user_id and use_personalization:
                stmt = text("SELECT post_id FROM interaction.likes WHERE user_id = :uid ORDER BY created_at ASC LIMIT 10")
                res = await db.execute(stmt, {"uid": user_id})
                history_ids = [row[0] for row in res.all()]

                if history_ids:
                    res = await db.execute(select(PostVector).where(PostVector.post_id.in_(history_ids)))
                    p_vectors = {v.post_id: v for v in res.scalars().all()}

                    def _get_user_vector_image():
                        hist_embs = []
                        with torch.no_grad():
                            for h_id in history_ids:
                                if h_id in p_vectors:
                                    v = p_vectors[h_id]
                                    cap_bytes = v.caption_vector
                                    if len(cap_bytes) == 768 * 4:
                                        c = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        meta = v.content_text or {}
                                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                        if mood:
                                            c = nlp_embedder.embed_text(mood).to(self.device).unsqueeze(0)
                                        else:
                                            c = torch.zeros(1, 768, device=self.device)
                                            
                                    img_bytes = v.image_vector
                                    if len(img_bytes) == 512 * 4:
                                        img = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        img = torch.zeros(1, 512, device=self.device)
                                        
                                    h_bytes = v.hashtag_vector
                                    h = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)                                         if h_bytes and len(h_bytes) == 768 * 4 else None
                                        
                                    hist_embs.append(self.model.get_item_embedding(c, img, h).squeeze(0))

                            if hist_embs:
                                seq_len = len(hist_embs)
                                user_vec = torch.zeros(128, device=self.device)
                                for idx, emb in enumerate(hist_embs):
                                    weight = 0.5 ** (seq_len - 1 - idx)
                                    user_vec += weight * emb
                                norm = torch.norm(user_vec)
                                if norm > 1e-5:
                                    user_vec = user_vec / norm
                                return user_vec.cpu().numpy()
                        return None

                    user_vec = await asyncio.to_thread(_get_user_vector_image)

            if user_vec is not None:
                user_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="vector")
                rank_lists.append([str(pid) for pid in user_ids])

            # A. Adaptive RRF merge for image search
            # rank_lists layout: [image_list, user_personalization_list]
            # Image search gets 1.0 weight; personalization gets 2.0 if available.
            exclude_set = set(str(i) for i in (exclude_history_ids or []))
            rrf_scores: Dict[str, float] = {}
            k_rrf = 60
            for list_idx, r_list in enumerate(rank_lists):
                # list_idx==0 → image search, list_idx==1 → user personalization
                weight = 2.0 if (list_idx == 1 and user_vec is not None) else 1.0
                for rank, pid in enumerate(r_list):
                    if pid in exclude_set:
                        continue
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + weight / (k_rrf + (rank + 1))

            sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            needed = skip + limit - len(sorted_items)
            if needed > 0:
                existing_ids = {pid for pid, _ in sorted_items} | exclude_set
                # B. Trending fallback: popularity × exp(-λ × hours_since_upload)
                pad_stmt = text("""
                    SELECT p.id FROM upload.post p
                    LEFT JOIN (
                        SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                    ) lc ON lc.post_id = p.id
                    WHERE p.is_deleted = FALSE
                    ORDER BY
                        COALESCE(lc.cnt, 0) * EXP(-0.01 *
                            EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 3600.0
                        ) DESC,
                        p.created_at DESC
                    LIMIT :lim
                """)
                pad_res = await db.execute(pad_stmt, {"lim": needed + len(existing_ids) + 20})
                for row in pad_res.all():
                    pid = str(row[0])
                    if pid not in existing_ids:
                        sorted_items.append((pid, 0.0))
                        existing_ids.add(pid)

            page = sorted_items[skip: skip + limit]
            return [{"id": pid, "score": float(score)} for pid, score in page]

        except Exception as e:
            logger.error(f"❌ Error in discover_by_image: {e}", exc_info=True)
            stmt = text("""
                SELECT p.id FROM upload.post p
                LEFT JOIN (
                    SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                ) lc ON lc.post_id = p.id
                WHERE p.is_deleted = FALSE
                ORDER BY
                    COALESCE(lc.cnt, 0) * EXP(-0.01 *
                        EXTRACT(EPOCH FROM (NOW() - p.created_at)) / 3600.0
                    ) DESC,
                    p.created_at DESC
                LIMIT :l OFFSET :s
            """)
            res = await db.execute(stmt, {"l": limit, "s": skip})
            return [{"id": str(row[0]), "score": 0.0} for row in res.all()]

    async def _get_cf_candidates(self, db: AsyncSession, history_ids: List[uuid.UUID], k: int = 200) -> List[str]:
        now = time.time()
        if self._cf_item_users is None or now - self._cf_cache_ts > 300:
            res = await db.execute(text("SELECT post_id, user_id FROM interaction.likes"))
            item_users: Dict[str, set] = {}
            for post_id, user_id in res.all():
                pid = str(post_id)
                item_users.setdefault(pid, set()).add(str(user_id))
            self._cf_item_users = {pid: frozenset(uids) for pid, uids in item_users.items()}
            self._cf_cache_ts = now

        history_set = {str(h) for h in history_ids}
        scores: Dict[str, float] = {}
        for cand_id, cand_users in self._cf_item_users.items():
            if cand_id in history_set or not cand_users:
                continue
            score = 0.0
            for hist_id in history_set:
                hist_users = self._cf_item_users.get(hist_id, frozenset())
                if not hist_users:
                    continue
                intersection = len(hist_users & cand_users)
                if intersection == 0:
                    continue
                score += intersection / len(hist_users | cand_users)
            if score > 0:
                scores[cand_id] = score

        return [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    async def index_post(self, db: AsyncSession, post_id: uuid.UUID,
                         caption_vec: np.ndarray, hashtag_vec: np.ndarray, image_vec: np.ndarray,
                         metadata: Dict[str, Any]):
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
            with torch.no_grad():
                c = torch.from_numpy(caption_vec).to(self.device).unsqueeze(0)
                img = torch.from_numpy(image_vec).to(self.device).unsqueeze(0)
                h = torch.from_numpy(hashtag_vec).to(self.device).unsqueeze(0) if hashtag_vec is not None else None
                p_vec = self.model.get_item_embedding(c, img, h).squeeze(0).cpu().numpy()

            # 3. Project image vector to 128-dim
            with torch.no_grad():
                proj_img_vec = self.model.image_proj(img).squeeze(0).cpu().numpy()

            # SBERT 768-dim text_vector is caption_vec itself
            sbert_vec = caption_vec

            vector_store.upsert_vector(str(post_id), p_vec, text_vector=sbert_vec, image_vector=proj_img_vec, metadata=metadata)

        except Exception as e:
            logger.error(f"❌ Indexing failed for {post_id}: {e}")

    async def backfill_all_posts(self, db: AsyncSession):
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
                        # If legacy 512-dim (or not 768-dim) -> recompute to SBERT 768-dim
                        if len(cap_bytes) != 768 * 4:
                            meta = p.content_text or {}
                            caption = meta.get("caption", "")
                            image_prompt = meta.get("image_prompt", "")
                            music_prompt = meta.get("music_prompt", "")
                            mood_text = f"{caption} {image_prompt} {music_prompt}".strip()
                            if not mood_text:
                                logger.warning(f"⚠️ No text for {p.post_id}, skipping backfill")
                                continue
                            c_t = nlp_embedder.embed_text(mood_text).unsqueeze(0).to(self.device)
                            # Update DB record to new SBERT 768-dim vector
                            new_cap_bytes = c_t.squeeze(0).cpu().numpy().astype(np.float32).tobytes()
                            await db.execute(
                                sa_update(PostVector)
                                .where(PostVector.post_id == p.post_id)
                                .values(caption_vector=new_cap_bytes)
                            )
                        else:
                            c_t = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)

                        if len(img_bytes) == 512 * 4:
                            img_t = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                        else:
                            img_t = torch.zeros(1, 512, device=self.device)

                        h_bytes = p.hashtag_vector
                        h_t = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)                               if h_bytes and len(h_bytes) == 768 * 4 else None

                        p_vec = self.model.get_item_embedding(c_t, img_t, h_t).squeeze(0).cpu().numpy()
                        proj_img_vec = self.model.image_proj(img_t).squeeze(0).cpu().numpy()

                    sbert_vec = c_t.squeeze(0).cpu().numpy()
                    
                    vector_store.upsert_vector(str(p.post_id), p_vec, text_vector=sbert_vec, image_vector=proj_img_vec, metadata=p.content_text)
                    count += 1
                except Exception as e:
                    logger.error(f"❌ Backfill failed for {p.post_id}: {e}")

            await db.commit()
            logger.info(f"✅ Backfilled {count}/{len(posts)} posts to Redis.")
        except Exception as e:
            logger.error(f"❌ Backfill failed: {e}")

    async def remove_post(self, post_id: uuid.UUID):
        from app.ml.vector_store import vector_store
        try:
            vector_store.r.delete(f"post:{post_id}")
        except Exception as e:
            logger.warning(f"Failed to remove {post_id} from Redis: {e}")

    async def train_daily_async(self, db: AsyncSession):
        asyncio.create_task(self.train_discovery(db))

    async def train_discovery(self, db: AsyncSession):
        try:
            total_start = time.time()
            BATCH = 32

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

            logger.info("🟢 [UserTower] Soft CLIP training on SBERT/CLIP item embeddings...")

            raw_clip: Dict[Any, tuple] = {}
            with torch.no_grad():
                for v in all_vectors:
                    cap_bytes = v.caption_vector
                    if len(cap_bytes) != 768 * 4:
                        meta = v.content_text or {}
                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                        c_t = nlp_embedder.embed_text(mood).unsqueeze(0) if mood else                               torch.zeros(1, 768)
                    else:
                        c_t = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).unsqueeze(0)
                    img_t = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).unsqueeze(0)
                    h_bytes = v.hashtag_vector
                    h_t = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).unsqueeze(0)                           if h_bytes and len(h_bytes) == 768 * 4 else None
                    raw_clip[v.post_id] = (c_t.cpu(), img_t.cpu(), h_t.cpu() if h_t is not None else None)

            def _build_lookup() -> Dict[Any, torch.Tensor]:
                lookup = {}
                with torch.no_grad():
                    for pid, (c, img, h) in raw_clip.items():
                        c_d = c.to(self.device)
                        img_d = img.to(self.device)
                        h_d = h.to(self.device) if h is not None else None
                        lookup[pid] = self.model.get_item_embedding(c_d, img_d, h_d).squeeze(0).detach()
                return lookup

            train_data = []
            for uid, pids in user_sequences.items():
                for i in range(1, len(pids)):
                    hist = pids[:i][-10:]
                    target = pids[i]
                    if target in p_vectors and all(h in p_vectors for h in hist):
                        train_data.append((hist, target))

            loss_history = await asyncio.to_thread(
                self._run_training_loop, raw_clip, train_data, _build_lookup, BATCH
            )
            self._last_loss_history = loss_history

            logger.info("✅ [UserTower+Projection] Training complete.")
            self.trainer.save_model(settings.MODEL_SAVE_PATH)
            logger.info(f"✅ Total Training Pipeline: {time.time()-total_start:.2f}s")

            logger.info("🔄 Re-indexing all items with updated projection layers...")
            asyncio.create_task(self.backfill_all_posts(db))

        except Exception as e:
            logger.error(f"❌ Training failure: {e}", exc_info=True)

    def _run_training_loop(self, raw_clip, train_data, build_lookup_fn, BATCH=32) -> list:
        item_emb_lookup = build_lookup_fn()
        EPOCHS = 100
        PATIENCE = 5
        MIN_DELTA = 0.001
        self.model.train()
        loss_history = []

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        best_loss = float("inf")
        patience_counter = 0

        try:
            for epoch in range(EPOCHS):
                random.shuffle(train_data)
                total_loss = 0.0
                n_batches = 0
                for i in range(0, len(train_data), BATCH):
                    batch = train_data[i:i + BATCH]
                    hist_embs, target_vecs, query_caps = [], [], []
                    for hist, target in batch:
                        h_embs = [item_emb_lookup[h_id] for h_id in hist]
                        while len(h_embs) < 10:
                            h_embs.insert(0, torch.zeros(128, device=self.device))
                        hist_embs.append(torch.stack(h_embs))

                        c_raw, img_raw, h_raw = raw_clip[target]
                        c_d = c_raw.to(self.device)
                        img_d = img_raw.to(self.device)
                        h_d = h_raw.to(self.device) if h_raw is not None else None
                        target_vec = self.model.get_item_embedding(c_d, img_d, h_d).squeeze(0)
                        target_vecs.append(target_vec)
                        query_caps.append(c_d.squeeze(0))

                    pool_keys = list(item_emb_lookup.keys())
                    target_set = {target for _, target in batch}
                    neg_candidates = [item_emb_lookup[k] for k in pool_keys if k not in target_set]
                    n_hard = min(64, len(neg_candidates))
                    extra_negs = torch.stack(random.sample(neg_candidates, n_hard)).to(self.device)                                  if n_hard > 0 else None

                    loss = self.trainer.train_user_query_step(
                        torch.stack(hist_embs),
                        torch.stack(target_vecs),
                        torch.stack(query_caps),
                        extra_negs,
                    )
                    total_loss += loss
                    n_batches += 1

                scheduler.step()
                avg_loss = total_loss / max(n_batches, 1)
                loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4)})
                logger.info(f"  Epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

                if avg_loss < best_loss - MIN_DELTA:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        logger.info(f"⏹ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                        break
        finally:
            self.model.eval()

        return loss_history

    async def evaluate_offline(self, db: AsyncSession):
        try:
            logger.info("📊 [Eval] Phase 1/2: Training on current data...")

            loss_history: list = []
            try:
                BATCH = 32
                res = await db.execute(text(
                    "SELECT user_id, post_id FROM interaction.likes ORDER BY created_at ASC LIMIT 1000"
                ))
                rows = res.all()
                user_sequences: Dict[Any, List] = {}
                for uid, pid in rows:
                    user_sequences.setdefault(uid, []).append(pid)

                res2 = await db.execute(select(PostVector))
                all_vectors = res2.scalars().all()
                p_vectors = {v.post_id: v for v in all_vectors}

                if all_vectors and len(user_sequences) >= 1:
                    raw_clip: Dict[Any, tuple] = {}
                    with torch.no_grad():
                        for v in all_vectors:
                            cap_bytes = v.caption_vector
                            if len(cap_bytes) != 768 * 4:
                                meta = v.content_text or {}
                                mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                c_t = nlp_embedder.embed_text(mood).unsqueeze(0) if mood else torch.zeros(1, 768)
                            else:
                                c_t = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).unsqueeze(0)
                            img_t = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).unsqueeze(0)
                            h_bytes = v.hashtag_vector
                            h_t = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).unsqueeze(0)                                   if h_bytes and len(h_bytes) == 768 * 4 else None
                            raw_clip[v.post_id] = (c_t.cpu(), img_t.cpu(), h_t.cpu() if h_t is not None else None)

                    def _eval_build_lookup():
                        lookup = {}
                        with torch.no_grad():
                            for pid, (c, img, h) in raw_clip.items():
                                c_d = c.to(self.device)
                                img_d = img.to(self.device)
                                h_d = h.to(self.device) if h is not None else None
                                lookup[pid] = self.model.get_item_embedding(c_d, img_d, h_d).squeeze(0).detach()
                        return lookup

                    train_data = []
                    for uid, pids in user_sequences.items():
                        for i in range(1, len(pids)):
                            hist = pids[:i][-10:]
                            target = pids[i]
                            if target in p_vectors and all(h in p_vectors for h in hist):
                                train_data.append((hist, target))

                    if len(train_data) >= 2:
                        loss_history = await asyncio.to_thread(
                            self._run_training_loop, raw_clip, train_data, _eval_build_lookup, BATCH
                        )
                        self._last_loss_history = loss_history
                        self.trainer.save_model(settings.MODEL_SAVE_PATH)
                        logger.info(f"✅ [Eval] Phase 1 done. {len(loss_history)} loss points recorded.")
                    else:
                        logger.info("ℹ️ [Eval] Skipped training (insufficient training pairs).")
                else:
                    logger.info("ℹ️ [Eval] Skipped training (no interaction data).")
            except Exception as e:
                logger.warning(f"⚠️ [Eval] Training phase failed, using current model: {e}")

            logger.info("📊 [Eval] Phase 2/2: Measuring NDCG/Recall...")
            
            res = await db.execute(text("SELECT user_id, post_id FROM interaction.likes ORDER BY created_at"))
            rows = res.all()
            
            user_history = {}
            for uid, pid in rows:
                if uid not in user_history: user_history[uid] = []
                user_history[uid].append(pid)
            
            test_users = {uid: pids for uid, pids in user_history.items() if len(pids) >= 5}
            
            k_list = [5, 10, 20]
            metrics = {k: {"recall": [], "ndcg": []} for k in k_list}
            
            if test_users:
                for user_id, pids in test_users.items():
                    history = pids[:-1]
                    target_id = pids[-1]
                    
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
                "metrics": final_summary,
                "training_loss": loss_history,
            }
        except Exception as e:
            logger.error(f"❌ Evaluation error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

intel_service = UnifiedIntelligenceService()
