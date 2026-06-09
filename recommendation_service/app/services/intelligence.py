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

logger = logging.getLogger(__name__)

class UnifiedIntelligenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cf_item_users = None
        self._cf_cache_ts = 0.0
        self._last_loss_history = []

    async def _fetch_512d_vectors(self, pids: List[str]) -> Dict[str, np.ndarray]:
        """
        Fetch 512-dim text_vector from Redis for a list of post IDs.
        If a vector is missing, fallback to a 512-dim zero vector.
        """
        from app.ml.vector_store import vector_store
        
        def _fetch():
            pipe = vector_store.r.pipeline()
            for pid in pids:
                pipe.hget(f"post:{pid}", "text_vector")
            return pipe.execute()
            
        raw_vecs = await asyncio.to_thread(_fetch)
        embeddings = {}
        for pid, v_bytes in zip(pids, raw_vecs):
            if v_bytes and len(v_bytes) == 512 * 4:
                embeddings[pid] = np.frombuffer(v_bytes, dtype=np.float32).copy()
            else:
                embeddings[pid] = np.zeros(512, dtype=np.float32)
        return embeddings

    async def discover(self, db: AsyncSession, query_text: str = None, user_id: uuid.UUID = None,
                       limit: int = 20, skip: int = 0, use_personalization: bool = True,
                       exclude_history_ids: List[uuid.UUID] = None,
                       history_ids: List[uuid.UUID] = None) -> List[Dict[str, Any]]:
        from app.ml.vector_store import vector_store
        try:
            exclude_set = set(str(i) for i in (exclude_history_ids or []))

            # 1. Pure Query Search Bypass
            if query_text:
                clip_vec = await asyncio.to_thread(nlp_embedder.embed_text_clip, query_text)
                clip_vec_np = clip_vec.cpu().numpy()
                
                text_ids = vector_store.search_knn(clip_vec_np, k=50, vector_field="text_vector")
                image_ids = vector_store.search_knn(clip_vec_np, k=50, vector_field="image_vector")
                
                # Combine using RRF
                rrf_scores: Dict[str, float] = {}
                k_rrf = 60
                for rank, pid in enumerate(text_ids):
                    pid_str = str(pid)
                    if pid_str not in exclude_set:
                        rrf_scores[pid_str] = rrf_scores.get(pid_str, 0.0) + 1.0 / (k_rrf + rank + 1)
                for rank, pid in enumerate(image_ids):
                    pid_str = str(pid)
                    if pid_str not in exclude_set:
                        rrf_scores[pid_str] = rrf_scores.get(pid_str, 0.0) + 1.0 / (k_rrf + rank + 1)
                        
                sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                
                # MMR Reranking on search results
                mmr_candidates = sorted_results[:limit + skip + 50]
                c_ids = [pid for pid, _ in mmr_candidates]
                embeddings = await self._fetch_512d_vectors(c_ids)
                reranked_pids = self._mmr_rerank(mmr_candidates, embeddings, lam=0.7, top_k=limit + skip)
                
                pid_to_score = dict(sorted_results)
                sorted_items_reranked = [(pid, pid_to_score[pid]) for pid in reranked_pids]
                reranked_set = set(reranked_pids)
                for pid, score in sorted_results:
                    if pid not in reranked_set:
                        sorted_items_reranked.append((pid, score))
                        
                results = [{"id": pid, "score": float(score)} for pid, score in sorted_items_reranked]
                return results[skip: skip + limit]

            # If no query_text, perform personalization
            rank_lists = []
            k_search = max(200, limit * 3 + skip)

            # 2. User preference vector assembly via stateless exponential time-decay pooling
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
                                    if len(cap_bytes) == 512 * 4:
                                        c = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        meta = v.content_text or {}
                                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                        if mood:
                                            c = nlp_embedder.embed_text_clip(mood).to(self.device).unsqueeze(0)
                                        else:
                                            c = torch.zeros(1, 512, device=self.device)
                                            
                                    img_bytes = v.image_vector
                                    if img_bytes and len(img_bytes) == 512 * 4:
                                        img = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        img = torch.zeros(1, 512, device=self.device)
                                        
                                    h_bytes = v.hashtag_vector
                                    h = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0) if h_bytes and len(h_bytes) == 512 * 4 else None
                                    
                                    meta = v.content_text or {}
                                    hashtags_str = meta.get("hashtags", "")
                                    num_hashtags = len([t for t in hashtags_str.split(",") if t.strip()]) if hashtags_str else 0
                                    
                                    # Fuse in 512d space directly
                                    if h is not None and torch.norm(h) > 1e-6:
                                        hashtag_weight = 0.3 + 0.1 * min(num_hashtags, 5)
                                        text_enriched = c + hashtag_weight * h
                                        text_in = F.normalize(text_enriched, p=2, dim=-1)
                                    else:
                                        text_in = F.normalize(c, p=2, dim=-1) if torch.norm(c) > 1e-6 else c
                                    
                                    if img is not None and torch.norm(img) > 1e-6:
                                        img_in = F.normalize(img, p=2, dim=-1)
                                        fused = text_in + img_in
                                        item_emb = F.normalize(fused, p=2, dim=-1).squeeze(0)
                                    else:
                                        item_emb = text_in.squeeze(0)
                                        
                                    hist_embs.append(item_emb)

                            if hist_embs:
                                seq_len = len(hist_embs)
                                user_vec = torch.zeros(512, device=self.device)
                                for idx, emb in enumerate(hist_embs):
                                    weight = 0.8 ** (seq_len - 1 - idx)
                                    user_vec += weight * emb
                                norm = torch.norm(user_vec)
                                if norm > 1e-5:
                                    user_vec = user_vec / norm
                                return user_vec.cpu().numpy()
                        return None

                    user_vec = await asyncio.to_thread(_get_user_vector)

            if user_vec is not None:
                user_text_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="text_vector", ef_runtime=400)
                user_image_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="image_vector", ef_runtime=400)
                
                user_rrf: Dict[str, float] = {}
                k_rrf = 60
                for rank, pid in enumerate(user_text_ids):
                    user_rrf[str(pid)] = user_rrf.get(str(pid), 0.0) + 1.0 / (k_rrf + rank + 1)
                for rank, pid in enumerate(user_image_ids):
                    user_rrf[str(pid)] = user_rrf.get(str(pid), 0.0) + 1.0 / (k_rrf + rank + 1)
                
                sorted_user_ids = [pid for pid, _ in sorted(user_rrf.items(), key=lambda x: x[1], reverse=True)]
                rank_lists.append(sorted_user_ids)

            # 3. Item-item Jaccard CF
            if history_ids:
                cf_ids = await self._get_cf_candidates(db, history_ids, k=k_search)
                if cf_ids:
                    rank_lists.append(cf_ids)

            if rank_lists:
                rrf_scores: Dict[str, float] = {}
                k_rrf = 60
                personalization_boost = 2.0 if user_vec is not None else 1.0
                for list_idx, r_list in enumerate(rank_lists):
                    weight = personalization_boost if list_idx == 0 else 1.0
                    for rank, pid in enumerate(r_list):
                        if pid in exclude_set:
                            continue
                        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + weight / (k_rrf + (rank + 1))

                sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

                # MMR reranking to diversify results in 512d CLIP space
                mmr_candidates = sorted_items[:limit + skip + 50]
                c_ids = [pid for pid, _ in mmr_candidates]
                
                embeddings = await self._fetch_512d_vectors(c_ids)
                reranked_pids = self._mmr_rerank(mmr_candidates, embeddings, lam=0.7, top_k=limit + skip)
                
                pid_to_score = dict(sorted_items)
                sorted_items_reranked = [(pid, pid_to_score[pid]) for pid in reranked_pids]
                
                reranked_set = set(reranked_pids)
                for pid, score in sorted_items:
                    if pid not in reranked_set:
                        sorted_items_reranked.append((pid, score))
                
                sorted_items = sorted_items_reranked

                needed = skip + limit - len(sorted_items)
                if needed > 0:
                    existing_ids = {pid for pid, _ in sorted_items} | exclude_set
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

            # B. Cold-start fallback trending
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
                return F.normalize(raw_img_vec, p=2, dim=-1).squeeze(0).cpu().numpy()
            
            img_norm = await asyncio.to_thread(_embed_image)
            k_search = max(200, limit * 3 + skip)

            rank_lists = []

            # 1. 이미지 검색 랭킹: CLIP 이미지 벡터로 image_vector + text_vector 크로스모달 RRF
            # CLIP 공간에서 이미지 임베딩은 텍스트 공간과도 정렬되어 있으므로 두 인덱스 모두 탐색
            img_image_ids = vector_store.search_knn(img_norm, k=k_search, vector_field="image_vector")
            img_text_ids = vector_store.search_knn(img_norm, k=k_search, vector_field="text_vector")

            img_rrf: Dict[str, float] = {}
            _k = 60
            for rank, pid in enumerate(img_image_ids):
                img_rrf[str(pid)] = img_rrf.get(str(pid), 0.0) + 1.0 / (_k + rank + 1)
            for rank, pid in enumerate(img_text_ids):
                img_rrf[str(pid)] = img_rrf.get(str(pid), 0.0) + 1.0 / (_k + rank + 1)
            rank_lists.append([pid for pid, _ in sorted(img_rrf.items(), key=lambda x: x[1], reverse=True)])

            # 2. 개인화 랭킹
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
                                    if len(cap_bytes) == 512 * 4:
                                        c = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        meta = v.content_text or {}
                                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                        if mood:
                                            c = nlp_embedder.embed_text_clip(mood).to(self.device).unsqueeze(0)
                                        else:
                                            c = torch.zeros(1, 512, device=self.device)
                                            
                                    img_bytes = v.image_vector
                                    if img_bytes and len(img_bytes) == 512 * 4:
                                        img = torch.from_numpy(np.frombuffer(img_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                    else:
                                        img = torch.zeros(1, 512, device=self.device)
                                        
                                    h_bytes = v.hashtag_vector
                                    h = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0) if h_bytes and len(h_bytes) == 512 * 4 else None
                                        
                                    meta = v.content_text or {}
                                    hashtags_str = meta.get("hashtags", "")
                                    num_hashtags = len([t for t in hashtags_str.split(",") if t.strip()]) if hashtags_str else 0
                                    
                                    # Fuse in 512d space directly
                                    if h is not None and torch.norm(h) > 1e-6:
                                        hashtag_weight = 0.3 + 0.1 * min(num_hashtags, 5)
                                        text_enriched = c + hashtag_weight * h
                                        text_in = F.normalize(text_enriched, p=2, dim=-1)
                                    else:
                                        text_in = F.normalize(c, p=2, dim=-1) if torch.norm(c) > 1e-6 else c
                                    
                                    if img is not None and torch.norm(img) > 1e-6:
                                        img_in = F.normalize(img, p=2, dim=-1)
                                        fused = text_in + img_in
                                        item_emb = F.normalize(fused, p=2, dim=-1).squeeze(0)
                                    else:
                                        item_emb = text_in.squeeze(0)
                                        
                                    hist_embs.append(item_emb)

                            if hist_embs:
                                seq_len = len(hist_embs)
                                user_vec = torch.zeros(512, device=self.device)
                                for idx, emb in enumerate(hist_embs):
                                    weight = 0.8 ** (seq_len - 1 - idx)
                                    user_vec += weight * emb
                                norm = torch.norm(user_vec)
                                if norm > 1e-5:
                                    user_vec = user_vec / norm
                                return user_vec.cpu().numpy()
                        return None

                    user_vec = await asyncio.to_thread(_get_user_vector_image)

            if user_vec is not None:
                user_text_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="text_vector")
                user_image_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="image_vector")
                
                user_rrf: Dict[str, float] = {}
                k_rrf = 60
                for rank, pid in enumerate(user_text_ids):
                    user_rrf[str(pid)] = user_rrf.get(str(pid), 0.0) + 1.0 / (k_rrf + rank + 1)
                for rank, pid in enumerate(user_image_ids):
                    user_rrf[str(pid)] = user_rrf.get(str(pid), 0.0) + 1.0 / (k_rrf + rank + 1)
                
                sorted_user_ids = [pid for pid, _ in sorted(user_rrf.items(), key=lambda x: x[1], reverse=True)]
                rank_lists.append(sorted_user_ids)

            # A. Adaptive RRF merge for image search
            exclude_set = set(str(i) for i in (exclude_history_ids or []))
            rrf_scores: Dict[str, float] = {}
            k_rrf = 60
            for list_idx, r_list in enumerate(rank_lists):
                # 이미지 검색(idx=0)이 주신호, 개인화(idx=1)는 보조 — 반대로 하면 쿼리 이미지 무시됨
                weight = 1.0 if (list_idx == 1 and user_vec is not None) else 2.0
                for rank, pid in enumerate(r_list):
                    if pid in exclude_set:
                        continue
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + weight / (k_rrf + (rank + 1))

            sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            # MMR reranking to diversify results in 512d CLIP space
            mmr_candidates = sorted_items[:limit + skip + 50]
            c_ids = [pid for pid, _ in mmr_candidates]
            
            embeddings = await self._fetch_512d_vectors(c_ids)
            reranked_pids = self._mmr_rerank(mmr_candidates, embeddings, lam=0.7, top_k=limit + skip)
            
            pid_to_score = dict(sorted_items)
            sorted_items_reranked = [(pid, pid_to_score[pid]) for pid in reranked_pids]
            
            reranked_set = set(reranked_pids)
            for pid, score in sorted_items:
                if pid not in reranked_set:
                    sorted_items_reranked.append((pid, score))
            
            sorted_items = sorted_items_reranked

            needed = skip + limit - len(sorted_items)
            if needed > 0:
                existing_ids = {pid for pid, _ in sorted_items} | exclude_set
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

    def _mmr_rerank(
        self,
        candidates: List[tuple[str, float]],
        embeddings: Dict[str, np.ndarray],
        lam: float = 0.7,
        top_k: int = None,
    ) -> List[str]:
        """
        Maximal Marginal Relevance (MMR) reranking.
        lam=0.7 (Relevance: 0.7, Diversity: 0.3)
        """
        if not candidates:
            return []
            
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            dot = np.dot(a, b)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            return dot / (na * nb + 1e-9)

        top_k = top_k or len(candidates)
        remaining = list(candidates)
        selected: List[str] = []

        while remaining and len(selected) < top_k:
            best_pid, best_score = None, float("-inf")
            for pid, rel_score in remaining:
                emb = embeddings.get(pid)
                if emb is None:
                    emb = np.zeros(512)
                
                if not selected:
                    mmr = lam * rel_score
                else:
                    max_sim = max(cosine_sim(emb, embeddings.get(s, np.zeros(512))) for s in selected)
                    mmr = lam * rel_score - (1.0 - lam) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_pid = pid
            if best_pid is None:
                break
            selected.append(best_pid)
            remaining = [(p, s) for p, s in remaining if p != best_pid]

        return selected

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
            vector_store.upsert_vector(str(post_id), text_vector=caption_vec, image_vector=image_vec, metadata=metadata)

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

                    # If legacy 768-dim (SBERT) -> recompute to CLIP 512-dim
                    if len(cap_bytes) != 512 * 4:
                        meta = p.content_text or {}
                        caption = meta.get("caption", "")
                        image_prompt = meta.get("image_prompt", "")
                        music_prompt = meta.get("music_prompt", "")
                        mood_text = f"{caption} {image_prompt} {music_prompt}".strip()
                        if not mood_text:
                            logger.warning(f"⚠️ No text for {p.post_id}, skipping backfill")
                            continue
                        c_t = nlp_embedder.embed_text_clip(mood_text).cpu().numpy()
                        new_cap_bytes = c_t.astype(np.float32).tobytes()
                        await db.execute(
                            sa_update(PostVector)
                            .where(PostVector.post_id == p.post_id)
                            .values(caption_vector=new_cap_bytes)
                        )
                    else:
                        c_t = np.frombuffer(cap_bytes, dtype=np.float32).copy()

                    if img_bytes and len(img_bytes) == 512 * 4:
                        img_t = np.frombuffer(img_bytes, dtype=np.float32).copy()
                    else:
                        # image_vector 없음 → 이미지 검색 품질 저하. Kafka 재소비 또는 /sync 재실행 필요
                        logger.warning(f"⚠️ post {p.post_id} has no valid image_vector — image search will be degraded for this post")
                        img_t = np.zeros(512, dtype=np.float32)

                    # PostgreSQL에서 해시태그 조회하여 metadata에 hashtags 추가
                    tag_stmt = text("""
                        SELECT h.tag 
                        FROM upload.hashtag h
                        JOIN upload.post_hashtag ph ON h.id = ph.hashtag_id
                        WHERE ph.post_id = :post_id
                    """)
                    tag_res = await db.execute(tag_stmt, {"post_id": p.post_id})
                    hashtag_texts = [row[0] for row in tag_res.all()]

                    meta = dict(p.content_text or {})
                    meta["hashtags"] = ",".join(hashtag_texts)

                    vector_store.upsert_vector(str(p.post_id), text_vector=c_t, image_vector=img_t, metadata=meta)
                    count += 1
                except Exception as e:
                    logger.error(f"❌ Backfill failed for {p.post_id}: {e}")

            await db.commit()
            logger.info(f"✅ Backfilled {count}/{len(posts)} posts to Redis.")
        except Exception as e:
            logger.error(f"❌ Backfill failed: {e}")

    async def get_search_suggestions(self, db: AsyncSession, q: str, limit: int = 10) -> List[Dict[str, Any]]:
        from app.ml.vector_store import vector_store
        try:
            # 1. 어휘적 접두사 매칭 (Lexical Match)
            prefix = f"{q}%"
            lex_limit = max(limit * 5, 100)
            lex_stmt = text("""
                SELECT h.tag, COUNT(ph.post_id) as cnt
                FROM upload.hashtag h
                LEFT JOIN upload.post_hashtag ph ON h.id = ph.hashtag_id
                WHERE h.tag ILIKE :prefix
                GROUP BY h.tag
                ORDER BY cnt DESC
                LIMIT :lim
            """)
            lex_res = await db.execute(lex_stmt, {"prefix": prefix, "lim": lex_limit})
            lexical_candidates = [row[0] for row in lex_res.all()]

            # 2. 의미론적 매칭 (Semantic Match)
            q_vec = await asyncio.to_thread(nlp_embedder.embed_text_clip, q)
            q_vec_np = q_vec.cpu().numpy()
            
            # Redis KNN 검색 실행
            discovered_ids = vector_store.search_knn(q_vec_np, k=50, vector_field="text_vector")
            
            # Redis Pipeline 활용하여 각 포스트 ID에 대한 hashtags 가져오기
            def _fetch_hashtags():
                pipe = vector_store.r.pipeline()
                for pid in discovered_ids:
                    pipe.hget(f"post:{pid}", "hashtags")
                return pipe.execute()

            results = await asyncio.to_thread(_fetch_hashtags)
            
            # 랭크 기반 가중치 누적
            sem_scores = {}
            for rank, h_bytes in enumerate(results):
                if not h_bytes:
                    continue
                h_str = h_bytes.decode('utf-8')
                tags = [t.strip() for t in h_str.split(",") if t.strip()]
                weight = 1.0 / (rank + 1)
                for tag in tags:
                    sem_scores[tag] = sem_scores.get(tag, 0.0) + weight

            # 가중치 기준 내림차순 정렬하여 의미론적 매칭 후보 생성
            semantic_candidates = [tag for tag, score in sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)]

            # 3. RRF (Reciprocal Rank Fusion) 융합
            all_tags = set(lexical_candidates) | set(semantic_candidates)
            rrf_scores = {}
            w_lex = 1.5
            w_sem = 1.0
            k_rrf = 60

            for h in all_tags:
                score = 0.0
                if h in lexical_candidates:
                    rank_lex = lexical_candidates.index(h) + 1
                    score += w_lex / (k_rrf + rank_lex)
                if h in semantic_candidates:
                    rank_sem = semantic_candidates.index(h) + 1
                    score += w_sem / (k_rrf + rank_sem)
                rrf_scores[h] = score

            # 융합 결과 정렬 및 중복 제거
            sorted_suggestions = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            suggestions = []
            for tag, score in sorted_suggestions[:limit]:
                is_lex = tag in lexical_candidates
                is_sem = tag in semantic_candidates
                if is_lex and is_sem:
                    source = "hybrid"
                elif is_lex:
                    source = "lexical"
                else:
                    source = "semantic"
                suggestions.append({
                    "keyword": tag,
                    "type": "hashtag",
                    "score": round(score, 4),
                    "source": source
                })
            return suggestions

        except Exception as e:
            logger.error(f"❌ Error in get_search_suggestions: {e}", exc_info=True)
            return []

    async def remove_post(self, post_id: uuid.UUID):
        from app.ml.vector_store import vector_store
        try:
            vector_store.r.delete(f"post:{post_id}")
        except Exception as e:
            logger.warning(f"Failed to remove {post_id} from Redis: {e}")

    async def train_daily_async(self, db: AsyncSession):
        asyncio.create_task(self.train_discovery(db))

    async def train_discovery(self, db: AsyncSession):
        logger.info("🟢 [UserTower] Pure CLIP space is active. Zero-training mode: skipping training.")
        return

    async def evaluate_offline(self, db: AsyncSession):
        try:
            logger.info("📊 [Eval] Zero-training mode active: skipping Phase 1 (training phase).")
            loss_history = []

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
