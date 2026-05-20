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
        검색 + 개인화 추천 (Late Fusion + RRF):
        - SBERT(768-dim)을 이용한 텍스트 검색 랭킹 리스트 생성
        - UserTower(512-dim)을 이용한 개인화 취향 랭킹 리스트 생성
        - RRF(Reciprocal Rank Fusion)를 사용하여 두 랭킹 리스트를 병합
        - 콜드스타트: 인기도 × 시간감쇠 트렌딩 폴백
        """
        from app.ml.vector_store import vector_store
        try:
            rank_lists = []
            k_search = max(200, limit * 3 + skip)

            # 1. 텍스트 검색어 분기 (SBERT 768-dim)
            if query_text:
                with torch.no_grad():
                    sbert_vec = nlp_embedder.embed_text(query_text).cpu().numpy()
                text_ids = vector_store.search_knn(sbert_vec, k=k_search, vector_field="text_vector")
                rank_lists.append([str(pid) for pid in text_ids])

            # 2. 유저 개인화 취향 분기 (UserTower -> CLIP 512-dim)
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
                                h_bytes = v.hashtag_vector
                                h = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0) \
                                    if h_bytes and len(h_bytes) == 512 * 4 else None
                                hist_embs.append(self.model.get_item_embedding(c, img, h).squeeze(0))

                        if hist_embs:
                            seq_len = 10
                            padded = torch.zeros(1, seq_len, 512, device=self.device)
                            for i, emb in enumerate(hist_embs[-seq_len:]):
                                padded[0, i] = emb
                            user_vec = self.model.get_user_embedding(padded).squeeze(0).cpu().numpy()

            if user_vec is not None:
                user_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="vector")
                rank_lists.append([str(pid) for pid in user_ids])

            # RRF 병합
            exclude_set = set(str(i) for i in (exclude_history_ids or []))
            if rank_lists:
                # DB validity check 제거: asyncpg가 UUID list를 ANY(:pids)로 못 받음.
                # Redis는 post 삭제 시 remove_post()로 정리됨 → 신뢰 가능.
                rrf_scores: Dict[str, float] = {}
                k_rrf = 60
                for r_list in rank_lists:
                    for rank, pid in enumerate(r_list):
                        if pid in exclude_set:
                            continue
                        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + (rank + 1))

                sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

                # 부족하면 trending으로 패딩 (interaction.likes 실제 집계 기준)
                needed = skip + limit - len(sorted_items)
                if needed > 0:
                    existing_ids = {pid for pid, _ in sorted_items} | exclude_set
                    pad_stmt = text("""
                        SELECT p.id FROM upload.post p
                        LEFT JOIN (
                            SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                        ) lc ON lc.post_id = p.id
                        WHERE p.is_deleted = FALSE
                        ORDER BY COALESCE(lc.cnt, 0) * 0.7 + COALESCE(p.view_count, 0) * 0.3 DESC,
                                 p.created_at DESC
                        LIMIT :lim
                    """)
                    pad_res = await db.execute(pad_stmt, {"lim": needed + len(existing_ids) + 20})
                    for row in pad_res.all():
                        pid = str(row[0])
                        if pid not in existing_ids:
                            sorted_items.append((pid, 0.0))
                            existing_ids.add(pid)

                # 다양성 확보: 페이지의 25%는 트렌딩/신규 콘텐츠로 채움
                # trending_items가 부족하면 나머지 personalized로 채워 항상 limit개 반환
                explore_slots = max(limit // 4, 3)
                personalized_slice = sorted_items[skip: skip + limit - explore_slots]
                personalized_ids = {pid for pid, _ in personalized_slice} | exclude_set

                explore_stmt = text("""
                    SELECT p.id FROM upload.post p
                    LEFT JOIN (
                        SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                    ) lc ON lc.post_id = p.id
                    WHERE p.is_deleted = FALSE
                    ORDER BY COALESCE(lc.cnt, 0) * 0.7 + COALESCE(p.view_count, 0) * 0.3 DESC,
                             p.created_at DESC
                    LIMIT :lim
                """)
                explore_res = await db.execute(explore_stmt, {"lim": explore_slots * 5 + 20})
                trending_items = []
                for row in explore_res.all():
                    pid = str(row[0])
                    if pid not in personalized_ids:
                        trending_items.append((pid, 0.0))
                        if len(trending_items) >= explore_slots:
                            break

                page = personalized_slice + trending_items

                # trending_items가 explore_slots보다 적으면 나머지 personalized로 보충
                if len(page) < limit:
                    page_ids = {pid for pid, _ in page}
                    for item in sorted_items[skip + limit - explore_slots:]:
                        if item[0] not in page_ids:
                            page.append(item)
                            page_ids.add(item[0])
                            if len(page) >= limit:
                                break

                return [{"id": pid, "score": float(score)} for pid, score in page]

            # Fallback: 쿼리도 유저도 없는 경우 → interaction.likes 실집계 기반 트렌딩
            stmt = text("""
                SELECT p.id FROM upload.post p
                LEFT JOIN (
                    SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                ) lc ON lc.post_id = p.id
                WHERE p.is_deleted = FALSE
                ORDER BY COALESCE(lc.cnt, 0) * 0.7 + COALESCE(p.view_count, 0) * 0.3 DESC,
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
        """
        이미지 검색 (Late Fusion + RRF):
        - CLIP image(512-dim) 기반 이미지 검색 랭킹 생성
        - UserTower(512-dim) 기반 개인화 추천 랭킹 생성
        - RRF(Reciprocal Rank Fusion)를 사용하여 두 랭킹 병합
        """
        import torch.nn.functional as F
        from app.ml.vector_store import vector_store
        try:
            raw_img_vec = nlp_embedder.embed_image(image_bytes).to(self.device)
            if torch.norm(raw_img_vec) < 1e-6:
                raise ValueError("Image vector is zero")

            img_norm = F.normalize(raw_img_vec, p=2, dim=-1).cpu().numpy()
            k_search = max(200, limit * 3 + skip)

            rank_lists = []

            # 1. 이미지 검색 랭킹 (pure image_vector 사용으로 시각 유사성 정확도 극대화)
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

                    hist_embs = []
                    with torch.no_grad():
                        for h_id in history_ids:
                            if h_id in p_vectors:
                                v = p_vectors[h_id]
                                cap_bytes = v.caption_vector
                                if len(cap_bytes) == 768 * 4:
                                    meta = v.content_text or {}
                                    mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                                    if not mood:
                                        continue
                                    c = nlp_embedder.embed_text_clip(mood).unsqueeze(0).to(self.device)
                                else:
                                    c = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                img = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).to(self.device).unsqueeze(0)
                                h_bytes = v.hashtag_vector
                                h = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0) \
                                    if h_bytes and len(h_bytes) == 512 * 4 else None
                                hist_embs.append(self.model.get_item_embedding(c, img, h).squeeze(0))

                        if hist_embs:
                            seq_len = 10
                            padded = torch.zeros(1, seq_len, 512, device=self.device)
                            for i, emb in enumerate(hist_embs[-seq_len:]):
                                padded[0, i] = emb
                            user_vec = self.model.get_user_embedding(padded).squeeze(0).cpu().numpy()

            if user_vec is not None:
                user_ids = vector_store.search_knn(user_vec, k=k_search, vector_field="vector")
                rank_lists.append([str(pid) for pid in user_ids])

            # RRF 병합 (DB validity check 제거 — asyncpg UUID array 타입 불일치 버그)
            exclude_set = set(str(i) for i in (exclude_history_ids or []))
            rrf_scores: Dict[str, float] = {}
            k_rrf = 60
            for r_list in rank_lists:
                for rank, pid in enumerate(r_list):
                    if pid in exclude_set:
                        continue
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + (rank + 1))

            sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            needed = skip + limit - len(sorted_items)
            if needed > 0:
                existing_ids = {pid for pid, _ in sorted_items} | exclude_set
                pad_stmt = text("""
                    SELECT p.id FROM upload.post p
                    LEFT JOIN (
                        SELECT post_id, COUNT(*) AS cnt FROM interaction.likes GROUP BY post_id
                    ) lc ON lc.post_id = p.id
                    WHERE p.is_deleted = FALSE
                    ORDER BY COALESCE(lc.cnt, 0) * 0.7 + COALESCE(p.view_count, 0) * 0.3 DESC,
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
                ORDER BY COALESCE(lc.cnt, 0) * 0.7 + COALESCE(p.view_count, 0) * 0.3 DESC,
                         p.created_at DESC
                LIMIT :l OFFSET :s
            """)
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
            # ItemFusionGate(CLIP_text + hashtag, CLIP_image) → 512-dim
            with torch.no_grad():
                c = torch.from_numpy(caption_vec).to(self.device).unsqueeze(0)
                img = torch.from_numpy(image_vec).to(self.device).unsqueeze(0)
                h = torch.from_numpy(hashtag_vec).to(self.device).unsqueeze(0) if hashtag_vec is not None else None
                p_vec = self.model.get_item_embedding(c, img, h).squeeze(0).cpu().numpy()

            # 3. SBERT 768-dim text_vector 계산 및 Redis 저장
            caption = metadata.get("caption", "")
            image_prompt = metadata.get("image_prompt", "")
            music_prompt = metadata.get("music_prompt", "")
            mood_text = f"{caption} {image_prompt} {music_prompt}".strip()
            if mood_text:
                with torch.no_grad():
                    sbert_vec = nlp_embedder.embed_text(mood_text).cpu().numpy()
            else:
                sbert_vec = np.zeros(768, dtype=np.float32)

            vector_store.upsert_vector(str(post_id), p_vec, text_vector=sbert_vec, image_vector=image_vec, metadata=metadata)

        except Exception as e:
            logger.error(f"❌ Indexing failed for {post_id}: {e}")

    async def backfill_all_posts(self, db: AsyncSession):
        """
        DB의 모든 포스트를 Redis Vector Store에 재동기화합니다.
        레거시 SBERT 768-dim caption_vector 감지 시 metadata에서 CLIP 재인코딩.
        동시에 SBERT 768-dim text_vector를 계산하여 Redis에 저장합니다.
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
                        h_bytes = p.hashtag_vector
                        h_t = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).to(self.device).unsqueeze(0) \
                              if h_bytes and len(h_bytes) == 512 * 4 else None
                        p_vec = self.model.get_item_embedding(c_t, img_t, h_t).squeeze(0).cpu().numpy()

                    # SBERT 768-dim text_vector 계산
                    meta = p.content_text or {}
                    caption = meta.get("caption", "")
                    image_prompt = meta.get("image_prompt", "")
                    music_prompt = meta.get("music_prompt", "")
                    mood_text = f"{caption} {image_prompt} {music_prompt}".strip()
                    if mood_text:
                        sbert_vec = nlp_embedder.embed_text(mood_text).cpu().numpy()
                    else:
                        sbert_vec = np.zeros(768, dtype=np.float32)

                    img_vec = np.frombuffer(img_bytes, dtype=np.float32).copy()
                    vector_store.upsert_vector(str(p.post_id), p_vec, text_vector=sbert_vec, image_vector=img_vec, metadata=p.content_text)
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

            # 원본 CLIP 텐서 저장 (target 재계산 + lookup 갱신에 사용)
            raw_clip: Dict[Any, tuple] = {}
            with torch.no_grad():
                for v in all_vectors:
                    cap_bytes = v.caption_vector
                    if len(cap_bytes) == 768 * 4:
                        meta = v.content_text or {}
                        mood = f"{meta.get('caption', '')} {meta.get('image_prompt', '')} {meta.get('music_prompt', '')}".strip()
                        c_t = nlp_embedder.embed_text_clip(mood).unsqueeze(0) if mood else \
                              torch.zeros(1, 512)
                    else:
                        c_t = torch.from_numpy(np.frombuffer(cap_bytes, dtype=np.float32).copy()).unsqueeze(0)
                    img_t = torch.from_numpy(np.frombuffer(v.image_vector, dtype=np.float32).copy()).unsqueeze(0)
                    h_bytes = v.hashtag_vector
                    h_t = torch.from_numpy(np.frombuffer(h_bytes, dtype=np.float32).copy()).unsqueeze(0) \
                          if h_bytes and len(h_bytes) == 512 * 4 else None
                    raw_clip[v.post_id] = (c_t.cpu(), img_t.cpu(), h_t.cpu() if h_t is not None else None)

            def _build_lookup() -> Dict[Any, torch.Tensor]:
                """현재 gate 가중치로 아이템 임베딩 사전 계산 (history 빌딩·hard negative용)."""
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

            # CPU-bound 학습 루프를 스레드에서 실행 → 이벤트 루프 블로킹 방지
            await asyncio.to_thread(
                self._run_training_loop, raw_clip, train_data, _build_lookup, BATCH
            )

            logger.info("✅ [UserTower+ItemGate] Training complete.")
            self.trainer.save_model(settings.MODEL_SAVE_PATH)
            logger.info(f"✅ Total Training Pipeline: {time.time()-total_start:.2f}s")

            # Gate 가중치가 변경됐으므로 Redis 전체 재인덱싱
            logger.info("🔄 Re-indexing all items with updated ItemFusionGate...")
            asyncio.create_task(self.backfill_all_posts(db))

        except Exception as e:
            logger.error(f"❌ Training failure: {e}", exc_info=True)

    def _run_training_loop(self, raw_clip, train_data, build_lookup_fn, BATCH=32):
        """순수 CPU/GPU 연산만 수행 — asyncio.to_thread에서 호출."""
        item_emb_lookup = build_lookup_fn()
        EPOCHS = 50
        # lookup 재빌드 주기: 10→5 에폭으로 단축하여 stale embedding으로 인한 손실 스파이크 감소
        REBUILD_EVERY = 5
        self.model.train()
        for epoch in range(EPOCHS):
            if epoch > 0 and epoch % REBUILD_EVERY == 0:
                item_emb_lookup = build_lookup_fn()

            random.shuffle(train_data)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(train_data), BATCH):
                batch = train_data[i:i + BATCH]
                hist_embs, target_vecs, query_caps = [], [], []
                for hist, target in batch:
                    h_embs = [item_emb_lookup[h_id] for h_id in hist]
                    while len(h_embs) < 10:
                        h_embs.insert(0, torch.zeros(512, device=self.device))
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
                extra_negs = torch.stack(random.sample(neg_candidates, n_hard)).to(self.device) \
                             if n_hard > 0 else None

                loss = self.trainer.train_user_query_step(
                    torch.stack(hist_embs),
                    torch.stack(target_vecs),
                    torch.stack(query_caps),
                    extra_negs,
                )
                total_loss += loss
                n_batches += 1
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                logger.info(f"  Epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.4f} (total={total_loss:.4f}, batches={n_batches})")

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
