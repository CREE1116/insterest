import torch
import numpy as np
import logging
import httpx
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.ml.model import UnifiedDiscoveryModel
from app.ml.trainer import UnifiedDiscoveryTrainer
from app.ml.vector_store import vector_store
from app.ml.nlp import nlp_embedder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from app.models.models import PostVector
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class UnifiedIntelligenceService:
    """
    Orchestrates Search, Recommendation, and Continuous Learning (128-dim Projection)
    """
    def __init__(self):
        # Load Model
        self.model = UnifiedDiscoveryModel()
        self.trainer = UnifiedDiscoveryTrainer(self.model)
        
        # CPU/GPU Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load weights if exist
        self.trainer.load_model(settings.MODEL_SAVE_PATH)

    async def get_user_context(self, user_id: uuid.UUID, db: AsyncSession, exclude_ids: Optional[List[uuid.UUID]] = None) -> Optional[torch.Tensor]:
        """
        사용자의 상호작용 이력을 바탕으로 128차원 취향 벡터를 생성합니다.
        """
        logger.info(f"👤 Fetching context for user: {user_id}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                res = await client.get(f"{settings.INTERACTION_SERVICE_URL}/api/v1/interactions/user/{user_id}")
                if res.status_code != 200:
                    return None
                
                interactions = res.json()
                liked_ids = [uuid.UUID(i["post_id"]) for i in interactions.get("likes", [])]
                saved_ids = [uuid.UUID(i["post_id"]) for i in interactions.get("saves", [])]
                
                all_ids = list(dict.fromkeys(liked_ids + saved_ids))
                # 벤치마크 시 정답 데이터 제외 (Leak 방지)
                if exclude_ids:
                    all_ids = [i for i in all_ids if i not in exclude_ids]
                
                history_ids = all_ids[-20:]
                if not history_ids:
                    return None

                result = await db.execute(select(PostVector).where(PostVector.post_id.in_(history_ids)))
                vectors = result.scalars().all()
                if not vectors:
                    return None

                projected_items = []
                with torch.no_grad():
                    for v in vectors:
                        try:
                            if hasattr(v, 'caption_vector') and v.caption_vector:
                                c_vec = torch.tensor(np.frombuffer(v.caption_vector, dtype=np.float32)).unsqueeze(0).to(self.device)
                                h_vec = torch.tensor(np.frombuffer(v.hashtag_vector, dtype=np.float32)).unsqueeze(0).to(self.device) if v.hashtag_vector else torch.zeros((1, 768)).to(self.device)
                                i_vec = torch.tensor(np.frombuffer(v.image_vector, dtype=np.float32)).unsqueeze(0).to(self.device) if v.image_vector else torch.zeros((1, 512)).to(self.device)
                            elif hasattr(v, 'vector_data') and v.vector_data:
                                c_vec = torch.tensor(np.frombuffer(v.vector_data, dtype=np.float32)).unsqueeze(0).to(self.device)
                                h_vec = torch.zeros((1, 768)).to(self.device)
                                i_vec = torch.zeros((1, 512)).to(self.device)
                            else:
                                continue
                                
                            item_emb = self.model.get_multimodal_item_embedding(c_vec, h_vec, i_vec)
                            projected_items.append(item_emb)
                        except: continue
                
                if not projected_items:
                    return None
                
                history_tensor = torch.cat(projected_items, dim=0).unsqueeze(0)
                user_vec = self.model.get_user_embedding(history_tensor)
                return user_vec
            except Exception as e:
                logger.error(f"Failed to fetch user context for {user_id}: {e}")
                return None

    async def discover(self, db: AsyncSession, user_id: Optional[uuid.UUID] = None, query_text: Optional[str] = None, skip: int = 0, limit: int = 20, query_weight: float = 0.7, use_personalization: bool = True, exclude_history_ids: Optional[List[uuid.UUID]] = None) -> List[uuid.UUID]:
        """
        하이브리드 탐색: 텍스트 매칭(우선) + 벡터 검색을 결합하여 품질을 극대화합니다.
        """
        logger.info(f"🔍 Discovery Hybrid Request: query={query_text}, skip={skip}, limit={limit}, personal={use_personalization}")
        final_ids = []
        
        try:
            # 1. Pure Semantic Vector Search (AI-driven context matching)
            user_vec = None
            if user_id and use_personalization:
                user_vec = await self.get_user_context(user_id, db, exclude_ids=exclude_history_ids)
            
            if user_vec is None:
                user_vec = torch.zeros((1, 128)).to(self.device)

            query_vec = None
            if query_text:
                # 텍스트 검색어를 128차원 벡터로 변환
                raw_query_vec = nlp_embedder.embed_text(query_text).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    query_vec = self.model.get_query_embedding(raw_query_vec)
                    query_vec = query_vec * query_weight
            else:
                query_vec = torch.zeros((1, 128)).to(self.device)

            # Target Vector 생성 (검색어 벡터 + 사용자 취향 벡터 결합)
            with torch.no_grad():
                target_vec = self.model.discovery(query_vec, user_vec)
            
            target_vec_np = target_vec.squeeze(0).cpu().numpy()

            if torch.norm(target_vec).item() > 1e-6:
                # 벡터 스토어에서 가장 유사한 아이템 검색
                search_k = skip + limit + 20
                final_ids = vector_store.search_knn(target_vec_np, k=search_k)
                logger.info(f"🧠 Pure semantic search found {len(final_ids)} ids")

            # 3. Popularity Fallback (최후의 수단)
            if len(final_ids) < (skip + limit):
                needed = (skip + limit) - len(final_ids)
                stmt = text("""
                    SELECT p.id FROM upload.post p WHERE p.is_deleted = FALSE
                    AND NOT (p.id = ANY(:excluded))
                    ORDER BY COALESCE(p.like_count, 0) DESC, p.created_at DESC 
                    LIMIT :l
                """).bindparams(excluded=final_ids if final_ids else [uuid.uuid4()], l=needed)
                
                result = await db.execute(stmt)
                for row in result.all():
                    if row[0] not in final_ids:
                        final_ids.append(row[0])

            # Apply Paging
            return final_ids[skip:skip+limit]

        except Exception as e:
            logger.error(f"❌ Error in hybrid discovery: {e}", exc_info=True)
            try:
                stmt = text("SELECT id FROM upload.post WHERE is_deleted = FALSE ORDER BY created_at DESC OFFSET :s LIMIT :l")
                res = await db.execute(stmt.bindparams(s=skip, l=limit))
                return [row[0] for row in res.all()]
            except: return []

    async def train_daily_async(self, db: AsyncSession):
        asyncio.create_task(self.train_daily(db))

    async def train_daily(self, db: AsyncSession):
        """
        에폭을 늘려 적은 데이터에서도 변별력을 갖도록 학습합니다.
        """
        try:
            logger.info("🏋️ [Advanced Training] Starting Enhanced Discovery Model (Epochs: 50)...")
            result = await db.execute(select(PostVector).limit(1000))
            vectors = result.scalars().all()
            if len(vectors) < 5: return

            post_map = {}
            for v in vectors:
                post_map[v.post_id] = {
                    "c": torch.tensor(np.frombuffer(v.caption_vector, dtype=np.float32)).to(self.device) if v.caption_vector else torch.zeros(768).to(self.device),
                    "h": torch.tensor(np.frombuffer(v.hashtag_vector, dtype=np.float32)).to(self.device) if v.hashtag_vector else torch.zeros(768).to(self.device),
                    "i": torch.tensor(np.frombuffer(v.image_vector, dtype=np.float32)).to(self.device) if v.image_vector else torch.zeros(512).to(self.device)
                }

            epochs = 50
            batch_size = 32
            for epoch in range(epochs):
                epoch_loss = 0.0
                indices = np.random.permutation(len(vectors))
                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    batch_vectors = [vectors[j] for j in batch_idx]
                    caps = torch.stack([post_map[v.post_id]["c"] for v in batch_vectors])
                    tags = torch.stack([post_map[v.post_id]["h"] for v in batch_vectors])
                    imgs = torch.stack([post_map[v.post_id]["i"] for v in batch_vectors])
                    
                    target_items = {"caption": caps, "hashtag": tags, "image": imgs}
                    query_signal = 0.8 * caps + 0.2 * tags 
                    
                    with torch.no_grad():
                        item_embs = self.model.get_item_embedding(caps, tags, imgs)
                        user_histories = item_embs.unsqueeze(1)
                    
                    loss = self.trainer.train_discovery_step(user_histories, target_items, query_signal)
                    epoch_loss += loss

                if (epoch + 1) % 10 == 0:
                    logger.info(f"📁 Epoch [{epoch+1}/{epochs}] - Avg Loss: {epoch_loss/(len(indices)//batch_size+1):.4f}")

            self.trainer.save_model(settings.MODEL_SAVE_PATH)
            logger.info("✅ Enhanced training completed. Triggering automatic sync (backfill)...")
            await self._do_backfill()
            logger.info("✅ Sync completed after training.")
        except Exception as e:
            logger.error(f"❌ Training failure: {e}", exc_info=True)

    async def index_post(self, db: AsyncSession, post_id: uuid.UUID, caption_vec: np.ndarray, hashtag_vec: np.ndarray, image_vec: np.ndarray):
        result = await db.execute(select(PostVector).where(PostVector.post_id == post_id))
        vector_obj = result.scalar_one_or_none()
        if not vector_obj:
            vector_obj = PostVector(post_id=post_id)
            db.add(vector_obj)
        
        vector_obj.caption_vector = caption_vec.tobytes()
        vector_obj.hashtag_vector = hashtag_vec.tobytes()
        vector_obj.image_vector = image_vec.tobytes()
        await db.commit()

        c_tensor = torch.tensor(caption_vec).unsqueeze(0).to(self.device)
        h_tensor = torch.tensor(hashtag_vec).unsqueeze(0).to(self.device)
        i_tensor = torch.tensor(image_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            projected_vec = self.model.get_multimodal_item_embedding(c_tensor, h_tensor, i_tensor)
            projected_vec_np = projected_vec.squeeze(0).cpu().numpy()
        
        vector_store.upsert_vector(post_id, projected_vec_np)
        logger.info(f"🚀 Indexed post {post_id}")

    async def backfill_all_posts(self, db: AsyncSession):
        asyncio.create_task(self._do_backfill())

    async def _do_backfill(self):
        from app.db.session import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    res = await client.get(f"{settings.UPLOAD_SERVICE_URL}/api/v1/upload/content/posts/all")
                    if res.status_code != 200: return
                    posts = res.json()
                    for post in posts:
                        post_id = uuid.UUID(post["id"])
                        caption = post.get("caption") or ""
                        image_prompt = post.get("content", {}).get("generation_meta", {}).get("image_prompt") or ""
                        base_text = f"{caption} {image_prompt}".strip()
                        c_vec = nlp_embedder.embed_text(base_text).cpu().numpy() if base_text else np.zeros(768, dtype=np.float32)
                        hashtag_texts = [h["tag"] if isinstance(h, dict) else str(h) for h in post.get("hashtags", [])]
                        h_vec = torch.mean(nlp_embedder.embed_batch(hashtag_texts), dim=0).cpu().numpy() if hashtag_texts else np.zeros(768, dtype=np.float32)
                        i_vec = np.zeros(512, dtype=np.float32)
                        image_url = next((m["url"] for m in post.get("content", {}).get("media_list", []) if m["type"] == "image"), None)
                        if image_url:
                            try:
                                img_res = await client.get(f"{settings.UPLOAD_SERVICE_URL}{image_url}")
                                if img_res.status_code == 200:
                                    i_vec = nlp_embedder.embed_image(img_res.content).cpu().numpy()
                            except: pass
                        await self.index_post(db, post_id, c_vec, h_vec, i_vec)
                    logger.info("✅ Backfill completed.")
                except Exception as e:
                    logger.error(f"❌ Backfill failed: {e}")

    async def run_quantitative_benchmark(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Recall@K, NDCG@K 지표를 정량적으로 계산합니다. (Data Leak 방지 적용)
        """
        logger.info("📊 Starting Quantitative Benchmark via API...")
        query = text("""
            SELECT user_id, post_id, created_at 
            FROM interaction.likes 
            UNION ALL
            SELECT user_id, post_id, created_at 
            FROM interaction.saves
            ORDER BY created_at ASC
        """)
        result = await db.execute(query)
        rows = result.all()
        
        user_history = {}
        for uid, pid, _ in rows:
            if uid not in user_history: user_history[uid] = []
            if pid not in user_history[uid]: user_history[uid].append(pid)
        
        test_users = {uid: pids for uid, pids in user_history.items() if len(pids) >= 5}
        if not test_users:
            return {"status": "error", "message": "데이터 부족 (활동 5개 이상 유저 없음)"}

        k_list = [10, 20, 50]
        metrics = {k: {"recall": [], "ndcg": []} for k in k_list}
        
        for user_id, pids in test_users.items():
            # Leave-one-out: 마지막 아이템을 정답(target)으로 설정
            target_id = pids[-1]
            try:
                # 1. 정답 아이템을 유저 히스토리에서 제외하고 추천 요청 (No Cheating)
                reco_ids = await self.discover(db, user_id=user_id, limit=50, exclude_history_ids=[target_id])
                
                for k in k_list:
                    top_k = reco_ids[:k]
                    # Recall
                    hit = 1.0 if target_id in top_k else 0.0
                    metrics[k]["recall"].append(hit)
                    # NDCG
                    if target_id in top_k:
                        rank = top_k.index(target_id) + 1
                        metrics[k]["ndcg"].append(1.0 / np.log2(rank + 1))
                    else:
                        metrics[k]["ndcg"].append(0.0)
            except Exception as e:
                logger.error(f"Benchmark error: {e}")

        # 최종 평균 계산
        final_metrics = {}
        for k in k_list:
            final_metrics[f"Recall@{k}"] = float(np.mean(metrics[k]["recall"])) if metrics[k]["recall"] else 0.0
            final_metrics[f"NDCG@{k}"] = float(np.mean(metrics[k]["ndcg"])) if metrics[k]["ndcg"] else 0.0

        return {
            "status": "success",
            "sample_size": len(test_users),
            "metrics": final_metrics
        }

intel_service = UnifiedIntelligenceService()
