import redis
import numpy as np
import logging
import uuid
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisVectorStore:
    """
    Real-time persistence for projected 128-dim vectors and raw 512-dim image vectors in Redis
    """
    def __init__(self):
        self.r = redis.Redis(host=settings.REDIS_HOST, port=settings.RECO_REDIS_PORT, decode_responses=False)
        self.index_name = "post_vectors"

    def create_index(self):
        """
        Create HNSW Index for both 128-dim and 512-dim vectors
        """
        try:
            # Drop old index to force schema update
            try:
                self.r.execute_command("FT.DROPINDEX", self.index_name)
                logger.info(f"Dropped old Redis index: {self.index_name}")
            except redis.exceptions.ResponseError:
                pass

            self.r.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", "post:",
                "SCHEMA",
                "post_id", "TAG",
                "vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "128",
                "DISTANCE_METRIC", "COSINE",
                "image_vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "512",
                "DISTANCE_METRIC", "COSINE"
            )
            logger.info(f"✅ Created Multi-Vector Redis HNSW Index: {self.index_name}")
        except redis.exceptions.ResponseError as e:
            logger.error(f"❌ Failed to create Redis index: {e}")

    def upsert_vector(self, post_id: uuid.UUID, vector: np.ndarray, image_vector: np.ndarray = None, metadata: Dict[str, Any] = None):
        """
        128차원 투영 벡터, 512차원 이미지 벡터 및 메타데이터를 Redis에 저장
        """
        key = f"post:{post_id}"
        vector_bytes = vector.astype(np.float32).tobytes()
        
        mapping = {
            "post_id": str(post_id),
            "vector": vector_bytes
        }

        if image_vector is not None:
            mapping["image_vector"] = image_vector.astype(np.float32).tobytes()
        
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    mapping[k] = str(v)
                    
        self.r.hset(key, mapping=mapping)

    def search_knn(self, query_vec: np.ndarray, k: int = 50, vector_field: str = "vector") -> List[uuid.UUID]:
        """
        K-Nearest Neighbors Search using COSINE similarity on a specified vector field ('vector' or 'image_vector')
        """
        query_vec_bytes = query_vec.astype(np.float32).tobytes()
        
        # Redis Vector Search Query
        query = (
            f"*=>[KNN {k} @{vector_field} $query_vec AS score]"
        )
        
        try:
            results = self.r.execute_command(
                "FT.SEARCH", self.index_name, query,
                "PARAMS", "2", "query_vec", query_vec_bytes,
                "SORTBY", "score", "ASC",
                "RETURN", "1", "post_id",
                "LIMIT", "0", str(k),
                "DIALECT", "2"
            )
            
            count = results[0]
            discovered_ids = []
            for i in range(1, len(results), 2):
                fields = results[i+1]
                pid = fields[1].decode('utf-8') if isinstance(fields[1], bytes) else fields[1]
                discovered_ids.append(uuid.UUID(pid))
                
            return discovered_ids
        except Exception as e:
            logger.error(f"❌ Redis search failed on field @{vector_field}: {e}")
            return []

vector_store = RedisVectorStore()
