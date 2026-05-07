import redis
import numpy as np
import logging
import uuid
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisVectorStore:
    """
    Real-time persistence for projected 128-dim vectors in Redis
    """
    def __init__(self):
        self.r = redis.Redis(host=settings.REDIS_HOST, port=settings.RECO_REDIS_PORT, decode_responses=False)
        self.index_name = "post_vectors"
        self.vector_dim = 128 # 128-dim projection space

    def create_index(self):
        """
        Create HNSW Index for 128-dim vectors
        """
        try:
            self.r.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", "post:",
                "SCHEMA",
                "post_id", "TAG",
                "vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self.vector_dim),
                "DISTANCE_METRIC", "COSINE"
            )
            logger.info(f"✅ Created Redis HNSW Index: {self.index_name}")
        except redis.exceptions.ResponseError as e:
            if "Index already exists" in str(e):
                logger.info(f"ℹ️ Redis Index {self.index_name} already exists.")
            else:
                logger.error(f"❌ Failed to create Redis index: {e}")

    def upsert_vector(self, post_id: uuid.UUID, vector: np.ndarray):
        """
        Store projected 128-dim vector in Redis
        """
        key = f"post:{post_id}"
        vector_bytes = vector.astype(np.float32).tobytes()
        
        mapping = {
            "post_id": str(post_id),
            "vector": vector_bytes
        }
        self.r.hset(key, mapping=mapping)

    def search_knn(self, query_vec: np.ndarray, k: int = 50) -> List[uuid.UUID]:
        """
        K-Nearest Neighbors Search using COSINE similarity
        """
        query_vec_bytes = query_vec.astype(np.float32).tobytes()
        
        # Redis Vector Search Query
        # (*)=>[KNN $K @vector $query_vec AS score]
        query = (
            f"*=>[KNN {k} @vector $query_vec AS score]"
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
            
            # results[0] is count, then [key, fields, key, fields, ...]
            count = results[0]
            discovered_ids = []
            for i in range(1, len(results), 2):
                fields = results[i+1]
                # fields is [name, value, name, value, ...]
                pid = fields[1].decode('utf-8') if isinstance(fields[1], bytes) else fields[1]
                discovered_ids.append(uuid.UUID(pid))
                
            return discovered_ids
        except Exception as e:
            logger.error(f"❌ Redis search failed: {e}")
            return []

vector_store = RedisVectorStore()
