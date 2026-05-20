import redis
import numpy as np
import logging
import uuid
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisVectorStore:
    """
    Dual HNSW index in Redis:
    - vector: 512-dim CLIP unified text+image space
    - text_vector: 768-dim SBERT semantic text space
    """
    def __init__(self):
        self.r = redis.Redis(host=settings.REDIS_HOST, port=settings.RECO_REDIS_PORT, decode_responses=False)
        self.index_name = "post_vectors"

    def create_index(self):
        """인덱스 생성. 이미 올바른 트리플 스키마로 존재하면 그대로 재사용."""
        try:
            # text_vector 와 image_vector 필드가 있는지 확인하여 없는 경우 드롭 후 재생성
            try:
                info = self.r.execute_command("FT.INFO", self.index_name)
                # info is a list of key-value pairs
                has_text_vector = False
                has_image_vector = False
                for x in info:
                    if isinstance(x, bytes):
                        if b"text_vector" in x:
                            has_text_vector = True
                        if b"image_vector" in x:
                            has_image_vector = True
                    elif isinstance(x, str):
                        if "text_vector" in x:
                            has_text_vector = True
                        if "image_vector" in x:
                            has_image_vector = True
                if not has_text_vector or not has_image_vector:
                    logger.warning("⚠️ Redis index doesn't contain 'text_vector' or 'image_vector'. Dropping to migrate to triple HNSW schema...")
                    self.r.execute_command("FT.DROPINDEX", self.index_name)
            except Exception:
                pass

            self.r.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", "post:",
                "SCHEMA",
                "post_id", "TAG",
                "vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "512",
                "DISTANCE_METRIC", "COSINE",
                "text_vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "768",
                "DISTANCE_METRIC", "COSINE",
                "image_vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "512",
                "DISTANCE_METRIC", "COSINE",
            )
            logger.info(f"✅ Created CLIP 512-dim & SBERT 768-dim Triple Redis HNSW Index: {self.index_name}")
        except redis.exceptions.ResponseError as e:
            if "Index already exists" in str(e):
                logger.info(f"✅ Redis index '{self.index_name}' already exists, reusing.")
            else:
                # Schema mismatch or unknown error — drop and recreate
                logger.warning(f"⚠️ Index error ({e}), dropping and recreating...")
                try:
                    self.r.execute_command("FT.DROPINDEX", self.index_name)
                except Exception:
                    pass
                try:
                    self.r.execute_command(
                        "FT.CREATE", self.index_name,
                        "ON", "HASH",
                        "PREFIX", "1", "post:",
                        "SCHEMA",
                        "post_id", "TAG",
                        "vector", "VECTOR", "HNSW", "6",
                        "TYPE", "FLOAT32",
                        "DIM", "512",
                        "DISTANCE_METRIC", "COSINE",
                        "text_vector", "VECTOR", "HNSW", "6",
                        "TYPE", "FLOAT32",
                        "DIM", "768",
                        "DISTANCE_METRIC", "COSINE",
                        "image_vector", "VECTOR", "HNSW", "6",
                        "TYPE", "FLOAT32",
                        "DIM", "512",
                        "DISTANCE_METRIC", "COSINE",
                    )
                    logger.info(f"✅ Recreated Redis index with triple HNSW schema: {self.index_name}")
                except redis.exceptions.ResponseError as e2:
                    logger.error(f"❌ Failed to recreate Redis index: {e2}")

    def count(self) -> int:
        """Redis 인덱스에 저장된 벡터 수 반환."""
        try:
            info = self.r.execute_command("FT.INFO", self.index_name)
            for i in range(0, len(info) - 1, 2):
                key = info[i]
                if key in (b"num_docs", "num_docs"):
                    return int(info[i + 1])
            return 0
        except Exception:
            return 0

    def upsert_vector(self, post_id: uuid.UUID, vector: np.ndarray, text_vector: np.ndarray = None, metadata: Dict[str, Any] = None, image_vector: np.ndarray = None):
        """
        CLIP 512차원 벡터, SBERT 768차원 벡터 및 메타데이터를 Redis에 저장
        """
        key = f"post:{post_id}"
        vector_bytes = vector.astype(np.float32).tobytes()
        
        mapping = {
            "post_id": str(post_id),
            "vector": vector_bytes
        }

        if text_vector is not None:
            mapping["text_vector"] = text_vector.astype(np.float32).tobytes()

        if image_vector is not None:
            mapping["image_vector"] = image_vector.astype(np.float32).tobytes()
        
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    mapping[k] = str(v)
                    
        self.r.hset(key, mapping=mapping)

    def search_knn(self, query_vec: np.ndarray, k: int = 50, vector_field: str = "vector") -> List[uuid.UUID]:
        """
        K-Nearest Neighbors Search using COSINE similarity on a specified vector field ('vector' or 'text_vector')
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
