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
    - vector: 128-dim projected unified space
    - text_vector: 768-dim SBERT semantic text space
    - image_vector: 128-dim projected image space
    """
    def __init__(self):
        self.r = redis.Redis(host=settings.REDIS_HOST, port=settings.RECO_REDIS_PORT, decode_responses=False)
        self.index_name = "post_vectors"

    def create_index(self):
        """인덱스 생성. 이미 올바른 스키마로 존재하면 그대로 재사용."""
        try:
            try:
                info = self.r.execute_command("FT.INFO", self.index_name)
                info_dict = {}
                for i in range(0, len(info) - 1, 2):
                    k = info[i].decode('utf-8') if isinstance(info[i], bytes) else info[i]
                    info_dict[k] = info[i+1]
                
                attributes = info_dict.get("attributes", [])
                text_vector_dim = None
                has_vector = False
                for attr in attributes:
                    attr_dict = {}
                    for j in range(0, len(attr) - 1, 2):
                        ak = attr[j].decode('utf-8') if isinstance(attr[j], bytes) else attr[j]
                        attr_dict[ak] = attr[j+1]
                    ident = attr_dict.get("identifier")
                    ident_str = ident.decode('utf-8') if isinstance(ident, bytes) else ident
                    if ident_str == "vector":
                        has_vector = True
                    elif ident_str == "text_vector":
                        text_vector_dim = attr_dict.get("dim")
                        if isinstance(text_vector_dim, bytes):
                            text_vector_dim = int(text_vector_dim)
                        elif isinstance(text_vector_dim, (str, int)):
                            text_vector_dim = int(text_vector_dim)
                
                if has_vector or text_vector_dim != 512:
                    logger.warning(f"⚠️ Redis index doesn't match 512-dim Pure CLIP Space (found text_vector dim {text_vector_dim}, has_vector {has_vector}). Dropping to migrate...")
                    self.r.execute_command("FT.DROPINDEX", self.index_name)
            except Exception as e:
                logger.debug(f"Index check failed or not found: {e}")

            self.r.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", "post:",
                "SCHEMA",
                "post_id", "TAG",
                "text_vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "512",
                "DISTANCE_METRIC", "COSINE",
                "image_vector", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", "512",
                "DISTANCE_METRIC", "COSINE",
            )
            logger.info(f"✅ Created Pure CLIP 512-dim Redis HNSW Index: {self.index_name}")
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
                        "text_vector", "VECTOR", "HNSW", "6",
                        "TYPE", "FLOAT32",
                        "DIM", "512",
                        "DISTANCE_METRIC", "COSINE",
                        "image_vector", "VECTOR", "HNSW", "6",
                        "TYPE", "FLOAT32",
                        "DIM", "512",
                        "DISTANCE_METRIC", "COSINE",
                    )
                    logger.info(f"✅ Recreated Redis index with 512-dim Pure CLIP schema: {self.index_name}")
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

    def upsert_vector(self, post_id: uuid.UUID, vector: np.ndarray = None, text_vector: np.ndarray = None, metadata: Dict[str, Any] = None, image_vector: np.ndarray = None):
        """
        CLIP 512-dim vectors and metadata stored in Redis
        """
        if image_vector is not None:
            assert image_vector.shape[-1] == 512, f"Expected image_vector dimension 512, got {image_vector.shape[-1]}"
            norm = np.linalg.norm(image_vector)
            if norm > 1e-5:
                image_vector = image_vector / norm
        if text_vector is not None:
            assert text_vector.shape[-1] == 512, f"Expected text_vector dimension 512, got {text_vector.shape[-1]}"

        key = f"post:{post_id}"
        
        mapping = {
            "post_id": str(post_id)
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

    def search_knn(self, query_vec: np.ndarray, k: int = 50, vector_field: str = "vector",
                   ef_runtime: int = 200) -> List[uuid.UUID]:
        """
        K-Nearest Neighbors Search using COSINE similarity on a specified vector field.

        ef_runtime: HNSW exploration factor at query time.
            Higher → better recall, slightly higher latency.
            Recommended: 200 (default, balanced), 400 (high-recall personalization).
        """
        query_vec_bytes = query_vec.astype(np.float32).tobytes()

        # Redis Vector Search Query with ef_runtime hint
        query = (
            f"*=>[KNN {k} @{vector_field} $query_vec EF_RUNTIME {ef_runtime} AS score]"
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
