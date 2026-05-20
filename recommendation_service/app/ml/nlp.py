import functools
import numpy as np
import torch
import logging
import io
from PIL import Image
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

class NLPEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading NLP Model: {settings.MODEL_NAME} on {self.device}")
        
        # 텍스트 전용 모델 (기존)
        self.model = SentenceTransformer(settings.MODEL_NAME, device=self.device)
        
        # 멀티모달 CLIP 모델 (이미지 임베딩용)
        self.clip_model = SentenceTransformer('clip-ViT-B-32', device=self.device)
        
    @functools.lru_cache(maxsize=1000)
    def _embed_text_cached(self, text: str) -> np.ndarray:
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.cpu().numpy()

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a single text string into a vector with LRU caching.
        """
        vec_np = self._embed_text_cached(text)
        return torch.from_numpy(vec_np).to(self.device, dtype=torch.float32)

    def embed_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Embed a list of text strings into vectors.
        """
        if not texts:
            return torch.zeros((0, 768), device=self.device)
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.to(dtype=torch.float32)

    def embed_text_clip(self, text: str) -> torch.Tensor:
        """
        CLIP 텍스트 인코더: 512-dim, embed_image()와 동일한 공간.
        캡션/검색어를 이미지와 직접 비교 가능.
        """
        with torch.no_grad():
            embedding = self.clip_model.encode(text, convert_to_tensor=True)
        return embedding.to(self.device, dtype=torch.float32)

    def embed_batch_clip(self, texts: list) -> torch.Tensor:
        """CLIP 텍스트 배치 인코딩: [N, 512]"""
        if not texts:
            return torch.zeros((0, 512), device=self.device)
        with torch.no_grad():
            embeddings = self.clip_model.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device, dtype=torch.float32)

    def embed_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        이미지 바이너리를 입력받아 CLIP 벡터를 생성합니다.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            with torch.no_grad():
                # CLIP 모델을 통해 이미지 임베딩 생성
                embedding = self.clip_model.encode(image, convert_to_tensor=True)
            
            # 만약 CLIP 모델의 차원이 768이 아니라면(예: 512), 
            # 텍스트 벡터와 결합하기 위해 선형 변환이 필요할 수 있습니다.
            # clip-ViT-B-32는 기본 512차원입니다. 
            # 여기서는 단순화를 위해 0 패딩을 하거나 차원을 맞추는 로직이 필요할 수 있습니다.
            # 우선은 512 -> 768 확장을 단순 선형 변환 없이 패딩으로 처리하거나 
            # 혹은 768차원 CLIP 모델(clip-ViT-L-14)을 사용하는 것을 권장합니다.
            
            # CLIP-ViT-B-32 provides 512-dim vectors.
            # No more zero-padding to 768. The Projection Layer will handle the dimension change.
            return embedding.to(dtype=torch.float32)
        except Exception as e:
            logger.error(f"❌ Image embedding failed: {e}")
            return torch.zeros(512, device=self.device)

nlp_embedder = NLPEmbedder()
