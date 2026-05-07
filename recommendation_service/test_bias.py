import torch
import torch.nn.functional as F
from app.ml.nlp import nlp_embedder
from app.services.intelligence import intel_service
import numpy as np

def test_similarity():
    # 1. 단어 임베딩 추출 (768차원)
    cat_raw = nlp_embedder.embed_text("고양이").unsqueeze(0)
    hedgehog_raw = nlp_embedder.embed_text("고슴도치").unsqueeze(0)
    
    raw_sim = F.cosine_similarity(cat_raw, hedgehog_raw).item()
    print(f"1. Raw SBERT Similarity (Cat vs Hedgehog): {raw_sim:.4f}")

    # 2. 모델 투영 후 임베딩 추출 (128차원)
    with torch.no_grad():
        cat_proj = intel_service.model.get_query_embedding(cat_raw.to(intel_service.device))
        hedgehog_proj = intel_service.model.get_query_embedding(hedgehog_raw.to(intel_service.device))
    
    proj_sim = F.cosine_similarity(cat_proj, hedgehog_proj).item()
    print(f"2. Projected (128d) Similarity: {proj_sim:.4f}")
    
    print("\n* 만약 2번 유사도가 1번보다 지나치게 높다면, 학습되지 않은 레이어가 모든 입력을 비슷한 공간으로 몰아넣고 있는 것입니다.")

if __name__ == "__main__":
    test_similarity()
