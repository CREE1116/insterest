import asyncio
import torch
import numpy as np
from sqlalchemy import select, text
from app.db.session import AsyncSessionLocal
from app.ml.nlp import nlp_embedder
from app.entities.models import PostVector
from app.ml.model import UnifiedDiscoveryModel
from sqlalchemy.ext.asyncio import AsyncSession
import torch.nn.functional as F

# 1. 모델 로드
model = UnifiedDiscoveryModel()
# 실제 환경에서는 학습된 가중치를 로드해야 합니다.
# model.load_state_dict(torch.load("/data/models/discovery_engine.pth", map_location='cpu'))
model.eval()

async def diagnose(query_text: str):
    print(f"\n🧪 --- 상세 검색 진단: '{query_text}' ---")
    
    # 2. 검색어 벡터화
    raw_query_vec = nlp_embedder.embed_text(query_text).unsqueeze(0)
    query_vec = model.get_query_embedding(raw_query_vec)
    
    async with AsyncSessionLocal() as db:
        # 3. DB에서 모든 포스트 벡터와 캡션 가져오기
        result = await db.execute(select(PostVector))
        posts = result.scalars().all()
        
        scores = []
        for post in posts:
            # 바이너리 벡터 복원
            item_vec = torch.from_numpy(np.frombuffer(post.caption_vector, dtype=np.float32)).unsqueeze(0)
            # 투영 (Project to 128-dim)
            with torch.no_grad():
                item_emb = model.get_item_embedding(item_vec)
                
            # 유사도 계산
            similarity = F.cosine_similarity(query_vec, item_emb).item()
            
            caption = post.content_text.get('caption', 'N/A') if post.content_text else "N/A"
            scores.append((similarity, caption))
        
        # 4. 점수순 정렬 후 상위 5개 출력
        scores.sort(key=lambda x: x[0], reverse=True)
        
        for i, (score, caption) in enumerate(scores[:5]):
            print(f"   [{i+1}] 점수: {score:.4f} | 캡션: {caption}")

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "고양이"
    asyncio.run(diagnose(query))
