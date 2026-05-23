import asyncio
import numpy as np
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.ml.nlp import nlp_embedder
from app.entities.models import PostVector

async def diagnose(query_text: str):
    print(f"\n🧪 --- 상세 검색 진단: '{query_text}' ---")
    
    # 2. 검색어 벡터화
    query_vec = nlp_embedder.embed_text_clip(query_text).cpu().numpy()
    norm_q = np.linalg.norm(query_vec)
    if norm_q > 1e-5:
        query_vec = query_vec / norm_q
    
    async with AsyncSessionLocal() as db:
        # 3. DB에서 모든 포스트 벡터와 캡션 가져오기
        result = await db.execute(select(PostVector))
        posts = result.scalars().all()
        
        scores = []
        for post in posts:
            # 바이너리 벡터 복원
            cap_bytes = post.caption_vector
            if not cap_bytes:
                continue
            item_vec = np.frombuffer(cap_bytes, dtype=np.float32).copy()
            if len(item_vec) != 512:
                continue
            norm_i = np.linalg.norm(item_vec)
            if norm_i > 1e-5:
                item_vec = item_vec / norm_i
                
            # 유사도 계산
            similarity = float(np.dot(query_vec, item_vec))
            
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
