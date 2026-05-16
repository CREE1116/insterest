import asyncio
import uuid
import numpy as np
import torch
import random
from app.db.session import AsyncSessionLocal
from app.services.intelligence import intel_service
from app.ml.nlp import nlp_embedder

# 테마별 키워드 조합기
THEMES = {
    "Pet": (["고양이", "강아지", "햄스터", "반려동물"], ["낮잠 자는", "신나게 노는", "함께 산책하는", "귀여운", "사랑스러운"]),
    "City": (["서울", "도심", "야경", "빌딩숲", "거리"], ["화려한", "활기찬", "비 내리는", "노을 지는", "북적이는"]),
    "Food": (["떡볶이", "커피", "디저트", "스테이크", "파스타"], ["맛있는", "매콤한", "고소한", "신선한", "인생맛집"]),
    "Nature": (["바다", "숲속", "계곡", "산 정상", "들판"], ["푸른", "조용한", "웅장한", "상쾌한", "평화로운"]),
    "Interior": (["거실", "침실", "주방", "서재", "카페 인테리어"], ["모던한", "미니멀한", "빈티지한", "아늑한", "감각적인"]),
    "Work": (["코딩", "재택근무", "회의실", "데스크테리어", "열공"], ["집중하는", "바쁜", "창의적인", "뿌듯한", "열정적인"]),
    "Travel": (["공항", "기차여행", "호텔", "캠핑장", "해외여행"], ["설레는", "자유로운", "여유로운", "특별한", "잊지 못할"]),
    "Art": (["전시회", "LP", "카메라", "그림", "공연"], ["예술적인", "클래식한", "감성적인", "신비로운", "독특한"]),
    "Style": (["데일리룩", "패션", "운동화", "향수", "런웨이"], ["멋진", "트렌디한", "나만의", "세련된", "힙한"]),
    "Health": (["러닝", "필라테스", "홈트", "명상", "등산"], ["땀 흘리는", "건강한", "활력 넘치는", "꾸준한", "개운한"])
}

def generate_100_posts():
    posts = []
    for theme, (subjects, adjectives) in THEMES.items():
        # 각 테마당 약 12개씩 생성 (10개 테마 * 12 = 120개)
        for _ in range(12):
            sub = random.choice(subjects)
            adj = random.choice(adjectives)
            caption = f"{adj} {sub} 풍경 ✨" if theme != "Pet" else f"{adj} {sub} 🐾"
            tags = [theme, sub, adj.replace(" ", "")]
            posts.append({"caption": caption, "tags": tags})
    return posts

async def seed():
    print("🌱 100+ 가상 데이터 대량 생성 시작...")
    dummy_posts = generate_100_posts()
    
    async with AsyncSessionLocal() as db:
        for i, item in enumerate(dummy_posts):
            post_id = uuid.uuid4()
            caption = item["caption"]
            tags = " ".join(item["tags"])
            
            if (i + 1) % 20 == 0:
                print(f"📦 생성 중... ({i+1}/{len(dummy_posts)})")
            
            c_vec = nlp_embedder.embed_text(caption).cpu().numpy()
            h_vec = nlp_embedder.embed_text(tags).cpu().numpy()
            i_vec = np.random.randn(512).astype(np.float32)
            
            await intel_service.index_post(
                db, 
                post_id, 
                c_vec, 
                h_vec, 
                i_vec, 
                metadata={"caption": caption, "tags": item["tags"], "is_dummy": True}
            )
            
    print(f"\n✅ 총 {len(dummy_posts)}개의 데이터 시딩 완료!")

if __name__ == "__main__":
    asyncio.run(seed())
