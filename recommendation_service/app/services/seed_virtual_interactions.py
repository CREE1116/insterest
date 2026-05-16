import asyncio
import uuid
import random
from sqlalchemy import text, select
from app.db.session import AsyncSessionLocal
from app.entities.models import PostVector

# 1. 가상 유저 페르소나와 관심 키워드
PERSONAS = {
    "CatLover": ["고양이", "강아지", "반려동물", "아기"],
    "CityWalker": ["도시", "야경", "네온사인", "서울", "스카이라인"],
    "CafeTraveler": ["카페", "커피", "여행", "바다", "휴가", "홈카페"],
    "Foodie": ["음식", "떡볶이", "먹스타그램", "맛있는", "점심"],
    "Architect": ["건축", "인테리어", "디자인", "거실", "스카이라인"],
    "Fashionista": ["패션", "런웨이", "모델", "트렌드", "빈티지"],
    "Athlete": ["운동", "마라톤", "자기관리", "등산", "폭포"],
    "TechGeek": ["코딩", "개발자", "모니터", "코드", "열정", "디자인"]
}

async def seed_interactions():
    print("👥 가상 유저 페르소나 기반 인터랙션 시딩 시작...")
    
    async with AsyncSessionLocal() as db:
        # DB의 모든 포스트 로드
        result = await db.execute(select(PostVector))
        posts = result.scalars().all()
        
        if not posts:
            print("⚠️ 포스트가 없습니다. seed_dummy_data.py를 먼저 실행하세요.")
            return

        for name, keywords in PERSONAS.items():
            user_id = uuid.uuid5(uuid.NAMESPACE_DNS, name) # 고정된 가상 유저 ID 생성
            print(f"👤 페르소나 생성: {name} (ID: {user_id})")
            
            # 관심 키워드가 포함된 포스트 찾기 (가중 샘플링)
            matched_posts = []
            for post in posts:
                caption = post.content_text.get("caption", "") if post.content_text else ""
                # 키워드 매칭 개수에 따라 가중치 부여
                match_count = sum(1 for k in keywords if k in caption)
                if match_count > 0:
                    matched_posts.extend([post.post_id] * match_count)
            
            if not matched_posts:
                # 매칭되는 게 없으면 랜덤으로 몇 개 선택
                matched_posts = [p.post_id for p in random.sample(posts, min(5, len(posts)))]
            
            # 유저당 최소 5개, 최대 15개의 좋아요 생성
            sample_size = min(len(set(matched_posts)), random.randint(5, 15))
            if sample_size < 5 and len(posts) >= 5:
                # 데이터가 너무 적으면 전체 포스트에서 랜덤으로 채움
                extra_posts = [p.post_id for p in random.sample(posts, 5 - sample_size)]
                target_posts = list(set(matched_posts)) + extra_posts
            else:
                target_posts = random.sample(list(set(matched_posts)), sample_size)
            
            for post_id in target_posts:
                try:
                    # interaction.likes 테이블에 직접 주입
                    await db.execute(text("""
                        INSERT INTO interaction.likes (user_id, post_id, created_at)
                        VALUES (:u, :p, NOW())
                        ON CONFLICT DO NOTHING
                    """), {"u": user_id, "p": post_id})
                except Exception as e:
                    print(f"   ❌ 좋아요 주입 실패: {e}")
            
            print(f"   ✅ {len(target_posts)}개의 '좋아요' 생성 완료.")
        
        await db.commit()
    print("\n🎉 모든 가상 유저 상호작용이 준비되었습니다! 이제 /metrics에서 개인화 점수를 확인해 보세요.")

if __name__ == "__main__":
    asyncio.run(seed_interactions())
