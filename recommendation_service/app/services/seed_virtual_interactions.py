import asyncio
import uuid
import random
from sqlalchemy import text, select
from app.db.session import AsyncSessionLocal
from app.entities.models import PostVector

# 22개 페르소나 — 각 페르소나는 1~2개 테마에 집중 (평가 정확도 향상)
# 키워드는 seed_dummy_data.py의 THEMES 내 subject/adjective와 매칭됨
PERSONAS = {
    # ── 반려동물 ──────────────────────────────────────
    "CatLover":         ["고양이", "귀여운", "사랑스러운"],
    "DogLover":         ["강아지", "신나게", "낮잠"],
    "PetParent":        ["고양이", "강아지", "햄스터", "반려동물"],

    # ── 도시 ──────────────────────────────────────────
    "CityNightOwl":     ["야경", "네온사인", "화려한", "빌딩숲"],
    "UrbanExplorer":    ["서울", "도심", "거리", "활기찬"],

    # ── 음식 ──────────────────────────────────────────
    "KoreanFoodie":     ["떡볶이", "맛있는", "매콤한"],
    "CafeAddict":       ["커피", "카페 인테리어", "고소한"],
    "DessertLover":     ["디저트", "신선한", "고소한"],
    "FineDining":       ["스테이크", "파스타", "인생맛집"],

    # ── 자연 ──────────────────────────────────────────
    "NatureLover":      ["바다", "숲속", "푸른", "평화로운"],
    "MountainHiker":    ["산 정상", "계곡", "웅장한", "상쾌한"],

    # ── 인테리어 / 라이프스타일 ─────────────────────────
    "InteriorDesigner": ["거실", "침실", "모던한", "미니멀한"],
    "CozyCafe":         ["카페 인테리어", "아늑한", "빈티지한"],

    # ── 패션 ──────────────────────────────────────────
    "Fashionista":      ["패션", "런웨이", "트렌디한", "세련된"],
    "SneakerHead":      ["운동화", "힙한", "나만의"],

    # ── 운동 / 건강 ────────────────────────────────────
    "Runner":           ["러닝", "땀 흘리는", "활력 넘치는"],
    "YogaPilates":      ["필라테스", "명상", "건강한", "꾸준한"],
    "OutdoorAthlete":   ["등산", "홈트", "개운한"],

    # ── 예술 ──────────────────────────────────────────
    "ArtEnthusiast":    ["전시회", "그림", "예술적인", "감성적인"],
    "MusicLover":       ["LP", "공연", "클래식한", "신비로운"],

    # ── 테크 / 업무 ────────────────────────────────────
    "TechDeveloper":    ["코딩", "개발자", "열정적인", "뿌듯한"],
    "RemoteWorker":     ["재택근무", "데스크테리어", "집중하는", "창의적인"],
}

async def seed_interactions():
    print("👥 가상 유저 페르소나 기반 인터랙션 시딩 시작...")

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(PostVector))
        posts = result.scalars().all()

        if not posts:
            print("⚠️ 포스트가 없습니다. seed_dummy_data.py를 먼저 실행하세요.")
            return

        for name, keywords in PERSONAS.items():
            user_id = uuid.uuid5(uuid.NAMESPACE_DNS, name)
            print(f"👤 페르소나: {name} (ID: {user_id})")

            # 키워드 매칭 — 매칭 횟수만큼 가중치 부여
            matched_posts = []
            for post in posts:
                caption = post.content_text.get("caption", "") if post.content_text else ""
                weight = sum(1 for k in keywords if k in caption)
                if weight > 0:
                    matched_posts.extend([post.post_id] * weight)

            if not matched_posts:
                matched_posts = [p.post_id for p in random.sample(posts, min(5, len(posts)))]

            # 페르소나당 8~20개 좋아요 (데이터 증가)
            unique = list(set(matched_posts))
            sample_size = min(len(unique), random.randint(8, 20))
            if sample_size < 5 and len(posts) >= 5:
                extra = [p.post_id for p in random.sample(posts, 5 - sample_size)]
                target_posts = unique[:sample_size] + extra
            else:
                target_posts = random.sample(unique, sample_size)

            # 시간 순서 있는 좋아요 (evaluate_offline이 시간순 last item 예측)
            for i, post_id in enumerate(target_posts):
                try:
                    await db.execute(text("""
                        INSERT INTO interaction.likes (user_id, post_id, created_at)
                        VALUES (:u, :p, NOW() - INTERVAL '1 second' * :offset)
                        ON CONFLICT DO NOTHING
                    """), {"u": user_id, "p": post_id, "offset": len(target_posts) - i})
                except Exception as e:
                    print(f"   ❌ 좋아요 주입 실패: {e}")

            print(f"   ✅ {len(target_posts)}개 '좋아요' 생성")

        await db.commit()
    print("\n🎉 22개 페르소나 인터랙션 시딩 완료!")

if __name__ == "__main__":
    asyncio.run(seed_interactions())
