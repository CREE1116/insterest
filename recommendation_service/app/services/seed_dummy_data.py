import asyncio
import uuid
import numpy as np
import torch
from app.db.session import AsyncSessionLocal
from app.services.intelligence import intel_service
from app.ml.nlp import nlp_embedder

# 가상 데이터 셋 (다양한 테마)
DUMMY_POSTS = [
    {"caption": "햇살 가득한 오후, 창가에서 낮잠 자는 아기 고양이 🐾", "tags": ["고양이", "나른한오후", "힐링"]},
    {"caption": "비 내리는 서울의 밤거리, 반짝이는 네온사인과 빗소리 ☔️", "tags": ["서울", "야경", "비오는날"]},
    {"caption": "직접 구운 고소한 크로와상과 따뜻한 아메리카노 한 잔 ☕️", "tags": ["홈카페", "빵지순례", "커피"]},
    {"caption": "끝없이 펼쳐진 에메랄드빛 바다와 하얀 모래사장 🏝", "tags": ["여행", "바다", "여름휴가"]},
    {"caption": "오늘 점심은 매콤한 떡볶이와 바삭한 모듬 튀김! 🌶", "tags": ["먹스타그램", "떡볶이", "Kfood"]},
    {"caption": "적막한 새벽, 책상 스탠드 아래에서 읽는 소설 한 권 📖", "tags": ["독서", "새벽감성", "조용한시간"]},
    {"caption": "화려한 도시의 스카이라인을 바라보며 마시는 와인 🍷", "tags": ["호캉스", "도시야경", "분위기"]},
    {"caption": "숲속 작은 오두막에서 맞이하는 상쾌한 아침 공기 🌲", "tags": ["캠핑", "자연", "미니멀라이프"]},
    {"caption": "귀여운 강아지와 함께하는 공원 산책, 꼬리 살랑살랑 🐶", "tags": ["강아지", "산책", "멍스타그램"]},
    {"caption": "노을 지는 강변 테라스에서 친구들과 나누는 수다 🌅", "tags": ["노을", "우정", "테라스"]},
    {"caption": "웅장한 유럽풍 건축물의 기하학적 미학 🏛️", "tags": ["건축", "유럽여행", "디자인"]},
    {"caption": "화려한 런웨이를 수놓는 올 시즌 트렌드 패션 👗", "tags": ["패션", "런웨이", "모델"]},
    {"caption": "땀 흘리며 달리는 마라톤, 한계를 넘어서는 순간 🏃", "tags": ["운동", "마라톤", "자기관리"]},
    {"caption": "밤하늘을 수놓는 수만 개의 별과 은하수 🌌", "tags": ["우주", "은하수", "별사진"]},
    {"caption": "오래된 레코드판에서 흘러나오는 클래식 재즈 선율 🎷", "tags": ["재즈", "LP", "아날로그"]},
    {"caption": "갓 따온 신선한 과일들이 가득한 활기찬 전통시장 🍎", "tags": ["시장", "과일", "활기찬하루"]},
    {"caption": "모던한 인테리어의 미니멀리즘 거실 풍경 🛋️", "tags": ["인테리어", "집꾸미기", "미니멀"]},
    {"caption": "폭포 소리가 우렁차게 들리는 깊은 계곡의 절경 🌊", "tags": ["대자연", "폭포", "등산"]},
    {"caption": "코딩에 집중하고 있는 개발자의 모니터 속 코드 💻", "tags": ["개발자", "코딩", "열정"]},
    {"caption": "빈티지 카메라로 담아낸 흑백 필름 사진의 매력 📷", "tags": ["사진", "필름카메라", "빈티지"]}
]

async def seed():
    print("🌱 가상 데이터 시딩(Seeding) 시작...")
    
    async with AsyncSessionLocal() as db:
        for item in DUMMY_POSTS:
            post_id = uuid.uuid4()
            caption = item["caption"]
            tags = " ".join(item["tags"])
            
            print(f"📦 인덱싱 중: {caption[:20]}...")
            
            # 1. 텍스트 벡터화
            c_vec = nlp_embedder.embed_text(caption).cpu().numpy()
            h_vec = nlp_embedder.embed_text(tags).cpu().numpy()
            
            # 2. 이미지 벡터 (가상은 0으로 처리하거나 랜덤 생성)
            i_vec = np.random.randn(512).astype(np.float32)
            
            # 3. 인덱싱 (메타데이터 포함 + 더미 플래그)
            await intel_service.index_post(
                db, 
                post_id, 
                c_vec, 
                h_vec, 
                i_vec, 
                metadata={"caption": caption, "tags": item["tags"], "is_dummy": True}
            )
            
    print("\n✅ 시딩 완료! 이제 검색과 지표를 확인해 보세요.")

if __name__ == "__main__":
    asyncio.run(seed())
