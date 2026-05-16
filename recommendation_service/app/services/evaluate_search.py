import asyncio
import httpx
import sys

# 테스트할 검색어 리스트
TEST_QUERIES = [
    "고양이",
    "조용한 밤하늘",
    "감성적인 새벽",
    "활기찬 도시",
    "맛있는 음식"
]

RECOMMENDATION_SERVICE_URL = "http://localhost:8008" # 로컬 포트포워딩 기준

async def evaluate():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n🔍 --- 검색 정확도 판별 벤치마크 시작 ---\n")
        
        for query in TEST_QUERIES:
            print(f"👉 검색어: '{query}'")
            try:
                # 1. Discovery API 호출 (개인화 제외하고 순수 검색 품질만 체크)
                response = await client.get(
                    f"{RECOMMENDATION_SERVICE_URL}/api/v1/discovery",
                    params={"query": query, "limit": 5, "use_personalization": "false"}
                )
                
                if response.status_code != 200:
                    print(f"   ❌ 에러 발생: {response.status_code}")
                    continue
                
                results = response.json()
                if not results:
                    print("   ⚠️ 검색 결과가 없습니다.")
                    continue
                
                # 2. 결과 출력
                for i, item_id in enumerate(results):
                    # 상세 정보를 위해 upload-service 등에서 데이터를 가져올 수 있으나, 
                    # 여기서는 ID와 검색어 간의 매칭 느낌을 확인합니다.
                    print(f"   [{i+1}] Post ID: {item_id}")
                
            except Exception as e:
                print(f"   ❌ 통신 에러: {e}")
            print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_QUERIES = [sys.argv[1]]
    
    asyncio.run(evaluate())
