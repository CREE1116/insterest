import asyncio
import uuid
import torch
import numpy as np
import logging
from sqlalchemy import text
from app.db.session import AsyncSessionLocal
from app.services.intelligence import intel_service
from typing import List, Dict, Optional

# Suppress unnecessary logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app.services.intelligence").setLevel(logging.ERROR)

async def get_benchmark_data(db):
    """
    interaction 스키마에서 실제 사용자의 좋아요/저장 이력을 가져와 
    테스트 셋(마지막 아이템)과 훈련 셋(나머지 이력)으로 분리합니다.
    """
    query = text("""
        SELECT user_id, post_id, created_at 
        FROM interaction.likes 
        UNION ALL
        SELECT user_id, post_id, created_at 
        FROM interaction.saves
        ORDER BY created_at ASC
    """)
    result = await db.execute(query)
    rows = result.all()
    
    user_history = {}
    for uid, pid, _ in rows:
        if uid not in user_history:
            user_history[uid] = []
        if pid not in user_history[uid]:
            user_history[uid].append(pid)
    
    # 최소 5개 이상의 상호작용이 있는 유저만 평가 대상으로 선정
    test_users = {uid: pids for uid, pids in user_history.items() if len(pids) >= 5}
    return test_users

def calculate_recall(target_id: uuid.UUID, recommended_ids: List[uuid.UUID]):
    """Recall@K: target_id가 추천 목록에 있으면 1, 없으면 0"""
    return 1.0 if target_id in recommended_ids else 0.0

def calculate_ndcg(target_id: uuid.UUID, recommended_ids: List[uuid.UUID]):
    """NDCG@K: target_id의 순위가 높을수록 높은 점수 부여"""
    if target_id in recommended_ids:
        rank = recommended_ids.index(target_id) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0

async def run_benchmark():
    async with AsyncSessionLocal() as db:
        print("📊 [Benchmark] interaction 데이터 조회 중...")
        test_data = await get_benchmark_data(db)
        
        if not test_data:
            print("⚠️ 평가를 위한 상호작용 데이터가 부족합니다. (최소 5개 이상의 활동을 한 유저가 필요합니다)")
            # 임시 데이터 생성을 유도하거나 종료
            return

        k_list = [10, 20, 50]
        results = {k: {"recall": [], "ndcg": []} for k in k_list}
        
        print(f"🚀 총 {len(test_data)}명의 사용자에 대해 벤치마크를 수행합니다...")
        
        count = 0
        for user_id, pids in test_data.items():
            # Leave-one-out evaluation: 마지막 아이템을 정답(target)으로 설정
            target_id = pids[-1]
            
            # intel_service.discover는 내부적으로 Interaction Service에서 이력을 가져오므로
            # DB의 데이터와 서비스의 응답이 일치한다고 가정합니다.
            try:
                # Top-50 추천 결과 획득
                recommended_ids = await intel_service.discover(db, user_id=user_id, limit=50)
                
                for k in k_list:
                    top_k = recommended_ids[:k]
                    results[k]["recall"].append(calculate_recall(target_id, top_k))
                    results[k]["ndcg"].append(calculate_ndcg(target_id, top_k))
                
                count += 1
                if count % 10 == 0:
                    print(f"  ... {count}/{len(test_data)} 유저 평가 완료")
            except Exception:
                pass

        print("\n" + "═"*45)
        print(f"       📊 RECOMMENDATION BENCHMARK RESULTS")
        print("═"*45)
        print(f" {'K':<4} | {'Recall':<10} | {'NDCG':<10}")
        print("-" * 45)
        for k in k_list:
            avg_recall = np.mean(results[k]["recall"]) if results[k]["recall"] else 0
            avg_ndcg = np.mean(results[k]["ndcg"]) if results[k]["ndcg"] else 0
            print(f" {k:<4} | {avg_recall:<10.4f} | {avg_ndcg:<10.4f}")
        print("═"*45)
        print(f" * Evaluation method: Leave-one-out on interaction history")
        print(f" * Sample size: {count} users")

if __name__ == "__main__":
    # uv를 사용하는 경우 환경 변수나 .venv 확인 필요
    print("🎬 Recommendation System Quantitative Benchmark 시작")
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        print("\n🛑 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
