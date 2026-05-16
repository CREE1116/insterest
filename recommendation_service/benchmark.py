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
        print("📊 [Benchmark] 데이터 조회 중...")
        test_data = await get_benchmark_data(db)
        
        if not test_data:
            print("⚠️ 데이터가 부족합니다.")
            return

        k_list = [5, 10, 20]
        reco_results = {k: {"recall": [], "ndcg": []} for k in k_list}
        search_results = {k: {"recall": [], "ndcg": []} for k in k_list}
        
        print(f"🚀 총 {len(test_data)}명의 유저로 추천 품질 평가...")
        for user_id, pids in list(test_data.items())[:50]: # 시간 관계상 상위 50명 샘플링
            target_id = pids[-1]
            try:
                # 딕셔너리 리스트가 반환됨: [{"id": "...", "score": 0.9, "caption": "..."}]
                raw_results = await intel_service.discover(db, user_id=user_id, limit=50)
                recommended_ids = [str(r["id"]) for r in raw_results]
                
                for k in k_list:
                    top_k = recommended_ids[:k]
                    reco_results[k]["recall"].append(calculate_recall(str(target_id), top_k))
                    reco_results[k]["ndcg"].append(calculate_ndcg(str(target_id), top_k))
            except: pass

        print(f"🔎 캡션 기반 검색 품질 평가...")
        test_posts = await db.execute(text("SELECT post_id, content_text FROM search.post_vectors LIMIT 50"))
        for pid, content in test_posts.all():
            try:
                caption = content.get("caption", "") if content else ""
                if not caption: continue
                
                raw_results = await intel_service.discover(db, query_text=caption, limit=50, use_personalization=False)
                recommended_ids = [str(r["id"]) for r in raw_results]
                
                for k in k_list:
                    top_k = recommended_ids[:k]
                    search_results[k]["recall"].append(calculate_recall(str(pid), top_k))
                    search_results[k]["ndcg"].append(calculate_ndcg(str(pid), top_k))
            except: pass

        print("\n" + "═"*60)
        print(f"       📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("═"*60)
        print(f" {'K':<4} | {'Rec Recall':<10} | {'Rec NDCG':<10} | {'Search NDCG':<10}")
        print("-" * 60)
        for k in k_list:
            r_recall = np.mean(reco_results[k]["recall"]) if reco_results[k]["recall"] else 0
            r_ndcg = np.mean(reco_results[k]["ndcg"]) if reco_results[k]["ndcg"] else 0
            s_ndcg = np.mean(search_results[k]["ndcg"]) if search_results[k]["ndcg"] else 0
            print(f" {k:<4} | {r_recall:<10.4f} | {r_ndcg:<10.4f} | {s_ndcg:<10.4f}")
        print("═"*60)

if __name__ == "__main__":
    # uv를 사용하는 경우 환경 변수나 .venv 확인 필요
    print("🎬 Recommendation System Quantitative Benchmark 시작")
    try:
        asyncio.run(run_benchmark())
    except KeyboardInterrupt:
        print("\n🛑 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
