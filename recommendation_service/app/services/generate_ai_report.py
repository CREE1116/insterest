import asyncio
import time
from sqlalchemy import text
from app.db.session import AsyncSessionLocal
from app.services.intelligence import intel_service
from app.services.seed_dummy_data import seed as seed_posts
from app.services.seed_virtual_interactions import seed_interactions

async def run_full_pipeline():
    print("🚀 [AI Report System] 통합 검증 프로세스 시작...")
    start_time = time.time()
    
    async with AsyncSessionLocal() as db:
        # 1. 데이터 시딩
        print("\nStep 1: 데이터 시딩 (Posts & Interactions)")
        await seed_posts()
        await seed_interactions()
        
        # 2. 모델 학습
        print("\nStep 2: AI 모델 학습 (User-Item Alignment)...")
        # 내부적으로 train_discovery를 호출하여 가중치를 업데이트합니다.
        await intel_service.train_discovery(db)
        
        # 3. 최종 지표 측정
        print("\nStep 3: 최종 성능 지표 산출...")
        report = await intel_service.evaluate_offline(db)
        
        # 4. 데이터 청소 (Cleanup)
        print("\nStep 4: 가상 데이터 청소 및 원복...")
        try:
            # 시딩된 가상 유저 ID들 (PERSONAS 기반 생성된 것들) 삭제
            from app.services.seed_virtual_interactions import PERSONAS
            import uuid
            virtual_user_ids = [uuid.uuid5(uuid.NAMESPACE_DNS, name) for name in PERSONAS.keys()]
            
            for uid in virtual_user_ids:
                await db.execute(text("DELETE FROM interaction.likes WHERE user_id = :uid"), {"uid": uid})
            
            # 시딩된 가상 포스트들 삭제 (content_text의 'is_dummy' 플래그 활용하거나 전체 삭제 후 재동기화)
            # 여기서는 안전하게 interaction 데이터만 날리고 포스트는 backfill로 관리하는 방식을 제안하지만,
            # 유저님의 요청대로 '싹 비우기' 위해 가상 포스트들도 정리합니다.
            await db.execute(text("DELETE FROM search.post_vectors WHERE content_text->>'is_dummy' = 'true'"))
            await db.commit()
            print("✅ Cleanup 완료: 가상 데이터가 성공적으로 제거되었습니다.")
        except Exception as e:
            print(f"⚠️ Cleanup 중 오류 발생: {e}")
            await db.rollback()
        
    duration = time.time() - start_time
    
    # 4. 보고서 출력
    print("\n" + "═"*60)
    print("       📊 INSTEREST AI RECOMMENDATION ENGINE REPORT")
    print("═"*60)
    print(f" * 생성 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" * 총 소요 시간: {duration:.2f}초")
    print(f" * 샘플 규모: 유저 {report['sample_size_users']}명 / 검색 {report['sample_size_search']}건")
    print("-" * 60)
    
    metrics = report["metrics"]
    print(f" [시맨틱 검색 성능 (Search Fidelity)]")
    for k, v in metrics["search_fidelity"].items():
        print(f"  - {k}: {v:.4f} " + ("✅" if v > 0.8 else "📈"))
        
    print(f"\n [개인화 추천 성능 (Recommendation Quality)]")
    for k, v in metrics["recommendation_quality"].items():
        print(f"  - {k}: {v:.4f} " + ("✅" if v > 0.5 else "📈"))
        
    print("-" * 60)
    print(" 💡 분석 결과: " + 
          ("시스템이 유저의 취향을 성공적으로 학습했습니다." if metrics["recommendation_quality"]["NDCG@10"] > 0.4 
           else "더 많은 데이터와 에폭(Epoch) 학습이 필요합니다."))
    print("═"*60)

    # 5. CSV 파일 저장 (Excel용)
    try:
        import csv
        csv_file = "ai_system_report.csv"
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Group", "Metric", "Score"])
            for group, group_metrics in metrics.items():
                for name, score in group_metrics.items():
                    writer.writerow([group, name, score])
        print(f"\n📂 CSV 리포트 생성 완료: {csv_file}")
    except Exception as e:
        print(f"⚠️ CSV 저장 실패: {e}")

if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
