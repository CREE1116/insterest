#!/bin/bash

# 1. 추천 서비스 파드 이름 찾기
POD_NAME=$(kubectl get pod -l app=recommendation-service -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "❌ 에러: recommendation-service 파드를 찾을 수 없습니다."
    exit 1
fi

echo "🚀 AI 시스템 보고서 생성 프로세스 시작 (Pod: $POD_NAME)..."

# 2. 파드 내부에서 AI 리포트 생성 스크립트 실행
kubectl exec -it $POD_NAME -- python -m app.services.generate_ai_report

# 3. 생성된 CSV 파일을 로컬로 복사
echo -e "\n📂 CSV 파일 다운로드 중..."
kubectl cp $POD_NAME:ai_system_report.csv ./ai_system_report.csv

if [ $? -eq 0 ]; then
    echo "✅ 성공! 보고서가 로컬에 저장되었습니다: $(pwd)/ai_system_report.csv"
    
    # OS가 Mac인 경우 엑셀(또는 기본 뷰어)로 즉시 열기
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open ./ai_system_report.csv
    fi
else
    echo "❌ 에러: CSV 파일 복사에 실패했습니다."
fi
