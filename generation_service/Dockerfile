FROM python:3.12-slim

WORKDIR /app

# 1. 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. 의존성 파일만 먼저 복사
COPY pyproject.toml ./

# 4. 기본 패키지 및 프로젝트 의존성 설치
RUN uv pip install --system --no-cache "uvicorn[standard]" websockets .

# 5. 모델 다운로드 스크립트만 복사해서 모델 먼저 캐싱
COPY download_models.py ./
ENV HF_HOME=/app/models
RUN mkdir -p $HF_HOME && python download_models.py

# 6. 이제 실제 애플리케이션 소스 코드 복사 (이 부분이 바뀌어도 위 단계들은 캐시됨)
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--ws", "websockets"]
