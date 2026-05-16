FROM python:3.12-slim

WORKDIR /app

# 1. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. 의존성 파일 복사 및 설치
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# 3. 실제 애플리케이션 소스 코드 복사
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--ws", "websockets"]
