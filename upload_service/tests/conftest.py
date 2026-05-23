import os
os.environ.setdefault("POSTGRES_SERVER", "localhost")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "test_db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-ci")

import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def client():
    from app.main import app

    with patch("app.main.startup", new_callable=AsyncMock), \
         patch("app.main.consume_generation_completed", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
