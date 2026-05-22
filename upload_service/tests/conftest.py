import os
os.environ.setdefault("POSTGRES_SERVER", "localhost")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "test_db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-ci")

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def client():
    from app.main import app
    from app.db.session import engine

    mock_conn = MagicMock()
    mock_conn.run_sync = AsyncMock()
    mock_conn.execute = AsyncMock()

    class _FakeCtx:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, *a):
            pass

    with patch("sqlalchemy.ext.asyncio.AsyncEngine.begin", return_value=_FakeCtx()), \
         patch("app.main.consume_generation_completed", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
