import os
os.environ.setdefault("POSTGRES_SERVER", "localhost")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "test_db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-ci")

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


def build_mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.delete = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    result.scalar.return_value = 0
    db.execute.return_value = result
    return db


@pytest.fixture
def mock_db():
    return build_mock_db()


@pytest.fixture
async def client(mock_db):
    from app.main import app
    from app.db.session import get_db, engine

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    mock_conn = MagicMock()
    mock_conn.run_sync = AsyncMock()
    mock_conn.execute = AsyncMock()

    class _FakeCtx:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, *a):
            pass

    with patch("sqlalchemy.ext.asyncio.AsyncEngine.begin", return_value=_FakeCtx()):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()
