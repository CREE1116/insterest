import uuid
from unittest.mock import MagicMock, AsyncMock
from app.core.security import get_password_hash


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_register_new_user(client, mock_db):
    # DB returns no existing user → registration proceeds
    mock_db.execute.return_value.scalars.return_value.first.return_value = None

    resp = await client.post("/api/v1/auth/register", json={
        "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
        "password": "StrongPass123!",
        "nickname": "tester",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["nickname"] == "tester"


async def test_register_duplicate_email_returns_400(client, mock_db):
    existing = MagicMock()
    existing.email = "dupe@example.com"
    mock_db.execute.return_value.scalars.return_value.first.return_value = existing

    resp = await client.post("/api/v1/auth/register", json={
        "email": "dupe@example.com",
        "password": "StrongPass123!",
        "nickname": "tester",
    })
    assert resp.status_code == 400


async def test_login_success(client, mock_db):
    pw = "StrongPass123!"
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    mock_user.email = "user@example.com"
    mock_user.hashed_password = get_password_hash(pw)
    mock_user.is_active = True
    mock_db.execute.return_value.scalars.return_value.first.return_value = mock_user

    resp = await client.post("/api/v1/auth/login", data={
        "username": "user@example.com",
        "password": pw,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


async def test_login_wrong_password_returns_400(client, mock_db):
    mock_user = MagicMock()
    mock_user.id = uuid.uuid4()
    mock_user.hashed_password = get_password_hash("correct")
    mock_user.is_active = True
    mock_db.execute.return_value.scalars.return_value.first.return_value = mock_user

    resp = await client.post("/api/v1/auth/login", data={
        "username": "user@example.com",
        "password": "wrong",
    })
    assert resp.status_code == 400


async def test_login_no_user_returns_400(client, mock_db):
    mock_db.execute.return_value.scalars.return_value.first.return_value = None

    resp = await client.post("/api/v1/auth/login", data={
        "username": "nobody@example.com",
        "password": "pass",
    })
    assert resp.status_code == 400
