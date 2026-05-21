import uuid
from unittest.mock import MagicMock
from app.models.interaction import Like


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_like_new_post(client, mock_db, auth_headers):
    post_id = uuid.uuid4()
    # No existing like → insert
    mock_db.execute.return_value.scalars.return_value.first.return_value = None

    resp = await client.post(f"/api/v1/interactions/{post_id}/like", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["liked"] is True


async def test_unlike_existing_post(client, mock_db, auth_headers):
    post_id = uuid.uuid4()
    # Existing like present → delete
    mock_db.execute.return_value.scalars.return_value.first.return_value = MagicMock(spec=Like)

    resp = await client.post(f"/api/v1/interactions/{post_id}/like", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["liked"] is False


async def test_like_requires_auth(client):
    post_id = uuid.uuid4()
    resp = await client.post(f"/api/v1/interactions/{post_id}/like")
    assert resp.status_code == 401


async def test_get_liked_posts_empty(client, mock_db, auth_headers):
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    resp = await client.get("/api/v1/interactions/me/liked", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_liked_posts_returns_ids(client, mock_db, auth_headers):
    ids = [uuid.uuid4(), uuid.uuid4()]
    mock_db.execute.return_value.scalars.return_value.all.return_value = ids

    resp = await client.get("/api/v1/interactions/me/liked", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2
