import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_create_comment_success(client, mock_db, auth_headers):
    post_id = uuid.uuid4()
    mock_comment = MagicMock()
    mock_comment.id = uuid.uuid4()
    mock_comment.post_id = post_id
    mock_comment.user_id = uuid.uuid4()
    mock_comment.content = "test comment"
    mock_comment.created_at = datetime.now(timezone.utc)
    mock_comment.updated_at = datetime.now(timezone.utc)
    mock_db.refresh = MagicMock(return_value=None)

    # After db.add + commit, refresh populates the object
    # The endpoint returns the new_comment object — we patch it via mock_db side effects
    resp = await client.post(
        "/api/v1/comments/",
        json={"post_id": str(post_id), "content": "test comment"},
        headers=auth_headers,
    )
    assert resp.status_code == 200


async def test_create_comment_requires_auth(client):
    post_id = uuid.uuid4()
    resp = await client.post(
        "/api/v1/comments/",
        json={"post_id": str(post_id), "content": "test"},
    )
    assert resp.status_code == 401


async def test_list_comments_empty(client, mock_db):
    post_id = uuid.uuid4()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    resp = await client.get(f"/api/v1/comments/{post_id}")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_comments_returns_data(client, mock_db):
    post_id = uuid.uuid4()
    c1 = MagicMock()
    c1.id = uuid.uuid4()
    c1.post_id = post_id
    c1.user_id = uuid.uuid4()
    c1.content = "hello"
    c1.created_at = datetime.now(timezone.utc)
    c1.updated_at = datetime.now(timezone.utc)
    mock_db.execute.return_value.scalars.return_value.all.return_value = [c1]

    resp = await client.get(f"/api/v1/comments/{post_id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
