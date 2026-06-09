import uuid
import pytest
from app.models.media import ContentType

async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

async def test_upload_photo_success(client, mock_db):
    author_id = uuid.uuid4()
    
    # We simulate files by sending them via multipart/form-data
    files = {
        "image_file": ("photo.jpg", b"fake_image_bytes", "image/jpeg")
    }
    data = {
        "author_id": str(author_id),
        "body": "This is a photo upload",
        "hashtags": "tag1,tag2",
        "prompt": "a beautiful landscape"
    }

    resp = await client.post("/api/v1/upload/", data=data, files=files)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    assert "content_id" in res_data
    assert "post_id" in res_data
    assert res_data["content_type"] == ContentType.PHOTO

async def test_upload_video_success(client, mock_db):
    author_id = uuid.uuid4()
    files = {
        "video_file": ("video.mp4", b"fake_video_bytes", "video/mp4")
    }
    data = {
        "author_id": str(author_id),
        "body": "This is a video upload"
    }

    resp = await client.post("/api/v1/upload/", data=data, files=files)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    assert res_data["content_type"] == ContentType.VIDEO

async def test_upload_photo_sound_success(client, mock_db):
    author_id = uuid.uuid4()
    files = {
        "image_file": ("photo.jpg", b"fake_image_bytes", "image/jpeg"),
        "sound_file": ("sound.mp3", b"fake_sound_bytes", "audio/mpeg")
    }
    data = {
        "author_id": str(author_id),
        "body": "This is a photo + sound upload",
        "hashtags": '["music", "sunset"]'
    }

    resp = await client.post("/api/v1/upload/", data=data, files=files)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    assert res_data["content_type"] == ContentType.PHOTO_SOUND

async def test_upload_sound_without_image_fails(client, mock_db):
    author_id = uuid.uuid4()
    files = {
        "sound_file": ("sound.mp3", b"fake_sound_bytes", "audio/mpeg")
    }
    data = {
        "author_id": str(author_id),
        "body": "Invalid upload with sound only"
    }

    resp = await client.post("/api/v1/upload/", data=data, files=files)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Sound file must be uploaded with an image file."

async def test_upload_no_media_fails(client, mock_db):
    author_id = uuid.uuid4()
    data = {
        "author_id": str(author_id),
        "body": "Invalid upload with no media at all"
    }

    resp = await client.post("/api/v1/upload/", data=data)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "At least one media file (image or video) is required."
