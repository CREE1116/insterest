from app.core.security import get_password_hash, verify_password, create_access_token, create_refresh_token
from jose import jwt


def test_password_hash_is_not_plaintext():
    hashed = get_password_hash("MyPassword1!")
    assert hashed != "MyPassword1!"


def test_verify_correct_password():
    hashed = get_password_hash("MyPassword1!")
    assert verify_password("MyPassword1!", hashed) is True


def test_verify_wrong_password():
    hashed = get_password_hash("correct")
    assert verify_password("wrong", hashed) is False


def test_access_token_contains_sub():
    token = create_access_token("user-abc")
    payload = jwt.decode(token, "test-secret-key-for-ci", algorithms=["HS256"])
    assert payload["sub"] == "user-abc"
    assert payload["type"] == "access"


def test_refresh_token_type():
    token = create_refresh_token("user-abc")
    payload = jwt.decode(token, "test-secret-key-for-ci", algorithms=["HS256"])
    assert payload["type"] == "refresh"
