import base64
import hashlib
import hmac
import json
import time
from typing import Any
from fastapi import Cookie, Header, HTTPException, Query, Request
from config import (
    ADMIN_EDIT_PASSWORD,
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    SESSION_COOKIE_NAME,
    SESSION_LIFETIME_SECONDS,
    SESSION_SECRET,
)


def encode_base64(value: bytes) -> str:
    """Encode bytes as unpadded URL-safe Base64 text."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def decode_base64(value: str) -> bytes:
    """Decode unpadded URL-safe Base64 text."""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_admin_session_token() -> str:
    """Create an expiring HMAC-signed admin session token."""
    if not SESSION_SECRET:
        raise HTTPException(status_code=503, detail="Admin sessions are not configured")
    payload = encode_base64(json.dumps({"expires_at": int(time.time()) + SESSION_LIFETIME_SECONDS}).encode("utf-8"))
    signature = hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest()
    return f"{payload}.{encode_base64(signature)}"


def is_valid_admin_credentials(username: Any, password: Any) -> bool:
    """Validate admin username and password against configured values."""
    if not isinstance(username, str) or not isinstance(password, str):
        return False
    # Check against configured username & password (default: 'admin' and 'admin')
    if hmac.compare_digest(username, ADMIN_USERNAME) and hmac.compare_digest(password, ADMIN_PASSWORD):
        return True
    # Support legacy admin password fallback
    if ADMIN_EDIT_PASSWORD and hmac.compare_digest(password, ADMIN_EDIT_PASSWORD):
        return True
    return False


def validate_session_token(session_token: str) -> None:
    """Validate HMAC signature and expiration for an admin session token."""
    if not SESSION_SECRET:
        raise HTTPException(status_code=503, detail="Admin sessions are not configured")
    try:
        payload, signature = session_token.split(".", maxsplit=1)
        expected_signature = hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(decode_base64(signature), expected_signature):
            raise ValueError("Invalid session signature")
        expires_at = json.loads(decode_base64(payload))["expires_at"]
        if not isinstance(expires_at, int) or expires_at <= time.time():
            raise ValueError("Expired session")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=401, detail="Admin session is invalid or expired") from None


async def require_admin_session(
    request: Request,
    username: str | None = Query(default=None, description="Admin username (default: 'admin')"),
    password: str | None = Query(default=None, description="Admin password (default: 'admin')"),
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    x_admin_username: str | None = Header(default=None, alias="X-Admin-Username"),
    x_admin_password: str | None = Header(default=None, alias="X-Admin-Password"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Authorize requests via username/password or a valid session cookie."""
    # Allow calling with a positional session_token string
    if isinstance(request, str):
        session_token = request

    user: str | None = username if isinstance(username, str) else None
    if not user and isinstance(x_admin_username, str):
        user = x_admin_username

    pwd: str | None = password if isinstance(password, str) else None
    if not pwd and isinstance(x_admin_password, str):
        pwd = x_admin_password

    auth_header: str | None = authorization if isinstance(authorization, str) else None
    token: str | None = session_token if isinstance(session_token, str) else None

    # Check request headers for plain username/password or basic auth
    if isinstance(request, Request):
        if not user:
            user = request.headers.get("username")
        if not pwd:
            pwd = request.headers.get("password")
        if not auth_header:
            auth_header = request.headers.get("authorization")

    # Decode HTTP Basic Auth if present
    if (not user or not pwd) and auth_header and auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:].strip()).decode("utf-8")
            if ":" in decoded:
                basic_user, basic_pwd = decoded.split(":", 1)
                user = user or basic_user
                pwd = pwd or basic_pwd
        except Exception:
            pass

    # Check JSON body for username/password in POST/PUT requests
    if (not user or not pwd) and isinstance(request, Request) and request.method in ("POST", "PUT", "PATCH"):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    data = json.loads(body_bytes)
                    if isinstance(data, dict):
                        if not user and isinstance(data.get("username"), str):
                            user = data["username"]
                        if not pwd and isinstance(data.get("password"), str):
                            pwd = data["password"]
            except Exception:
                pass

    # If username or password is provided, validate credentials
    if user is not None or pwd is not None:
        if is_valid_admin_credentials(user, pwd):
            return
        raise HTTPException(status_code=403, detail="Invalid admin credentials")

    # Otherwise, fall back to checking the session cookie
    if token:
        validate_session_token(token)
        return

    raise HTTPException(status_code=401, detail="Admin authentication is required")