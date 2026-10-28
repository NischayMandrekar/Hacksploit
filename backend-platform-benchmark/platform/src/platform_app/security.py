import hashlib, secrets
from datetime import datetime, timezone
from fastapi import Depends, Header
from sqlalchemy.orm import Session
from platform_app.config import settings
from platform_app.db import get_db
from platform_app.errors import PlatformError
from platform_app.models import APIKey


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def new_key() -> str:
    return "plk_" + secrets.token_urlsafe(32)

def require_master_key(x_master_key: str | None = Header(default=None)):
    expected = settings.app_master_key
    if not expected:
        raise PlatformError(503, "key_issuance_disabled", "Set APP_MASTER_KEY to enable API-key administration.")
    if not x_master_key or not secrets.compare_digest(x_master_key, expected):
        raise PlatformError(401, "invalid_master_key", "A valid master key is required.")
    return True

def current_api_key(x_api_key: str | None = Header(default=None), db: Session = Depends(get_db)) -> APIKey:
    if not x_api_key:
        raise PlatformError(401, "missing_api_key", "Provide an API key in X-API-Key.")
    key = db.query(APIKey).filter(APIKey.key_hash == hash_key(x_api_key)).first()
    now = datetime.now(timezone.utc)
    if not key or not key.is_active or key.revoked_at is not None or (key.expires_at and key.expires_at <= now):
        raise PlatformError(401, "invalid_api_key", "The API key is invalid, expired, or revoked.")
    return key

def require_scope(scope: str):
    def dependency(key: APIKey = Depends(current_api_key)) -> APIKey:
        if scope not in key.scopes:
            raise PlatformError(403, "insufficient_scope", f"This API key requires the {scope} scope.")
        return key
    return dependency
