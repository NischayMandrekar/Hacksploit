"""Human-user authentication: Argon2id passwords, JWT access tokens, rotating opaque refresh tokens."""
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, Header
from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from platform_app.config import settings
from platform_app.db import get_db
from platform_app.errors import PlatformError
from platform_app.models import APIKey, AuthSession, User, utcnow
from platform_app.security import hash_key

_password_hasher = PasswordHasher()
# A dummy Argon2 hash keeps unknown-account login failures closer in cost to bad passwords.
_DUMMY_PASSWORD_HASH = _password_hasher.hash("not-a-real-user-password-for-timing-equalization")


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        return _password_hasher.verify(encoded, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def _jwt_secret() -> str:
    if len(settings.auth_jwt_secret) < 32:
        raise PlatformError(503, "authentication_not_configured", "Set AUTH_JWT_SECRET to a random secret of at least 32 characters.")
    return settings.auth_jwt_secret


def issue_access_token(user: User, now: datetime | None = None) -> tuple[str, datetime]:
    issued = now or utcnow()
    expires = issued + timedelta(seconds=settings.auth_access_token_seconds)
    payload = {
        "sub": str(user.id), "role": user.role, "iss": settings.auth_jwt_issuer,
        "aud": settings.auth_jwt_audience, "iat": issued, "exp": expires,
        "jti": uuid4().hex, "typ": "access",
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return token, expires


def new_refresh_token() -> str:
    return "plr_" + secrets.token_urlsafe(48)


def refresh_expiry(now: datetime | None = None) -> datetime:
    return (now or utcnow()) + timedelta(days=settings.auth_refresh_token_days)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    user_id: UUID | None
    api_key_id: UUID | None
    role: str
    scopes: frozenset[str]
    auth_kind: str


def current_user(authorization: str | None = Header(default=None, alias="Authorization"), db: Session = Depends(get_db)) -> User:
    if not authorization:
        raise PlatformError(401, "missing_bearer_token", "Provide an access token using Authorization: Bearer.")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise PlatformError(401, "missing_bearer_token", "Provide an access token using Authorization: Bearer.")
    token = token.strip()
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=["HS256"], issuer=settings.auth_jwt_issuer, audience=settings.auth_jwt_audience, options={"require": ["sub", "exp", "iat", "iss", "aud", "jti", "typ"]})
        if payload.get("typ") != "access":
            raise jwt.InvalidTokenError("wrong token type")
        user_id = UUID(payload["sub"])
    except PlatformError:
        raise
    except (jwt.PyJWTError, ValueError, KeyError):
        raise PlatformError(401, "invalid_access_token", "The access token is invalid or expired.")
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise PlatformError(401, "inactive_account", "The account is unavailable.")
    return user


def current_principal(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Principal:
    if authorization:
        user = current_user(authorization, db)
        scopes = frozenset({"jobs:create", "jobs:read", "jobs:cancel", "api-keys:manage"})
        if user.role == "admin":
            scopes = scopes | frozenset({"admin:users:read", "admin:users:write"})
        return Principal(user.id, None, user.role, scopes, "user")
    if x_api_key:
        key = db.execute(select(APIKey).where(APIKey.key_hash == hash_key(x_api_key))).scalar_one_or_none()
        now = utcnow()
        if not key or not key.is_active or key.revoked_at is not None or (key.expires_at and key.expires_at <= now):
            raise PlatformError(401, "invalid_api_key", "The API key is invalid, expired, or revoked.")
        # A user-owned key remains subordinate to the account's current active status.
        if key.user_id:
            user = db.get(User, key.user_id)
            if user is None or not user.is_active:
                raise PlatformError(401, "inactive_account", "The account is unavailable.")
            role = user.role
        else:
            role = "service"
        return Principal(key.user_id, key.id, role, frozenset(key.scopes), "api_key")
    raise PlatformError(401, "missing_credentials", "Provide a bearer access token or X-API-Key.")


def require_permission(scope: str):
    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if scope not in principal.scopes:
            raise PlatformError(403, "insufficient_scope", f"This identity requires the {scope} permission.")
        return principal
    return dependency


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise PlatformError(403, "admin_required", "Administrator privileges are required.")
    return user
