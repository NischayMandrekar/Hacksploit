"""User registration, login, refresh rotation, profile and API-key management."""
import re
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from platform_app.auth import (
    _DUMMY_PASSWORD_HASH, current_user, hash_password, issue_access_token, new_refresh_token,
    normalize_email, refresh_expiry, token_hash, verify_password,
)
from platform_app.cache import rate_limiter
from platform_app.config import settings
from platform_app.db import get_db
from platform_app.errors import PlatformError
from platform_app.models import APIKey, AuthSession, User, utcnow
from platform_app.schemas import KeyCreate, KeyIssued
from platform_app.security import hash_key, new_key

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        value = normalize_email(value)
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("must be a valid email address")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize(cls, value: str) -> str:
        return normalize_email(value)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class SessionView(BaseModel):
    family_id: UUID
    created_at: datetime
    last_rotated_at: datetime
    expires_at: datetime
    revoked: bool


class UserView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str
    role: str
    created_at: datetime


class AuthTokens(BaseModel):
    token_type: str = "Bearer"
    access_token: str
    access_token_expires_at: datetime
    refresh_token: str
    refresh_token_expires_at: datetime
    user: UserView


class ProfileUpdate(BaseModel):
    email: str | None = Field(default=None, min_length=3, max_length=320)

    @field_validator("email")
    @classmethod
    def normalize_email_field(cls, value):
        if value is None:
            return value
        normalized = normalize_email(value)
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized):
            raise ValueError("must be a valid email address")
        return normalized


class RoleUpdate(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, value):
        if value not in {"user", "admin"}:
            raise ValueError("role must be user or admin")
        return value


class APIKeySummary(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    scopes: list[str]
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None


def _check_auth_rate_limit(request: Request, action: str) -> None:
    # The API is not host-published; Nginx overwrites X-Real-IP at the trusted ingress.
    # Use that client IP rather than request.client (which is the Nginx container for
    # every request), otherwise all clients would share one in-process rate-limit bucket.
    client_ip = request.headers.get("X-Real-IP")
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allowed(f"auth:{action}:{client_ip}"):
        raise PlatformError(429, "rate_limit_exceeded", "Too many authentication requests; try again later.")


def _create_session(db: Session, user: User, family_id: UUID | None = None) -> tuple[str, datetime]:
    raw = new_refresh_token()
    expires = refresh_expiry()
    session = AuthSession(user_id=user.id, family_id=family_id or uuid4(), token_hash=token_hash(raw), expires_at=expires)
    db.add(session)
    return raw, expires


def _token_response(db: Session, user: User, family_id: UUID | None = None) -> AuthTokens:
    access, access_expires = issue_access_token(user)
    refresh, refresh_expires = _create_session(db, user, family_id)
    return AuthTokens(access_token=access, access_token_expires_at=access_expires,
                      refresh_token=refresh, refresh_token_expires_at=refresh_expires,
                      user=UserView.model_validate(user))


@router.post("/auth/register", response_model=UserView, status_code=201)
def register(body: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    _check_auth_rate_limit(request, "register")
    user = User(email=body.email, password_hash=hash_password(body.password), role="user", is_active=True)
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        # Email is uniquely indexed case-insensitively; never reveal account details.
        raise PlatformError(409, "email_unavailable", "An account with that email cannot be created.")
    return UserView.model_validate(user)


@router.post("/auth/login", response_model=AuthTokens)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    _check_auth_rate_limit(request, "login")
    user = db.execute(select(User).where(func.lower(User.email) == body.email)).scalar_one_or_none()
    # Use a generic response for unknown accounts and wrong passwords.
    encoded_hash = user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
    password_ok = verify_password(body.password, encoded_hash)
    if user is None or not user.is_active or not password_ok:
        raise PlatformError(401, "invalid_credentials", "Email or password is incorrect.")
    tokens = _token_response(db, user)
    db.commit()
    return tokens


@router.post("/auth/refresh", response_model=AuthTokens)
def refresh(body: RefreshRequest, request: Request, db: Session = Depends(get_db)):
    _check_auth_rate_limit(request, "refresh")
    hashed = token_hash(body.refresh_token)
    stored = db.execute(select(AuthSession).where(AuthSession.token_hash == hashed).with_for_update()).scalar_one_or_none()
    now = utcnow()
    if stored is None:
        raise PlatformError(401, "invalid_refresh_token", "The refresh token is invalid or expired.")
    if stored.revoked_at is not None:
        if stored.replaced_by_hash is not None:
            # A rotated token was replayed: revoke all sessions in its family.
            db.execute(update(AuthSession).where(AuthSession.family_id == stored.family_id, AuthSession.revoked_at.is_(None)).values(revoked_at=now))
            db.commit()
        raise PlatformError(401, "refresh_token_reused", "The refresh token is no longer valid; sign in again.")
    if stored.expires_at <= now:
        stored.revoked_at = now
        db.commit()
        raise PlatformError(401, "invalid_refresh_token", "The refresh token is invalid or expired.")
    user = db.get(User, stored.user_id)
    if user is None or not user.is_active:
        stored.revoked_at = now
        db.commit()
        raise PlatformError(401, "inactive_account", "The account is unavailable.")

    new_raw = new_refresh_token()
    new_hash = token_hash(new_raw)
    new_expiry = refresh_expiry(now)
    stored.revoked_at = now
    stored.replaced_by_hash = new_hash
    db.add(AuthSession(user_id=user.id, family_id=stored.family_id, token_hash=new_hash, expires_at=new_expiry))
    access, access_expiry = issue_access_token(user, now)
    db.commit()
    return AuthTokens(access_token=access, access_token_expires_at=access_expiry,
                      refresh_token=new_raw, refresh_token_expires_at=new_expiry,
                      user=UserView.model_validate(user))


@router.post("/auth/logout", status_code=204)
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    stored = db.execute(select(AuthSession).where(AuthSession.token_hash == token_hash(body.refresh_token)).with_for_update()).scalar_one_or_none()
    if stored is not None and stored.revoked_at is None:
        stored.revoked_at = utcnow()
        db.commit()
    return Response(status_code=204)


@router.post("/auth/logout-all", status_code=204)
def logout_all(user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.execute(update(AuthSession).where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None)).values(revoked_at=utcnow()))
    db.commit()
    return Response(status_code=204)


@router.post("/auth/change-password", status_code=204)
def change_password(body: PasswordChangeRequest, request: Request,
                    user: User = Depends(current_user), db: Session = Depends(get_db)):
    _check_auth_rate_limit(request, "change-password")
    if not verify_password(body.current_password, user.password_hash):
        raise PlatformError(401, "invalid_current_password", "The current password is incorrect.")
    if body.current_password == body.new_password:
        raise PlatformError(422, "password_unchanged", "The new password must differ from the current password.")
    user.password_hash = hash_password(body.new_password)
    user.updated_at = utcnow()
    now = utcnow()
    db.execute(update(AuthSession).where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None)).values(revoked_at=now))
    db.commit()
    return Response(status_code=204)


@router.get("/users/me/sessions", response_model=list[SessionView])
def list_sessions(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10000),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    # Aggregate rotation history in PostgreSQL rather than loading every token row into
    # application memory. Pagination keeps this endpoint bounded for long-lived accounts.
    rows = db.execute(
        select(
            AuthSession.family_id.label("family_id"),
            func.min(AuthSession.created_at).label("created_at"),
            func.max(AuthSession.created_at).label("last_rotated_at"),
            func.max(AuthSession.expires_at).label("expires_at"),
            func.bool_and(AuthSession.revoked_at.is_not(None)).label("revoked"),
        )
        .where(AuthSession.user_id == user.id)
        .group_by(AuthSession.family_id)
        .order_by(func.max(AuthSession.created_at).desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [SessionView(
        family_id=row.family_id, created_at=row.created_at,
        last_rotated_at=row.last_rotated_at, expires_at=row.expires_at,
        revoked=row.revoked,
    ) for row in rows]


@router.delete("/users/me/sessions/{family_id}", status_code=204)
def revoke_session(family_id: UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(select(AuthSession).where(
        AuthSession.user_id == user.id, AuthSession.family_id == family_id
    ).with_for_update()).scalars().all()
    if not rows:
        raise PlatformError(404, "not_found", "Session not found.")
    now = utcnow()
    for row in rows:
        if row.revoked_at is None:
            row.revoked_at = now
    db.commit()
    return Response(status_code=204)


@router.get("/users/me", response_model=UserView)
def get_profile(user: User = Depends(current_user)):
    return UserView.model_validate(user)


@router.patch("/users/me", response_model=UserView)
def update_profile(body: ProfileUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if "email" in body.model_fields_set and body.email is not None:
        user.email = body.email
        user.updated_at = utcnow()
        try:
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            raise PlatformError(409, "email_unavailable", "An account with that email cannot use this email.")
    return UserView.model_validate(user)


@router.post("/users/me/api-keys", response_model=KeyIssued, status_code=201)
def create_user_api_key(body: KeyCreate,
                        user: User = Depends(current_user), db: Session = Depends(get_db)):
    raw = new_key()
    now = utcnow()
    if body.expires_at and body.expires_at <= now:
        raise PlatformError(422, "invalid_expiry", "expires_at must be in the future.")
    key = APIKey(user_id=user.id, name=body.name, key_prefix=raw[:12], key_hash=hash_key(raw),
                 scopes=body.scopes, created_at=now, expires_at=body.expires_at)
    db.add(key)
    db.commit()
    db.refresh(key)
    return KeyIssued(id=key.id, name=key.name, key_prefix=key.key_prefix, raw_key=raw,
                     scopes=key.scopes, created_at=key.created_at, expires_at=key.expires_at)


@router.get("/users/me/api-keys", response_model=list[APIKeySummary])
def list_user_api_keys(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(select(APIKey).where(APIKey.user_id == user.id).order_by(APIKey.created_at.desc())).scalars().all()
    return [APIKeySummary(id=row.id, name=row.name, key_prefix=row.key_prefix, scopes=row.scopes,
                          created_at=row.created_at, expires_at=row.expires_at, revoked_at=row.revoked_at) for row in rows]


@router.delete("/users/me/api-keys/{key_id}", status_code=204)
def revoke_user_api_key(key_id: UUID, user: User = Depends(current_user), db: Session = Depends(get_db)):
    key = db.execute(select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id).with_for_update()).scalar_one_or_none()
    if key is None:
        raise PlatformError(404, "not_found", "API key not found.")
    if key.revoked_at is None:
        key.is_active = False
        key.revoked_at = utcnow()
        db.commit()
    return Response(status_code=204)


def _lock_admin_users(db: Session) -> list[User]:
    """Serialize role changes by locking current admin rows in stable order.

    Every role-change path must take these locks before modifying a role. This
    prevents two concurrent requests from independently demoting the last admins.
    """
    return list(db.execute(
        select(User).where(User.role == "admin").order_by(User.id).with_for_update()
    ).scalars().all())


def _prevent_last_admin_demotion(target: User, new_role: str, admins: list[User]) -> None:
    if target.role == "admin" and new_role != "admin" and len(admins) <= 1:
        raise PlatformError(409, "last_admin_required", "At least one administrator account must remain.")


@router.post("/admin/bootstrap/{user_id}/role", response_model=UserView)
def bootstrap_assign_role(user_id: UUID, body: RoleUpdate, x_master_key: str | None = Header(default=None, alias="X-Master-Key"), db: Session = Depends(get_db)):
    expected = settings.app_master_key
    if not expected or not x_master_key or not secrets.compare_digest(x_master_key, expected):
        raise PlatformError(401, "invalid_master_key", "A valid master key is required.")
    admins = _lock_admin_users(db)
    user = db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one_or_none()
    if user is None:
        raise PlatformError(404, "not_found", "User not found.")
    _prevent_last_admin_demotion(user, body.role, admins)
    user.role = body.role
    user.updated_at = utcnow()
    db.commit()
    db.refresh(user)
    return UserView.model_validate(user)


@router.get("/admin/users", response_model=list[UserView])
def admin_list_users(user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "admin":
        raise PlatformError(403, "admin_required", "Administrator privileges are required.")
    rows = db.execute(select(User).order_by(User.created_at.desc()).limit(200)).scalars().all()
    return [UserView.model_validate(row) for row in rows]


@router.patch("/admin/users/{user_id}/role", response_model=UserView)
def admin_update_role(user_id: UUID, body: RoleUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "admin":
        raise PlatformError(403, "admin_required", "Administrator privileges are required.")
    admins = _lock_admin_users(db)
    target = db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one_or_none()
    if target is None:
        raise PlatformError(404, "not_found", "User not found.")
    if target.id == user.id and body.role != "admin":
        raise PlatformError(409, "cannot_demote_self", "Use another administrator to change your own role.")
    _prevent_last_admin_demotion(target, body.role, admins)
    target.role = body.role
    target.updated_at = utcnow()
    db.commit()
    db.refresh(target)
    return UserView.model_validate(target)
