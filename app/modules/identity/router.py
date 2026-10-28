import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import PlatformError
from app.infrastructure.database import get_db
from app.modules.identity.auth import (
    _DUMMY_PASSWORD_HASH,
    current_user,
    hash_api_key,
    hash_password,
    issue_access_token,
    new_api_key,
    new_refresh_token,
    refresh_expiry,
    require_admin,
    token_hash,
    verify_password,
)
from app.modules.identity.models import APIKey, AuthSession, User, ensure_utc, utcnow
from app.modules.identity.schemas import (
    AuthTokens,
    KeyCreate,
    KeyIssued,
    LoginRequest,
    PasswordChangeRequest,
    RefreshRequest,
    RegisterRequest,
    RoleUpdate,
    UserView,
)

router = APIRouter(tags=["Identity & Authentication"])


def _create_session(db: Session, user: User, family_id: uuid.UUID | None = None) -> tuple[str, datetime]:
    raw_token = new_refresh_token()
    expires = refresh_expiry()
    session = AuthSession(
        user_id=user.id,
        family_id=family_id or uuid.uuid4(),
        token_hash=token_hash(raw_token),
        expires_at=expires,
        created_at=utcnow(),
    )
    db.add(session)
    return raw_token, expires


def _token_response(db: Session, user: User, family_id: uuid.UUID | None = None) -> AuthTokens:
    access, access_expires = issue_access_token(user)
    refresh, refresh_expires = _create_session(db, user, family_id)
    return AuthTokens(
        access_token=access,
        access_token_expires_at=access_expires,
        refresh_token=refresh,
        refresh_token_expires_at=refresh_expires,
        user=UserView.model_validate(user),
    )


@router.post("/auth/register", response_model=UserView, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    # Auto-promote first registered user in the database to admin for bootstrap convenience
    user_count = db.execute(select(func.count(User.id))).scalar_one()
    assigned_role = "admin" if user_count == 0 else "user"

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        role=assigned_role,
        is_active=True,
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise PlatformError(409, "email_taken", "An account with that email already exists.")

    return UserView.model_validate(user)


@router.post("/auth/login", response_model=AuthTokens)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.email == body.email)).scalar_one_or_none()
    encoded_hash = user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
    password_ok = verify_password(body.password, encoded_hash)

    if user is None or not user.is_active or not password_ok:
        raise PlatformError(401, "invalid_credentials", "Incorrect email or password.")

    tokens = _token_response(db, user)
    db.commit()
    return tokens


@router.post("/auth/refresh", response_model=AuthTokens)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)):
    hashed = token_hash(body.refresh_token)
    stored = db.execute(select(AuthSession).where(AuthSession.token_hash == hashed)).scalar_one_or_none()
    now = utcnow()

    if stored is None:
        raise PlatformError(401, "invalid_refresh_token", "The refresh token is invalid or expired.")

    if stored.revoked_at is not None:
        if stored.replaced_by_hash is not None:
            # Replay Grace Period: Check if rotation occurred within the last 30 seconds
            # to accommodate concurrent client retransmissions without locking out legitimate users.
            rotation_age = (now - ensure_utc(stored.revoked_at)).total_seconds()
            if rotation_age > 30:
                # Real replay attack outside grace window: revoke all sessions in family
                db.execute(
                    update(AuthSession)
                    .where(AuthSession.family_id == stored.family_id, AuthSession.revoked_at.is_(None))
                    .values(revoked_at=now)
                )
                db.commit()
                raise PlatformError(
                    401, "refresh_token_reused", "Token family revoked due to replay detection. Sign in again."
                )
            else:
                # Inside grace window: fetch the active replacement session
                active = db.execute(
                    select(AuthSession).where(
                        AuthSession.family_id == stored.family_id,
                        AuthSession.revoked_at.is_(None),
                        AuthSession.expires_at > now,
                    )
                ).scalar_one_or_none()
                user = db.get(User, stored.user_id)
                if active and user and user.is_active:
                    access, access_expires = issue_access_token(user, now)
                    return AuthTokens(
                        access_token=access,
                        access_token_expires_at=access_expires,
                        refresh_token=body.refresh_token,
                        refresh_token_expires_at=active.expires_at,
                        user=UserView.model_validate(user),
                    )

        raise PlatformError(401, "invalid_refresh_token", "The refresh token has been revoked.")

    if ensure_utc(stored.expires_at) <= now:
        stored.revoked_at = now
        db.commit()
        raise PlatformError(401, "expired_refresh_token", "The refresh token has expired.")

    user = db.get(User, stored.user_id)
    if user is None or not user.is_active:
        stored.revoked_at = now
        db.commit()
        raise PlatformError(401, "inactive_user", "User account is inactive.")

    new_raw = new_refresh_token()
    new_hash = token_hash(new_raw)
    new_expiry = refresh_expiry(now)

    stored.revoked_at = now
    stored.replaced_by_hash = new_hash

    db.add(
        AuthSession(
            user_id=user.id,
            family_id=stored.family_id,
            token_hash=new_hash,
            expires_at=new_expiry,
            created_at=now,
        )
    )
    access, access_expires = issue_access_token(user, now)
    db.commit()

    return AuthTokens(
        access_token=access,
        access_token_expires_at=access_expires,
        refresh_token=new_raw,
        refresh_token_expires_at=new_expiry,
        user=UserView.model_validate(user),
    )


@router.post("/auth/logout", status_code=204)
def logout(body: RefreshRequest, db: Session = Depends(get_db)):
    stored = db.execute(select(AuthSession).where(AuthSession.token_hash == token_hash(body.refresh_token))).scalar_one_or_none()
    if stored is not None and stored.revoked_at is None:
        stored.revoked_at = utcnow()
        db.commit()
    return Response(status_code=204)


@router.post("/auth/logout-all", status_code=204)
def logout_all(user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    db.commit()
    return Response(status_code=204)


@router.post("/auth/change-password", status_code=204)
def change_password(
    body: PasswordChangeRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(body.current_password, user.password_hash):
        raise PlatformError(401, "invalid_current_password", "Current password is incorrect.")
    if body.current_password == body.new_password:
        raise PlatformError(422, "password_unchanged", "New password must differ from current password.")

    user.password_hash = hash_password(body.new_password)
    user.updated_at = utcnow()
    # Revoke all existing refresh sessions on password change
    db.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    db.commit()
    return Response(status_code=204)


@router.get("/users/me", response_model=UserView)
def get_current_user_profile(user: User = Depends(current_user)):
    return UserView.model_validate(user)


@router.post("/api-keys", response_model=KeyIssued, status_code=201)
def create_api_key(
    body: KeyCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    raw_key = new_api_key()
    now = utcnow()
    if body.expires_at and ensure_utc(body.expires_at) <= now:
        raise PlatformError(422, "invalid_expiry", "API key expiry must be in the future.")

    key = APIKey(
        user_id=user.id,
        name=body.name,
        key_prefix=raw_key[:14],
        key_hash=hash_api_key(raw_key),
        scopes=body.scopes,
        created_at=now,
        expires_at=body.expires_at,
        is_active=True,
    )
    db.add(key)
    db.commit()
    db.refresh(key)

    return KeyIssued(
        id=key.id,
        name=key.name,
        key_prefix=key.key_prefix,
        raw_key=raw_key,
        scopes=key.scopes,
        created_at=key.created_at,
        expires_at=key.expires_at,
    )


@router.delete("/api-keys/{key_id}", status_code=204)
def revoke_api_key(
    key_id: uuid.UUID,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    key = db.get(APIKey, key_id)
    if not key:
        raise PlatformError(404, "not_found", "API key not found.")
    # Users can only revoke their own keys unless they are admin
    if key.user_id != user.id and user.role != "admin":
        raise PlatformError(403, "forbidden", "Cannot revoke an API key belonging to another account.")

    key.is_active = False
    key.revoked_at = utcnow()
    db.commit()
    return Response(status_code=204)


@router.post("/admin/users/{user_id}/role", response_model=UserView)
def update_user_role(
    user_id: uuid.UUID,
    body: RoleUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target_user = db.get(User, user_id)
    if not target_user:
        raise PlatformError(404, "not_found", "User not found.")

    target_user.role = body.role
    target_user.updated_at = utcnow()
    db.commit()
    db.refresh(target_user)
    return UserView.model_validate(target_user)
