from datetime import datetime
from typing import Any, Literal
from uuid import UUID
import math
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCOPES = {"jobs:create", "jobs:read", "jobs:cancel"}

class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    scopes: list[str] = Field(min_length=1)
    expires_at: datetime | None = None
    @field_validator("expires_at")
    @classmethod
    def expiry_must_be_timezone_aware(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("expires_at must include a timezone, for example Z or +00:00")
        return value
    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, value):
        unknown = set(value) - SCOPES
        if unknown: raise ValueError(f"Unknown scopes: {', '.join(sorted(unknown))}")
        if len(value) != len(set(value)): raise ValueError("Scopes must be unique")
        return value

class KeyIssued(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    raw_key: str
    scopes: list[str]
    created_at: datetime
    expires_at: datetime | None

class JobCreate(BaseModel):
    job_type: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.-]+$")
    input: dict[str, Any]

    @field_validator("input")
    @classmethod
    def validate_json_input(cls, value):
        # JSON parsers in Python may accept non-standard NaN/Infinity constants.
        # Walk iteratively to reject them and bound nesting before persistence.
        stack = [(value, 0)]
        while stack:
            current, depth = stack.pop()
            if depth > 32:
                raise ValueError("input nesting must not exceed 32 levels")
            if isinstance(current, dict):
                for key, child in current.items():
                    if not isinstance(key, str):
                        raise ValueError("input object keys must be strings")
                    stack.append((child, depth + 1))
            elif isinstance(current, list):
                stack.extend((child, depth + 1) for child in current)
            elif isinstance(current, float) and not math.isfinite(current):
                raise ValueError("input numbers must be finite; NaN and Infinity are not allowed")
        return value

    @field_validator("timeout_seconds", "max_attempts", mode="before")
    @classmethod
    def reject_boolean_integer_values(cls, value):
        # bool is a subclass of int in Python, but is not a meaningful timeout/attempt count.
        if isinstance(value, bool):
            raise ValueError("must be an integer, not a boolean")
        return value

    timeout_seconds: int | None = Field(default=None, ge=1, le=300)
    max_attempts: int | None = Field(default=None, ge=1, le=10)

class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    job_type: str
    status: str
    attempt_count: int
    max_attempts: int
    available_at: datetime
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    last_error: str | None = None

class CancelView(BaseModel):
    id: UUID
    status: str
    cancel_requested: bool
