import pytest
from pydantic import ValidationError

from platform_app.config import Settings


def test_master_and_jwt_secrets_must_not_be_reused():
    shared = "s" * 64
    with pytest.raises(ValidationError, match="must be different secrets"):
        Settings(
            _env_file=None,
            app_master_key=shared,
            auth_jwt_secret=shared,
        )


def test_distinct_strong_master_and_jwt_secrets_are_accepted():
    settings = Settings(
        _env_file=None,
        app_master_key="m" * 64,
        auth_jwt_secret="j" * 64,
    )
    assert settings.app_master_key != settings.auth_jwt_secret
