from functools import lru_cache
import hmac
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def secrets_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode(), right.encode())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    app_env: str = "development"
    app_log_level: str = "INFO"
    app_master_key: str = ""
    auth_jwt_secret: str = ""
    auth_jwt_issuer: str = "local-backend-platform"
    auth_jwt_audience: str = "local-backend-platform-api"
    auth_access_token_seconds: int = Field(default=600, ge=60, le=3600)
    auth_refresh_token_days: int = Field(default=14, ge=1, le=90)
    database_url: str = "postgresql+psycopg://platform:platform_dev_only@localhost:5432/platform"
    worker_id: str = ""
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=30)
    worker_concurrency: int = Field(default=2, ge=1, le=32)
    worker_lease_seconds: int = Field(default=45, ge=5, le=600)
    worker_execution_timeout_seconds: int = Field(default=30, ge=1, le=300)
    worker_max_attempts: int = Field(default=3, ge=1, le=10)
    job_max_input_bytes: int = Field(default=65536, ge=1024, le=1048576)
    job_max_output_bytes: int = Field(default=65536, ge=1024, le=1048576)
    job_default_timeout_seconds: int = Field(default=10, ge=1, le=300)
    job_max_timeout_seconds: int = Field(default=30, ge=1, le=300)
    rabbitmq_url: str = ""
    rabbitmq_queue: str = "platform.job-wakeups"
    redis_url: str = ""
    kafka_bootstrap: str = ""
    kafka_topic: str = "platform.job-events"
    rate_limit_requests: int = Field(default=60, ge=1, le=10000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    prometheus_enabled: bool = True

    @model_validator(mode="after")
    def validate_worker_timing(self):
        if self.app_master_key and (len(self.app_master_key) < 32 or "replace_with" in self.app_master_key):
            raise ValueError("APP_MASTER_KEY must be a real random secret of at least 32 characters")
        if self.auth_jwt_secret and (len(self.auth_jwt_secret) < 32 or "replace_with" in self.auth_jwt_secret):
            raise ValueError("AUTH_JWT_SECRET must be a distinct real random secret of at least 32 characters")
        if self.app_master_key and self.auth_jwt_secret and secrets_compare(self.app_master_key, self.auth_jwt_secret):
            raise ValueError("APP_MASTER_KEY and AUTH_JWT_SECRET must be different secrets")
        if self.worker_lease_seconds <= self.worker_execution_timeout_seconds:
            raise ValueError("WORKER_LEASE_SECONDS must exceed WORKER_EXECUTION_TIMEOUT_SECONDS")
        if self.job_default_timeout_seconds > self.job_max_timeout_seconds:
            raise ValueError("JOB_DEFAULT_TIMEOUT_SECONDS cannot exceed JOB_MAX_TIMEOUT_SECONDS")
        if self.worker_execution_timeout_seconds < self.job_max_timeout_seconds:
            raise ValueError(
                "WORKER_EXECUTION_TIMEOUT_SECONDS must be at least JOB_MAX_TIMEOUT_SECONDS"
            )
        return self

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
