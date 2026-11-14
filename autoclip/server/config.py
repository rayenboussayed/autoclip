from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    max_file_size: int = Field(default=1024 * 1024 * 1024, validation_alias="MAX_FILE_SIZE")  # 1 GiB
    max_concurrent_jobs: int = Field(default=1, validation_alias="MAX_CONCURRENT_JOBS")
    max_pending_jobs: int = Field(default=8, validation_alias="MAX_PENDING_JOBS")
    cors_allow_origins: str = Field(default="", validation_alias="CORS_ALLOW_ORIGINS")  # comma-separated; empty = same-origin only
    auth_username: str = Field(default="", validation_alias="AUTH_USERNAME")
    auth_password: str = Field(default="", validation_alias="AUTH_PASSWORD")
    output_dir: Path = Field(default=BASE_DIR / ".server" / "outputs", validation_alias="OUTPUT_DIR")
    cache_dir: Path = Field(default=BASE_DIR / ".server" / "cache", validation_alias="CACHE_DIR")
    job_ttl_seconds: int = Field(default=3600, validation_alias="JOB_TTL_SECONDS")
    otp_endpoint: str = Field(default="", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_username and self.auth_password)

    @model_validator(mode="after")
    def _auth_policy(self) -> Settings:
        partial = bool(self.auth_username) != bool(self.auth_password)
        if partial:
            raise ValueError("AUTH_USERNAME and AUTH_PASSWORD must be set together (or both be empty)")
        if self.host != "127.0.0.1" and not self.auth_enabled:
            raise ValueError(f"refusing to bind {self.host} with auth disabled; set both AUTH_USERNAME and AUTH_PASSWORD")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
