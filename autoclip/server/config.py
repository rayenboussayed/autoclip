from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = Field(default="0.0.0.0", validation_alias="HOST")  # noqa: S104
    port: int = Field(default=8000, validation_alias="PORT")
    max_file_size: int = Field(default=1024 * 1024 * 1024, validation_alias="MAX_FILE_SIZE")  # 1 GiB
    max_concurrent_jobs: int = Field(default=2, validation_alias="MAX_CONCURRENT_JOBS")
    auth_username: str = Field(default="", validation_alias="AUTH_USERNAME")
    auth_password: str = Field(default="", validation_alias="AUTH_PASSWORD")
    output_dir: Path = Field(default=BASE_DIR / ".server" / "outputs", validation_alias="OUTPUT_DIR")
    cache_dir: Path = Field(default=BASE_DIR / ".server" / "cache", validation_alias="CACHE_DIR")
    job_ttl_seconds: int = Field(default=3600, validation_alias="JOB_TTL_SECONDS")
    otp_endpoint: str = Field(default="", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_username and self.auth_password)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
