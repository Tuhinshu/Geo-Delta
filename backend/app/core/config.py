"""
Configuration Settings for GeoDelta GEOINT Platform
Loads environment variables and enforces air-gapped system parameters.
"""

from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # System Identification
    PROJECT_NAME: str = "GeoDelta Automated GEOINT Platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "geodelta-tactical-secret-key-mod-sih26227-airgapped-auth"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # PostgreSQL + PostGIS 3.4
    POSTGRES_SERVER: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "geodelta_admin"
    POSTGRES_PASSWORD: str = "geodelta_secure_pass_2026"
    POSTGRES_DB: str = "geodelta_geoint"
    DATABASE_URL: str = "postgresql+asyncpg://geodelta_admin:geodelta_secure_pass_2026@db:5432/geodelta_geoint"
    SYNC_DATABASE_URL: str = "postgresql://geodelta_admin:geodelta_secure_pass_2026@db:5432/geodelta_geoint"

    # Redis Task Broker
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://redis:6379/0"

    # MinIO Local S3 Object Store
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_PUBLIC_ENDPOINT: str = "http://localhost:9000"
    MINIO_ROOT_USER: str = "minio_admin"
    MINIO_ROOT_PASSWORD: str = "minio_secure_pass_2026"
    MINIO_BUCKET_COGS: str = "cog-rasters"
    MINIO_BUCKET_DOSSIERS: str = "dossiers"
    MINIO_BUCKET_MODELS: str = "model-weights"
    MINIO_USE_SSL: bool = False

    # TiTiler Dynamic Raster Proxy
    TITILER_ENDPOINT: str = "http://titiler:8000"

    # GPU & Hardware Acceleration
    DEVICE: str = "cuda"
    CUDA_VISIBLE_DEVICES: str = "0"
    MODEL_WEIGHTS_DIR: str = "/models/weights"
    MAX_GPU_MEMORY_ALLOCATION_MB: int = 7500

    # Geospatial Invariant Limits (Physical Laws)
    MAX_PERMITTED_CLOUD_RATIO: float = 0.35  # Law 4: Abort if > 35% cloud cover
    MIN_ECC_CORRELATION_SCORE: float = 0.65  # Law 3: Abort if ECC < 0.65
    MIN_FEATURE_AREA_SQ_METERS: float = 50.0  # Drop clutter < 50 m^2

    # CORS
    ALLOWED_ORIGINS: Union[str, List[str]] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []


settings = Settings()
