import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    EVENT_SLUG: str = "christ-sports-2025"

    STORAGE_BACKEND: str = "cloudinary"
    CLOUDINARY_URL: str | None = None

    # Prefer Railway-provided var name
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data.db")

    FAISS_METRIC: str = "cosine"
    MATCH_THRESHOLD: float = 0.35
    TOP_K: int = 50

    MAX_UPLOAD_MB: int = 8
    SECRET_KEY: str = "change-me"

    # ✅ Use /tmp/media by default
    MEDIA_ROOT: str = os.getenv("MEDIA_ROOT", "/tmp/media")


settings = Settings()

# Create required folders
os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
os.makedirs(os.path.join(settings.MEDIA_ROOT, "indices"), exist_ok=True)
os.makedirs(os.path.join(settings.MEDIA_ROOT, "embeddings"), exist_ok=True)
