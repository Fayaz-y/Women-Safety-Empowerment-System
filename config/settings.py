"""
Women Safety AI — Central Settings Module
==========================================
Pydantic BaseSettings class that reads all configuration from
environment variables and a .env file. Single source of truth
for all tuneable parameters across the entire system.

Usage:
    from config.settings import settings
    print(settings.device)
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Database ──
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/women_safety",
        description="PostgreSQL connection string",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )

    # ── Device / Inference ──
    device: str = Field(default="cuda", description="PyTorch device (cuda | cpu)")
    use_fp16: bool = Field(default=True, description="Enable FP16 mixed precision")
    model_dir: str = Field(default="./models", description="Model weights directory")
    snapshot_dir: str = Field(
        default="./data/snapshots", description="Incident snapshot images"
    )
    clip_dir: str = Field(
        default="./data/clips", description="Incident video clips"
    )

    # ── Detection Thresholds ──
    fusion_threshold: float = Field(
        default=0.75, description="Fusion score trigger threshold"
    )
    proximity_radius_px: int = Field(
        default=150, description="Male proximity radius in pixels"
    )
    isolation_seconds: float = Field(
        default=5.0, description="Seconds before a woman is considered 'lone'"
    )
    alert_cooldown_seconds: int = Field(
        default=30, description="Minimum seconds between alerts per track"
    )

    # ── Twilio SMS ──
    twilio_account_sid: str = Field(default="", description="Twilio Account SID")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token")
    twilio_from_number: str = Field(default="", description="Twilio sender phone")
    alert_phone_number: str = Field(default="", description="Recipient phone number")

    # ── Camera ──
    camera_toggle: int = Field(
        default=0,
        description="0 = laptop webcam, 1 = external USB cam, 2 = video file",
    )
    camera_source_path: str = Field(
        default="",
        description="Path to video file (used when camera_toggle=2)",
    )

    # ── API / Networking ──
    api_url: str = Field(
        default="http://localhost:8000",
        description="Backend API URL (for Cloudflare tunneling)",
    )
    allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated CORS allowed origins",
    )

    # ── Model Inference (Remote) ──
    use_remote_model: bool = Field(
        default=False,
        description="If True, use remote model via MODEL_API_URL instead of loading locally",
    )
    model_api_url: str = Field(
        default="http://localhost:9000",
        description="Remote model inference server URL (e.g., https://model.yourdomain.com)",
    )
    model_api_timeout: int = Field(
        default=30,
        description="Timeout in seconds for model API requests",
    )

    # ── JWT Auth ──
    jwt_secret_key: str = Field(
        default="change-me-in-production-minimum-32-chars",
        description="JWT signing key",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(
        default=480, description="JWT expiry in minutes (8 hours)"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# ── Module-level singleton ──────────────────────────────────────────────────
# Import as: from config.settings import settings
settings = Settings()
