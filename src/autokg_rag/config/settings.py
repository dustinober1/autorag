"""Typed runtime settings."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Global settings resolved from config, env, and CLI overrides."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: Path = Path("data/artifacts")
    log_level: str = "INFO"
    chunk_word_size: int = Field(default=120, ge=20)
    chunk_word_overlap: int = Field(default=20, ge=0)
    top_k: int = Field(default=3, ge=1)
