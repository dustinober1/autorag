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
    sentence_window_size: int = Field(default=5, ge=1)
    semantic_similarity_breakpoint: float = Field(default=0.2, ge=0.0, le=1.0)
    embedding_dim: int = Field(default=256, ge=8)
    top_k: int = Field(default=3, ge=1)
