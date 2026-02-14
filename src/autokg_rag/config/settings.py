"""Typed runtime settings."""

from pathlib import Path
from typing import Literal

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
    embedding_provider: Literal["local_hash", "ollama"] = "local_hash"
    embedding_model: str = Field(default="bge-small-en-v1.5", min_length=1)
    ollama_base_url: str = Field(default="http://localhost:11434", min_length=1)
    ollama_timeout_seconds: float = Field(default=30.0, gt=0.0)
    reranker_enabled: bool = False
    reranker_model: str = Field(default="llama3:8b", min_length=1)
    reranker_candidate_k: int = Field(default=30, ge=1)
    answer_use_local: bool = False
    answer_model: str = Field(default="llama3", min_length=1)
    answer_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    answer_max_tokens: int = Field(default=512, ge=1)
    graph_max_depth: int = Field(default=2, ge=1)
    hybrid_vector_weight: float = Field(default=0.6, ge=0.0)
    hybrid_graph_weight: float = Field(default=0.4, ge=0.0)
    top_k: int = Field(default=8, ge=1)
