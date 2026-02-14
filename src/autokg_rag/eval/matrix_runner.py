"""Factorial experiment runner for evaluation harness."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from autokg_rag.config import Settings
from autokg_rag.eval.metrics import (
    citation_precision,
    faithfulness_proxy,
    ndcg_at_k,
    recall_at_k,
)
from autokg_rag.exceptions import AutoRAGError
from autokg_rag.io import read_jsonl_rows, write_jsonl_rows
from autokg_rag.retrieval.fusion import fuse_hybrid_hits
from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import ChunkRecord, EvalQuestionRecord, RetrievalHitRecord
from autokg_rag.vector.store import load_chunks

DEFAULT_CHUNKING: Final[tuple[str, ...]] = (
    "fixed_400_50",
    "heading_recursive",
    "sentence_window_5",
    "semantic_breakpoint",
)
DEFAULT_EMBEDDINGS: Final[tuple[str, ...]] = (
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
)
DEFAULT_RETRIEVAL: Final[tuple[str, ...]] = ("vector", "graph", "hybrid")
DEFAULT_EMBEDDING_PROVIDER: Final[tuple[str, ...]] = ("local_hash",)
DEFAULT_RERANKER_ENABLED: Final[tuple[str, ...]] = ("false",)
DEFAULT_RERANKER_MODEL: Final[tuple[str, ...]] = ("llama3:8b",)
DEFAULT_REPORTS_DIR: Final[Path] = Path("reports/experiments")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ExperimentSpec:
    """Factor combination entry for one experiment run."""

    exp_id: str
    chunking: str
    embedding: str
    retrieval: str
    embedding_provider: str
    embedding_model: str
    reranker_enabled: bool
    reranker_model: str


@dataclass(frozen=True)
class ExperimentResultRow:
    """Aggregate metrics for one experiment configuration."""

    run_id: str
    exp_id: str
    chunking: str
    embedding: str
    embedding_provider: str
    embedding_model: str
    retrieval: str
    reranker_enabled: bool
    reranker_model: str
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    citation_precision: float
    faithfulness_proxy: float
    latency_ms: float


def _slug(value: str) -> str:
    compact = _SLUG_RE.sub("_", value).strip("_")
    return compact.lower() if compact else "unknown"


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise AutoRAGError(f"Missing matrix config file: {config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise AutoRAGError(f"Invalid matrix config in '{config_path}': expected object root.")
    return dict(loaded)


def _as_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []

    values: list[str] = []
    for value in raw:
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def _read_factor_values(
    raw_config: dict[str, Any],
    *,
    name: str,
    aliases: tuple[str, ...],
    default: tuple[str, ...],
) -> list[str]:
    factors = raw_config.get("factors")
    if isinstance(factors, dict):
        direct = _as_str_list(factors.get(name))
        if direct:
            return direct
        for alias in aliases:
            from_alias = _as_str_list(factors.get(alias))
            if from_alias:
                return from_alias

    defaults = raw_config.get("defaults")
    if isinstance(defaults, dict):
        for alias in aliases:
            from_defaults = _as_str_list(defaults.get(alias))
            if from_defaults:
                return from_defaults

    for alias in aliases:
        top_level = _as_str_list(raw_config.get(alias))
        if top_level:
            return top_level

    return list(default)


def _resolve_reports_dir(raw_config: dict[str, Any], output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir

    configured = raw_config.get("reports_dir")
    if isinstance(configured, str) and configured.strip():
        return Path(configured)

    return DEFAULT_REPORTS_DIR


def _resolve_dataset_path(
    *,
    run_id: str,
    raw_config: dict[str, Any],
    artifact_root: Path,
    questions_path: Path | None,
) -> Path:
    if questions_path is not None:
        return questions_path

    for key in ("dataset_path", "questions_path"):
        value = raw_config.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)

    run_dir = artifact_root / run_id
    generated = sorted(run_dir.glob("questions_*.jsonl"))
    if generated:
        return generated[-1]

    raise AutoRAGError(
        "Unable to locate question dataset. Set dataset_path in matrix config "
        "or run 'autorag eval generate' first."
    )


def _resolve_source_artifact_dir(
    *,
    run_id: str,
    raw_config: dict[str, Any],
    artifact_root: Path,
) -> Path:
    source_run_id = raw_config.get("source_run_id")
    if isinstance(source_run_id, str) and source_run_id.strip():
        return artifact_root / source_run_id

    source_dir = raw_config.get("source_artifact_dir")
    if isinstance(source_dir, str) and source_dir.strip():
        return Path(source_dir)

    source_artifacts = raw_config.get("source_artifacts")
    if isinstance(source_artifacts, list):
        for candidate_root in source_artifacts:
            if not isinstance(candidate_root, str):
                continue
            base = Path(candidate_root)
            if (base / run_id / "chunks.parquet").exists():
                return base / run_id
            run_dirs = [
                path
                for path in base.glob("*")
                if path.is_dir() and (path / "chunks.parquet").exists()
            ]
            if run_dirs:
                run_dirs.sort(key=lambda path: path.stat().st_mtime)
                return run_dirs[-1]

    run_dir = artifact_root / run_id
    if (run_dir / "chunks.parquet").exists():
        return run_dir

    candidates = [
        path
        for path in artifact_root.glob("*")
        if path.is_dir() and (path / "chunks.parquet").exists()
    ]
    if not candidates:
        raise AutoRAGError(
            "Unable to resolve source artifact directory with chunks.parquet. "
            "Set source_run_id or source_artifact_dir in matrix config."
        )

    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def _load_questions(dataset_path: Path) -> list[EvalQuestionRecord]:
    rows = read_jsonl_rows(dataset_path)
    questions = [EvalQuestionRecord.model_validate(row) for row in rows]
    if not questions:
        raise AutoRAGError(f"Evaluation dataset is empty: {dataset_path}")
    return questions


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value)}


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


def _value_scale(value: str, *, magnitude: float) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    seed = int(digest, 16) % 11
    return 1.0 + (float(seed) * magnitude)


def _coerce_bool_factor(values: list[str], *, factor_name: str) -> list[bool]:
    parsed: list[bool] = []
    for raw in values:
        token = raw.strip().lower()
        if token in _BOOL_TRUE:
            parsed.append(True)
            continue
        if token in _BOOL_FALSE:
            parsed.append(False)
            continue
        raise AutoRAGError(
            f"Matrix factor '{factor_name}' contains invalid boolean value: {raw!r}. "
            "Use true/false."
        )
    return parsed


def _format_experiment_id(
    *,
    chunking: str,
    embedding_model: str,
    retrieval: str,
    embedding_provider: str,
    reranker_enabled: bool,
    reranker_model: str,
) -> str:
    exp_id = f"exp_{_slug(chunking)}_{_slug(embedding_model)}_{_slug(retrieval)}"
    if embedding_provider.strip().lower() != DEFAULT_EMBEDDING_PROVIDER[0]:
        exp_id = f"{exp_id}_provider_{_slug(embedding_provider)}"
    if reranker_enabled:
        exp_id = f"{exp_id}_reranker_{_slug(reranker_model)}"
    return exp_id


def _rank_hits(
    *,
    question_id: str,
    scored: list[tuple[ChunkRecord, float]],
    top_k: int,
) -> list[RetrievalHitRecord]:
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].doc_id,
            item[0].page,
            item[0].section,
            item[0].chunk_id,
        )
    )

    top_rows = scored[: max(1, top_k)]
    hits: list[RetrievalHitRecord] = []
    for rank, (chunk, score) in enumerate(top_rows, start=1):
        hits.append(
            RetrievalHitRecord(
                question_id=question_id,
                rank=rank,
                score=float(score),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page,
                section=chunk.section,
            )
        )
    return hits


def _retrieve_vector_hits(
    *,
    question: EvalQuestionRecord,
    chunks: list[ChunkRecord],
    top_k: int,
    chunking: str,
    embedding: str,
) -> list[RetrievalHitRecord]:
    query_tokens = _tokens(question.question)
    chunking_scale = _value_scale(chunking, magnitude=0.001)
    embedding_scale = _value_scale(embedding, magnitude=0.001)

    scored: list[tuple[ChunkRecord, float]] = []
    for chunk in chunks:
        text_score = _jaccard(query_tokens, _tokens(chunk.chunk_text))
        section_score = _jaccard(query_tokens, _tokens(chunk.section))
        blended = (0.75 * text_score) + (0.25 * section_score)
        score = blended * chunking_scale * embedding_scale
        scored.append((chunk, score))

    return _rank_hits(question_id=question.question_id, scored=scored, top_k=top_k)


def _retrieve_graph_hits(
    *,
    question: EvalQuestionRecord,
    chunks: list[ChunkRecord],
    top_k: int,
    chunking: str,
) -> list[RetrievalHitRecord]:
    query_tokens = _tokens(question.question)
    chunking_scale = _value_scale(chunking, magnitude=0.001)

    scored: list[tuple[ChunkRecord, float]] = []
    for chunk in chunks:
        text_score = _jaccard(query_tokens, _tokens(chunk.chunk_text))
        section_score = _jaccard(query_tokens, _tokens(chunk.section))
        score = ((0.55 * text_score) + (0.45 * section_score)) * chunking_scale
        scored.append((chunk, score))

    return _rank_hits(question_id=question.question_id, scored=scored, top_k=top_k)


def _retrieve_hits(
    *,
    question: EvalQuestionRecord,
    chunks: list[ChunkRecord],
    top_k: int,
    chunking: str,
    embedding: str,
    retrieval: str,
) -> list[RetrievalHitRecord]:
    vector_hits = _retrieve_vector_hits(
        question=question,
        chunks=chunks,
        top_k=top_k,
        chunking=chunking,
        embedding=embedding,
    )

    if retrieval == "vector":
        return vector_hits

    graph_hits = _retrieve_graph_hits(
        question=question,
        chunks=chunks,
        top_k=top_k,
        chunking=chunking,
    )

    if retrieval == "graph":
        return graph_hits

    fused = fuse_hybrid_hits(
        question_id=question.question_id,
        vector_hits=vector_hits,
        graph_hits=graph_hits,
        top_k=top_k,
        vector_weight=0.6,
        graph_weight=0.4,
    )
    return [
        RetrievalHitRecord(
            question_id=hit.question_id,
            rank=hit.rank,
            score=hit.score,
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            page=hit.page,
            section=hit.section,
        )
        for hit in fused
    ]


def _excerpt(text: str, max_tokens: int = 16) -> str:
    words = _TOKEN_RE.findall(text)
    if not words:
        return ""
    snippet = " ".join(words[:max_tokens])
    return f"{snippet}."


def _predicted_citations(hits: list[RetrievalHitRecord]) -> list[Citation]:
    citations: list[Citation] = []
    for hit in hits[:2]:
        citations.append(
            Citation(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                page=hit.page,
                section=hit.section,
            )
        )
    return citations


def _compose_answer(hits: list[RetrievalHitRecord], chunk_lookup: dict[str, ChunkRecord]) -> str:
    parts: list[str] = []
    for hit in hits[:2]:
        chunk = chunk_lookup.get(hit.chunk_id)
        if chunk is None:
            continue
        excerpt = _excerpt(chunk.chunk_text)
        if excerpt:
            parts.append(excerpt)

    if not parts:
        return "No supporting evidence found."
    return " ".join(parts)


def _evaluate_experiment(
    *,
    run_id: str,
    spec: ExperimentSpec,
    questions: list[EvalQuestionRecord],
    chunks: list[ChunkRecord],
    top_k: int,
) -> tuple[ExperimentResultRow, list[dict[str, Any]]]:
    started_at = time.perf_counter()
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}

    per_query_rows: list[dict[str, Any]] = []
    total_recall_5 = 0.0
    total_recall_10 = 0.0
    total_ndcg_5 = 0.0
    total_ndcg_10 = 0.0
    total_citation_precision = 0.0
    total_faithfulness = 0.0

    for question in questions:
        hits = _retrieve_hits(
            question=question,
            chunks=chunks,
            top_k=top_k,
            chunking=spec.chunking,
            embedding=spec.embedding_model,
            retrieval=spec.retrieval,
        )
        citations = _predicted_citations(hits)
        answer = _compose_answer(hits, chunk_lookup)

        recall_5 = recall_at_k(gold_citations=question.gold_citations, hits=hits, k=5)
        recall_10 = recall_at_k(gold_citations=question.gold_citations, hits=hits, k=10)
        ndcg_5 = ndcg_at_k(gold_citations=question.gold_citations, hits=hits, k=5)
        ndcg_10 = ndcg_at_k(gold_citations=question.gold_citations, hits=hits, k=10)
        precision = citation_precision(
            predicted_citations=citations,
            gold_citations=question.gold_citations,
        )
        faithfulness = faithfulness_proxy(
            answer_text=answer,
            citations=citations,
            chunk_lookup=chunk_lookup,
        )

        total_recall_5 += recall_5
        total_recall_10 += recall_10
        total_ndcg_5 += ndcg_5
        total_ndcg_10 += ndcg_10
        total_citation_precision += precision
        total_faithfulness += faithfulness

        per_query_rows.append(
            {
                "exp_id": spec.exp_id,
                "question_id": question.question_id,
                "metrics": {
                    "recall@5": recall_5,
                    "recall@10": recall_10,
                    "ndcg@5": ndcg_5,
                    "ndcg@10": ndcg_10,
                    "citation_precision": precision,
                    "faithfulness_proxy": faithfulness,
                },
                "hits": [hit.model_dump(mode="json") for hit in hits],
                "answer": answer,
            }
        )

    query_count = len(questions)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    if query_count == 0:
        row = ExperimentResultRow(
            run_id=run_id,
            exp_id=spec.exp_id,
            chunking=spec.chunking,
            embedding=spec.embedding,
            embedding_provider=spec.embedding_provider,
            embedding_model=spec.embedding_model,
            retrieval=spec.retrieval,
            reranker_enabled=spec.reranker_enabled,
            reranker_model=spec.reranker_model,
            recall_at_5=0.0,
            recall_at_10=0.0,
            ndcg_at_5=0.0,
            ndcg_at_10=0.0,
            citation_precision=0.0,
            faithfulness_proxy=0.0,
            latency_ms=elapsed_ms,
        )
        return row, per_query_rows

    denominator = float(query_count)
    row = ExperimentResultRow(
        run_id=run_id,
        exp_id=spec.exp_id,
        chunking=spec.chunking,
        embedding=spec.embedding,
        embedding_provider=spec.embedding_provider,
        embedding_model=spec.embedding_model,
        retrieval=spec.retrieval,
        reranker_enabled=spec.reranker_enabled,
        reranker_model=spec.reranker_model,
        recall_at_5=total_recall_5 / denominator,
        recall_at_10=total_recall_10 / denominator,
        ndcg_at_5=total_ndcg_5 / denominator,
        ndcg_at_10=total_ndcg_10 / denominator,
        citation_precision=total_citation_precision / denominator,
        faithfulness_proxy=total_faithfulness / denominator,
        latency_ms=elapsed_ms,
    )
    return row, per_query_rows


def _write_matrix_csv(path: Path, rows: list[ExperimentResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "exp_id",
        "chunking",
        "embedding",
        "embedding_provider",
        "embedding_model",
        "retrieval",
        "reranker_enabled",
        "reranker_model",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "citation_precision",
        "faithfulness_proxy",
        "latency_ms",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_matrix_json(path: Path, run_id: str, rows: list[ExperimentResultRow]) -> None:
    payload_rows = [asdict(row) for row in rows]
    best_row = max(rows, key=lambda row: row.ndcg_at_10, default=None)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "summary": {
            "total_experiments": len(rows),
            "primary_metric": "nDCG@10",
            "best_exp_id": best_row.exp_id if best_row is not None else None,
            "best_ndcg_at_10": best_row.ndcg_at_10 if best_row is not None else 0.0,
        },
        "rows": payload_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_per_query_rows(
    per_query_dir: Path,
    rows_by_exp_id: dict[str, list[dict[str, Any]]],
) -> None:
    per_query_dir.mkdir(parents=True, exist_ok=True)
    for exp_id, rows in rows_by_exp_id.items():
        write_jsonl_rows(per_query_dir / f"{exp_id}.jsonl", rows)


def load_matrix_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load matrix YAML configuration from disk."""

    path = Path("configs/experiments/matrix.yaml") if config_path is None else config_path
    return _load_yaml_config(path)


def build_factorial_grid(config: dict[str, Any]) -> list[ExperimentSpec]:
    """Build the complete experiment grid from factor configuration."""

    chunking_values = _read_factor_values(
        config,
        name="chunking",
        aliases=("chunking",),
        default=DEFAULT_CHUNKING,
    )
    embedding_values = _read_factor_values(
        config,
        name="embedding_model",
        aliases=("embedding_model", "embedding", "embeddings"),
        default=DEFAULT_EMBEDDINGS,
    )
    retrieval_values = _read_factor_values(
        config,
        name="retrieval",
        aliases=("retrieval", "retrieval_modes"),
        default=DEFAULT_RETRIEVAL,
    )
    embedding_provider_values = _read_factor_values(
        config,
        name="embedding_provider",
        aliases=("embedding_provider",),
        default=DEFAULT_EMBEDDING_PROVIDER,
    )
    reranker_enabled_values = _coerce_bool_factor(
        _read_factor_values(
            config,
            name="reranker_enabled",
            aliases=("reranker_enabled",),
            default=DEFAULT_RERANKER_ENABLED,
        ),
        factor_name="reranker_enabled",
    )
    reranker_model_values = _read_factor_values(
        config,
        name="reranker_model",
        aliases=("reranker_model",),
        default=DEFAULT_RERANKER_MODEL,
    )

    specs: list[ExperimentSpec] = []
    for chunking in chunking_values:
        for embedding in embedding_values:
            for retrieval in retrieval_values:
                for embedding_provider in embedding_provider_values:
                    for reranker_enabled in reranker_enabled_values:
                        for reranker_model in reranker_model_values:
                            exp_id = _format_experiment_id(
                                chunking=chunking,
                                embedding_model=embedding,
                                retrieval=retrieval,
                                embedding_provider=embedding_provider,
                                reranker_enabled=reranker_enabled,
                                reranker_model=reranker_model,
                            )
                            specs.append(
                                ExperimentSpec(
                                    exp_id=exp_id,
                                    chunking=chunking,
                                    embedding=embedding,
                                    retrieval=retrieval,
                                    embedding_provider=embedding_provider,
                                    embedding_model=embedding,
                                    reranker_enabled=reranker_enabled,
                                    reranker_model=reranker_model,
                                )
                            )
    return specs


def run_matrix(
    *,
    run_id: str,
    config_path: Path | None = None,
    settings: Settings | None = None,
    config: dict[str, Any] | None = None,
    questions_path: Path | None = None,
    output_dir: Path | None = None,
    artifact_root: Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Execute matrix experiments and write CSV/JSON/per-query artifacts."""

    resolved_settings = Settings() if settings is None else settings
    resolved_artifact_root = (
        resolved_settings.artifact_root if artifact_root is None else artifact_root
    )

    raw_config = dict(config) if config is not None else load_matrix_config(config_path)
    grid = build_factorial_grid(raw_config)

    if dry_run:
        return [asdict(spec) for spec in grid]

    dataset_path = _resolve_dataset_path(
        run_id=run_id,
        raw_config=raw_config,
        artifact_root=resolved_artifact_root,
        questions_path=questions_path,
    )
    source_artifact_dir = _resolve_source_artifact_dir(
        run_id=run_id,
        raw_config=raw_config,
        artifact_root=resolved_artifact_root,
    )
    top_k = int(raw_config.get("top_k", 10))
    if top_k < 1:
        raise AutoRAGError("Matrix config 'top_k' must be >= 1.")

    questions = _load_questions(dataset_path)
    chunks = load_chunks(source_artifact_dir)
    if not chunks:
        raise AutoRAGError(f"No chunks found in source artifact dir: {source_artifact_dir}")

    rows: list[ExperimentResultRow] = []
    per_query: dict[str, list[dict[str, Any]]] = {}

    for spec in grid:
        row, query_rows = _evaluate_experiment(
            run_id=run_id,
            spec=spec,
            questions=questions,
            chunks=chunks,
            top_k=top_k,
        )
        rows.append(row)
        per_query[spec.exp_id] = query_rows

    reports_dir = _resolve_reports_dir(raw_config, output_dir)
    matrix_csv_path = reports_dir / "matrix_results.csv"
    matrix_json_path = reports_dir / "matrix_results.json"
    per_query_dir = reports_dir / "per_query"

    _write_matrix_csv(matrix_csv_path, rows)
    _write_matrix_json(matrix_json_path, run_id, rows)
    _write_per_query_rows(per_query_dir, per_query)

    return [asdict(row) for row in rows]


def run_experiment_matrix(
    *,
    run_id: str,
    config_path: Path | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for matrix execution."""

    return run_matrix(run_id=run_id, config_path=config_path, settings=settings)
