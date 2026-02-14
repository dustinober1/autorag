"""Evaluation dataset builders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from autokg_rag.io import read_jsonl_rows, read_parquet_rows, write_jsonl_rows
from autokg_rag.schemas.provenance import Citation
from autokg_rag.schemas.records import ChunkRecord, EvalQuestionRecord, QuestionType

DEFAULT_TYPE_MIX: Final[dict[QuestionType, float]] = {
    "fact": 0.45,
    "multi_hop": 0.35,
    "contrast": 0.20,
}
TYPE_ORDER: Final[tuple[QuestionType, ...]] = ("fact", "multi_hop", "contrast")
STARTER_DATASET_SIZE: Final[int] = 20

_FACT_TEMPLATES: Final[tuple[str, ...]] = (
    "What does {doc_id} page {page} describe in the '{section}' section?",
    "According to '{section}' in {doc_id}, what is the key point?",
    "Which detail is stated in {doc_id} p{page} under '{section}'?",
)
_MULTI_HOP_TEMPLATES: Final[tuple[str, ...]] = (
    "How does '{section_a}' in {doc_a} connect to '{section_b}' in {doc_b}?",
    "What relationship links {doc_a} p{page_a} '{section_a}' with {doc_b} p{page_b} '{section_b}'?",
    "Using both '{section_a}' and '{section_b}', what combined conclusion is supported?",
)
_CONTRAST_TEMPLATES: Final[tuple[str, ...]] = (
    "Contrast '{section_a}' from {doc_a} with '{section_b}' from {doc_b}.",
    "How does {doc_a} p{page_a} '{section_a}' differ from {doc_b} p{page_b} '{section_b}'?",
    "What is the main contrast between '{section_a}' and '{section_b}'?",
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SPACE_RE = re.compile(r"\s+")

_StarterCitation = tuple[str, int, str, str]
_StarterRow = tuple[QuestionType, str, str | None, tuple[_StarterCitation, ...]]

_STARTER_ROWS: Final[tuple[_StarterRow, ...]] = (
    (
        "fact",
        "What is the purpose of a project charter?",
        "It formally authorizes the project and grants authority to start work.",
        (
            (
                "fixture_pm_scope",
                2,
                "Project Charter",
                "fixture_pm_scope_p2_project_charter_c0001",
            ),
        ),
    ),
    (
        "fact",
        "What does a scope baseline include?",
        "It includes the scope statement, WBS, and WBS dictionary.",
        (
            (
                "fixture_pm_scope",
                5,
                "Scope Baseline",
                "fixture_pm_scope_p5_scope_baseline_c0001",
            ),
        ),
    ),
    (
        "fact",
        "What is recorded in a risk register entry?",
        "A risk entry captures cause, event, impact, owner, and response.",
        (("fixture_pm_risk", 3, "Risk Register", "fixture_pm_risk_p3_risk_register_c0001"),),
    ),
    (
        "fact",
        "What is the difference between mitigation and contingency?",
        "Mitigation is pre-event reduction; contingency is post-trigger action.",
        (
            (
                "fixture_pm_risk",
                6,
                "Response Planning",
                "fixture_pm_risk_p6_response_planning_c0002",
            ),
        ),
    ),
    (
        "fact",
        "What does self-attention compute in a transformer block?",
        "It computes weighted token interactions to form contextual representations.",
        (
            (
                "fixture_transformers",
                2,
                "Self-Attention",
                "fixture_transformers_p2_self_attention_c0001",
            ),
        ),
    ),
    (
        "fact",
        "Why are positional encodings needed in transformers?",
        "They inject sequence order information into attention-based models.",
        (
            (
                "fixture_transformers",
                3,
                "Positional Encoding",
                "fixture_transformers_p3_positional_encoding_c0001",
            ),
        ),
    ),
    (
        "fact",
        "What are the two main stages in retrieval-augmented generation?",
        "The system retrieves evidence and then generates conditioned on that evidence.",
        (("fixture_rag", 2, "RAG Pipeline", "fixture_rag_p2_rag_pipeline_c0001"),),
    ),
    (
        "fact",
        "Which provenance fields are required for each citation?",
        "Each citation must include doc_id, page, section, and chunk_id.",
        (
            (
                "fixture_rag",
                4,
                "Grounding and Citations",
                "fixture_rag_p4_grounding_c0001",
            ),
        ),
    ),
    (
        "fact",
        "What does recall@k measure in retrieval evaluation?",
        "It measures how many relevant evidence chunks are found in the top-k results.",
        (
            (
                "fixture_rag",
                5,
                "Evaluation Metrics",
                "fixture_rag_p5_metrics_c0001",
            ),
        ),
    ),
    (
        "multi_hop",
        "How does scope change control connect to risk updates?",
        "Approved scope changes trigger risk reassessment and risk register updates.",
        (
            (
                "fixture_pm_scope",
                8,
                "Integrated Change Control",
                "fixture_pm_scope_p8_change_control_c0001",
            ),
            (
                "fixture_pm_risk",
                4,
                "Risk Monitoring",
                "fixture_pm_risk_p4_risk_monitoring_c0001",
            ),
        ),
    ),
    (
        "multi_hop",
        "Why does clearer scope decomposition improve retrieval quality?",
        "Structured scope decomposition yields better section boundaries and chunk precision.",
        (
            (
                "fixture_pm_scope",
                6,
                "WBS Decomposition",
                "fixture_pm_scope_p6_wbs_decomposition_c0001",
            ),
            ("fixture_rag", 3, "Chunking Effects", "fixture_rag_p3_chunking_effects_c0001"),
        ),
    ),
    (
        "multi_hop",
        "How do self-attention and positional encoding work together?",
        "Self-attention models interactions while positional encoding preserves order.",
        (
            (
                "fixture_transformers",
                2,
                "Self-Attention",
                "fixture_transformers_p2_self_attention_c0001",
            ),
            (
                "fixture_transformers",
                3,
                "Positional Encoding",
                "fixture_transformers_p3_positional_encoding_c0001",
            ),
        ),
    ),
    (
        "multi_hop",
        "How can chunk overlap affect recall and citation precision?",
        "Overlap can improve recall but excessive overlap can reduce citation precision.",
        (
            ("fixture_rag", 3, "Chunking Effects", "fixture_rag_p3_chunking_effects_c0002"),
            ("fixture_rag", 5, "Evaluation Metrics", "fixture_rag_p5_metrics_c0002"),
        ),
    ),
    (
        "multi_hop",
        "How does ontology extraction support graph retrieval?",
        "Entities and relations become graph structure for multi-hop traversal to evidence.",
        (
            (
                "fixture_rag",
                6,
                "Ontology Extraction",
                "fixture_rag_p6_ontology_extraction_c0001",
            ),
            ("fixture_rag", 7, "Graph Retrieval", "fixture_rag_p7_graph_retrieval_c0001"),
        ),
    ),
    (
        "multi_hop",
        "How does hybrid retrieval combine vector and graph signals?",
        "Hybrid retrieval mixes semantic similarity with relationship-aware traversal evidence.",
        (
            ("fixture_rag", 7, "Graph Retrieval", "fixture_rag_p7_graph_retrieval_c0002"),
            ("fixture_rag", 8, "Hybrid Fusion", "fixture_rag_p8_hybrid_fusion_c0001"),
        ),
    ),
    (
        "multi_hop",
        "How is citation precision computed from answer sentences and chunks?",
        "Precision is supported citations divided by total citations emitted.",
        (
            ("fixture_rag", 5, "Evaluation Metrics", "fixture_rag_p5_metrics_c0002"),
            (
                "fixture_rag",
                4,
                "Grounding and Citations",
                "fixture_rag_p4_grounding_c0002",
            ),
        ),
    ),
    (
        "contrast",
        "Contrast risk mitigation and risk acceptance.",
        "Mitigation actively reduces exposure; acceptance monitors and responds only if needed.",
        (
            (
                "fixture_pm_risk",
                6,
                "Response Planning",
                "fixture_pm_risk_p6_response_planning_c0001",
            ),
        ),
    ),
    (
        "contrast",
        "Contrast vector-only retrieval with graph-only retrieval.",
        (
            "Vector retrieval relies on semantic similarity while graph retrieval "
            "uses entity relations."
        ),
        (
            ("fixture_rag", 7, "Graph Retrieval", "fixture_rag_p7_graph_retrieval_c0003"),
            ("fixture_rag", 2, "RAG Pipeline", "fixture_rag_p2_rag_pipeline_c0002"),
        ),
    ),
    (
        "contrast",
        "Contrast fixed-size chunking and semantic-breakpoint chunking.",
        "Fixed chunking uses size boundaries while semantic chunking follows meaning boundaries.",
        (
            ("fixture_rag", 3, "Chunking Effects", "fixture_rag_p3_chunking_effects_c0003"),
        ),
    ),
    (
        "contrast",
        "Contrast encoder-only transformers with encoder-decoder models for QA generation.",
        (
            "Encoder-only architectures focus on representation; encoder-decoder "
            "is optimized for generation."
        ),
        (
            (
                "fixture_transformers",
                5,
                "Model Families",
                "fixture_transformers_p5_model_families_c0001",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class _DraftQuestion:
    question_type: QuestionType
    question: str
    gold_citations: tuple[Citation, ...]
    gold_answer: str | None


def _normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).lower()


def _clean_text(value: str) -> str:
    cleaned = _SPACE_RE.sub(" ", value.strip())
    return cleaned or "Unknown Section"


def _citation_from_chunk(chunk: ChunkRecord) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page=chunk.page,
        section=chunk.section,
    )


def _citation_from_values(
    doc_id: str,
    page: int,
    section: str,
    chunk_id: str,
) -> Citation:
    return Citation(chunk_id=chunk_id, doc_id=doc_id, page=page, section=section)


def _excerpt(text: str, *, max_words: int = 22) -> str:
    words = _WORD_RE.findall(text)
    if not words:
        return ""
    trimmed = " ".join(words[:max_words]).strip()
    if len(words) > max_words:
        return f"{trimmed}."
    return f"{trimmed}."


def _ensure_unique_question(
    *,
    question: str,
    seen: set[str],
    seed: int,
) -> str:
    base = question.strip()
    normalized = _normalize_text(base)
    if normalized not in seen:
        seen.add(normalized)
        return base

    suffix = seed + 1
    while True:
        candidate = f"{base} ({suffix})"
        normalized_candidate = _normalize_text(candidate)
        if normalized_candidate not in seen:
            seen.add(normalized_candidate)
            return candidate
        suffix += 1


def starter_questions() -> list[EvalQuestionRecord]:
    """Return the fixed 20-row starter question set."""

    rows: list[EvalQuestionRecord] = []
    for idx, row in enumerate(_STARTER_ROWS, start=1):
        question_type, question, gold_answer, citations = row
        citation_models = [
            _citation_from_values(
                doc_id=doc_id,
                page=page,
                section=section,
                chunk_id=chunk_id,
            )
            for doc_id, page, section, chunk_id in citations
        ]
        rows.append(
            EvalQuestionRecord(
                question_id=f"q{idx:03d}",
                type=question_type,
                question=question,
                gold_citations=citation_models,
                gold_answer=gold_answer,
            )
        )
    return rows


def bootstrap_starter_dataset(out_path: Path) -> list[EvalQuestionRecord]:
    """Write the fixed 20-row starter dataset to `out_path`."""

    rows = starter_questions()
    write_jsonl_rows(
        out_path,
        [row.model_dump(mode="json", exclude_none=True) for row in rows],
    )
    return rows


def load_chunks_for_eval(input_artifact_dir: Path) -> list[ChunkRecord]:
    """Load and validate chunks from a prior artifact directory."""

    parquet_path = input_artifact_dir / "chunks.parquet"
    jsonl_path = input_artifact_dir / "chunks.jsonl"

    if parquet_path.exists():
        rows = read_parquet_rows(parquet_path)
    elif jsonl_path.exists():
        rows = read_jsonl_rows(jsonl_path)
    else:
        raise FileNotFoundError(
            f"Missing chunks artifact in '{input_artifact_dir}'. "
            "Expected chunks.parquet or chunks.jsonl."
        )

    chunks = [ChunkRecord.model_validate(row) for row in rows]
    chunks.sort(key=lambda chunk: (chunk.doc_id, chunk.page, chunk.section, chunk.chunk_id))
    if not chunks:
        raise ValueError(f"No chunks found in '{input_artifact_dir}'.")
    return chunks


def compute_type_targets(
    target_size: int,
    type_mix: dict[QuestionType, float] | None = None,
) -> dict[QuestionType, int]:
    """Return deterministic per-type counts that sum to `target_size`."""

    if target_size < 1:
        raise ValueError("target_size must be >= 1.")

    mix = dict(DEFAULT_TYPE_MIX if type_mix is None else type_mix)
    for question_type in TYPE_ORDER:
        if question_type not in mix:
            raise ValueError(f"type_mix must include '{question_type}'.")
        if mix[question_type] < 0.0:
            raise ValueError(f"type_mix for '{question_type}' must be >= 0.")

    total_weight = sum(mix[question_type] for question_type in TYPE_ORDER)
    if total_weight <= 0.0:
        raise ValueError("type_mix weights must sum to a positive value.")

    scaled: dict[QuestionType, float] = {
        question_type: (mix[question_type] / total_weight) * float(target_size)
        for question_type in TYPE_ORDER
    }
    counts: dict[QuestionType, int] = {
        question_type: int(scaled[question_type]) for question_type in TYPE_ORDER
    }

    remainder = target_size - sum(counts.values())
    if remainder > 0:
        fractional_order = sorted(
            TYPE_ORDER,
            key=lambda question_type: (
                scaled[question_type] - float(counts[question_type]),
                -TYPE_ORDER.index(question_type),
            ),
            reverse=True,
        )
        for question_type in fractional_order[:remainder]:
            counts[question_type] += 1

    return counts


def _chunk_pair(
    chunks: list[ChunkRecord],
    idx: int,
    *,
    stride: int,
) -> tuple[ChunkRecord, ChunkRecord]:
    chunk_count = len(chunks)
    first = chunks[idx % chunk_count]
    if chunk_count == 1:
        return first, first

    offset = ((idx // chunk_count) % (chunk_count - 1)) + 1
    second = chunks[(idx + (offset * stride)) % chunk_count]
    return first, second


def _fact_question(chunk: ChunkRecord, idx: int) -> str:
    template = _FACT_TEMPLATES[idx % len(_FACT_TEMPLATES)]
    return template.format(
        doc_id=chunk.doc_id,
        page=chunk.page,
        section=_clean_text(chunk.section),
    )


def _multi_hop_question(first: ChunkRecord, second: ChunkRecord, idx: int) -> str:
    template = _MULTI_HOP_TEMPLATES[idx % len(_MULTI_HOP_TEMPLATES)]
    return template.format(
        section_a=_clean_text(first.section),
        section_b=_clean_text(second.section),
        doc_a=first.doc_id,
        doc_b=second.doc_id,
        page_a=first.page,
        page_b=second.page,
    )


def _contrast_question(first: ChunkRecord, second: ChunkRecord, idx: int) -> str:
    template = _CONTRAST_TEMPLATES[idx % len(_CONTRAST_TEMPLATES)]
    return template.format(
        section_a=_clean_text(first.section),
        section_b=_clean_text(second.section),
        doc_a=first.doc_id,
        doc_b=second.doc_id,
        page_a=first.page,
        page_b=second.page,
    )


def _build_fact_drafts(
    chunks: list[ChunkRecord],
    count: int,
    seen_questions: set[str],
) -> list[_DraftQuestion]:
    drafts: list[_DraftQuestion] = []
    for idx in range(count):
        chunk = chunks[idx % len(chunks)]
        question = _fact_question(chunk, idx)
        unique_question = _ensure_unique_question(question=question, seen=seen_questions, seed=idx)
        drafts.append(
            _DraftQuestion(
                question_type="fact",
                question=unique_question,
                gold_citations=(_citation_from_chunk(chunk),),
                gold_answer=_excerpt(chunk.chunk_text),
            )
        )
    return drafts


def _build_multi_hop_drafts(
    chunks: list[ChunkRecord],
    count: int,
    seen_questions: set[str],
) -> list[_DraftQuestion]:
    drafts: list[_DraftQuestion] = []
    for idx in range(count):
        first, second = _chunk_pair(chunks, idx, stride=1)
        question = _multi_hop_question(first, second, idx)
        unique_question = _ensure_unique_question(question=question, seen=seen_questions, seed=idx)
        answer = (
            f"{_excerpt(first.chunk_text, max_words=12)} "
            f"{_excerpt(second.chunk_text, max_words=12)}"
        )
        drafts.append(
            _DraftQuestion(
                question_type="multi_hop",
                question=unique_question,
                gold_citations=(_citation_from_chunk(first), _citation_from_chunk(second)),
                gold_answer=answer.strip(),
            )
        )
    return drafts


def _build_contrast_drafts(
    chunks: list[ChunkRecord],
    count: int,
    seen_questions: set[str],
) -> list[_DraftQuestion]:
    drafts: list[_DraftQuestion] = []
    for idx in range(count):
        first, second = _chunk_pair(chunks, idx, stride=2)
        question = _contrast_question(first, second, idx)
        unique_question = _ensure_unique_question(question=question, seen=seen_questions, seed=idx)
        answer = (
            f"{_clean_text(first.section)} focuses on {_excerpt(first.chunk_text, max_words=10)} "
            "while "
            f"{_clean_text(second.section)} "
            f"emphasizes {_excerpt(second.chunk_text, max_words=10)}"
        )
        drafts.append(
            _DraftQuestion(
                question_type="contrast",
                question=unique_question,
                gold_citations=(_citation_from_chunk(first), _citation_from_chunk(second)),
                gold_answer=answer.strip(),
            )
        )
    return drafts


def build_eval_questions(
    *,
    run_id: str,
    chunks: list[ChunkRecord],
    target_size: int,
    type_mix: dict[QuestionType, float] | None = None,
) -> list[EvalQuestionRecord]:
    """Build a deterministic evaluation question set from chunks."""

    clean_run_id = run_id.strip()
    if not clean_run_id:
        raise ValueError("run_id must be non-empty.")
    if not chunks:
        raise ValueError("chunks must be non-empty.")

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.doc_id, chunk.page, chunk.section, chunk.chunk_id),
    )
    targets = compute_type_targets(target_size=target_size, type_mix=type_mix)
    seen_questions: set[str] = set()

    drafts: list[_DraftQuestion] = []
    drafts.extend(_build_fact_drafts(ordered_chunks, targets["fact"], seen_questions))
    drafts.extend(_build_multi_hop_drafts(ordered_chunks, targets["multi_hop"], seen_questions))
    drafts.extend(_build_contrast_drafts(ordered_chunks, targets["contrast"], seen_questions))

    if len(drafts) != target_size:
        raise ValueError("Internal error: generated question count does not match target size.")

    records: list[EvalQuestionRecord] = []
    for idx, draft in enumerate(drafts, start=1):
        records.append(
            EvalQuestionRecord(
                question_id=f"{clean_run_id}:q{idx:04d}",
                type=draft.question_type,
                question=draft.question,
                gold_citations=list(draft.gold_citations),
                gold_answer=draft.gold_answer,
            )
        )
    return records


def dataset_output_path(
    *,
    run_id: str,
    target_size: int,
    artifact_root: Path,
) -> Path:
    """Return the generated questions JSONL output path."""

    return artifact_root / run_id / f"questions_{target_size}.jsonl"


def generate_dataset_records(
    *,
    run_id: str,
    input_artifact_dir: Path,
    target_size: int,
    type_mix: dict[QuestionType, float] | None = None,
) -> list[EvalQuestionRecord]:
    """Generate deterministic eval question rows from chunk artifacts."""

    chunks = load_chunks_for_eval(input_artifact_dir)
    return build_eval_questions(
        run_id=run_id,
        chunks=chunks,
        target_size=target_size,
        type_mix=type_mix,
    )


def generate_dataset_from_chunks(
    *,
    run_id: str,
    input_artifact_dir: Path,
    target_size: int,
    output_artifact_root: Path | None = None,
    type_mix: dict[QuestionType, float] | None = None,
) -> Path:
    """Generate questions and write `questions_<target>.jsonl`."""

    records = generate_dataset_records(
        run_id=run_id,
        input_artifact_dir=input_artifact_dir,
        target_size=target_size,
        type_mix=type_mix,
    )
    artifact_root = (
        input_artifact_dir.parent
        if output_artifact_root is None
        else output_artifact_root
    )
    output_path = dataset_output_path(
        run_id=run_id,
        target_size=target_size,
        artifact_root=artifact_root,
    )
    write_jsonl_rows(
        output_path,
        [record.model_dump(mode="json", exclude_none=True) for record in records],
    )
    return output_path
