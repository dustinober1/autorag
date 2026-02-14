"""Evaluation dataset and metric helpers."""

from autokg_rag.eval.ab_test import ABTestResult, format_ab_test_report, run_ab_test
from autokg_rag.eval.dataset_builder import (
    DEFAULT_TYPE_MIX,
    STARTER_DATASET_SIZE,
    bootstrap_starter_dataset,
    build_eval_questions,
    compute_type_targets,
    dataset_output_path,
    generate_dataset_from_chunks,
    generate_dataset_records,
    load_chunks_for_eval,
    starter_questions,
)
from autokg_rag.eval.judge import evaluate_answer_set, evaluate_with_llm_judge
from autokg_rag.eval.matrix_runner import (
    build_factorial_grid,
    load_matrix_config,
    run_experiment_matrix,
    run_matrix,
)
from autokg_rag.eval.metrics import (
    SupportsChunkId,
    aggregate_metric_rows,
    citation_precision,
    evaluate_and_aggregate,
    evaluate_query_metrics,
    faithfulness_proxy,
    ndcg_at_k,
    recall_at_k,
)
from autokg_rag.eval.report import build_experiment_report

__all__ = [
    "DEFAULT_TYPE_MIX",
    "STARTER_DATASET_SIZE",
    "ABTestResult",
    "SupportsChunkId",
    "aggregate_metric_rows",
    "bootstrap_starter_dataset",
    "build_eval_questions",
    "evaluate_answer_set",
    "evaluate_with_llm_judge",
    "format_ab_test_report",
    "citation_precision",
    "compute_type_targets",
    "dataset_output_path",
    "evaluate_and_aggregate",
    "evaluate_query_metrics",
    "faithfulness_proxy",
    "generate_dataset_from_chunks",
    "generate_dataset_records",
    "load_chunks_for_eval",
    "ndcg_at_k",
    "recall_at_k",
    "starter_questions",
    "build_factorial_grid",
    "load_matrix_config",
    "run_experiment_matrix",
    "run_ab_test",
    "run_matrix",
    "build_experiment_report",
]
