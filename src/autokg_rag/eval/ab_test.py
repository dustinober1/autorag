"""A/B testing helpers for retrieval strategy comparison."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

RetrievalMode = Literal["vector", "graph", "hybrid"]


@dataclass(frozen=True)
class ABTestResult:
    """Results from one retrieval A/B test comparison."""

    mode_a: RetrievalMode
    mode_b: RetrievalMode
    metric: str
    scores_a: list[float]
    scores_b: list[float]
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float | None
    winner: RetrievalMode | None


def _maybe_ttest(scores_a: list[float], scores_b: list[float]) -> float | None:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return None

    try:
        from scipy import stats  # type: ignore
    except ImportError:
        return None

    _, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    if p_value is None:
        return None
    try:
        return float(p_value)
    except (TypeError, ValueError):
        return None


def run_ab_test(
    *,
    questions: list[dict[str, object]],
    run_retrieval_fn: Callable[[dict[str, object], RetrievalMode, int], dict[str, float]],
    mode_a: RetrievalMode = "vector",
    mode_b: RetrievalMode = "hybrid",
    metric: str = "recall_at_k",
    k: int = 5,
    n_runs: int = 1,
) -> ABTestResult:
    """Run retrieval A/B test and compute summary statistics."""

    scores_a: list[float] = []
    scores_b: list[float] = []

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")

    retrieval_fn = run_retrieval_fn

    for _ in range(n_runs):
        shuffled = list(questions)
        random.shuffle(shuffled)
        for question in shuffled:
            result_a = retrieval_fn(question, mode_a, k)
            result_b = retrieval_fn(question, mode_b, k)
            scores_a.append(float(result_a.get(metric, 0.0)))
            scores_b.append(float(result_b.get(metric, 0.0)))

    mean_a = float(np.mean(scores_a)) if scores_a else 0.0
    mean_b = float(np.mean(scores_b)) if scores_b else 0.0
    std_a = float(np.std(scores_a)) if scores_a else 0.0
    std_b = float(np.std(scores_b)) if scores_b else 0.0
    p_value = _maybe_ttest(scores_a, scores_b)

    winner: RetrievalMode | None = None
    if mean_a > mean_b:
        winner = mode_a
    elif mean_b > mean_a:
        winner = mode_b

    return ABTestResult(
        mode_a=mode_a,
        mode_b=mode_b,
        metric=metric,
        scores_a=scores_a,
        scores_b=scores_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        p_value=p_value,
        winner=winner,
    )


def format_ab_test_report(result: ABTestResult) -> str:
    """Render A/B test results as markdown."""

    significance = bool(result.p_value is not None and result.p_value < 0.05)
    p_value_text = "N/A" if result.p_value is None else f"{result.p_value:.4f}"

    lines = [
        f"# A/B Test Results: {result.mode_a} vs {result.mode_b}",
        "",
        "## Summary",
        "",
        f"| Metric | {result.mode_a} | {result.mode_b} |",
        "|---|---:|---:|",
        f"| Mean {result.metric} | {result.mean_a:.4f} | {result.mean_b:.4f} |",
        f"| Std Dev | {result.std_a:.4f} | {result.std_b:.4f} |",
        "",
        "## Statistical Significance",
        "",
        f"- p-value: {p_value_text}",
        f"- Statistically significant (p<0.05): {'Yes' if significance else 'No'}",
        "",
        "## Winner",
        "",
        f"**{result.winner.upper() if result.winner else 'TIE'}**",
    ]

    if result.winner:
        diff = abs(result.mean_a - result.mean_b)
        denom = max(result.mean_a, result.mean_b, 1e-8)
        pct = (diff / denom) * 100.0
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    f"The {result.winner} strategy shows a {pct:.1f}% improvement in "
                    f"{result.metric}."
                ),
            ]
        )

    return "\n".join(lines)
