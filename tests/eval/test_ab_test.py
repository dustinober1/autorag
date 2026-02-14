from __future__ import annotations

from autokg_rag.eval.ab_test import format_ab_test_report, run_ab_test


def _fake_retrieval(question: dict[str, object], mode: str, k: int) -> dict[str, float]:
    del question
    del k
    if mode == "vector":
        return {"recall_at_k": 0.4, "ndcg_at_k": 0.45}
    if mode == "hybrid":
        return {"recall_at_k": 0.7, "ndcg_at_k": 0.75}
    return {"recall_at_k": 0.5, "ndcg_at_k": 0.55}


def test_run_ab_test_computes_summary_and_winner() -> None:
    questions = [{"question_id": "q1"}, {"question_id": "q2"}]
    result = run_ab_test(
        questions=questions,
        run_retrieval_fn=_fake_retrieval,
        mode_a="vector",
        mode_b="hybrid",
        metric="recall_at_k",
        k=5,
        n_runs=2,
    )

    assert result.mean_a == 0.4
    assert result.mean_b == 0.7
    assert result.winner == "hybrid"
    assert len(result.scores_a) == 4
    assert len(result.scores_b) == 4


def test_format_ab_test_report_contains_key_sections() -> None:
    questions = [{"question_id": "q1"}]
    result = run_ab_test(
        questions=questions,
        run_retrieval_fn=_fake_retrieval,
        mode_a="vector",
        mode_b="hybrid",
        metric="ndcg_at_k",
        k=5,
        n_runs=1,
    )
    report = format_ab_test_report(result)

    assert "# A/B Test Results: vector vs hybrid" in report
    assert "## Statistical Significance" in report
    assert "## Winner" in report
