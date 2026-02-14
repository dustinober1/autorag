from __future__ import annotations

from autokg_rag.eval.judge import evaluate_answer_set, evaluate_with_llm_judge


class _FakeJudgeClient:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool = False,
        format: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "format": format,
            }
        )
        return {"response": self.output}


def test_evaluate_with_llm_judge_parses_score_and_reasoning() -> None:
    client = _FakeJudgeClient(output="Score: 8.5\nReasoning: Correct and grounded.")
    result = evaluate_with_llm_judge(
        question="What is a charter?",
        answer="It authorizes the project.",
        context=["A project charter authorizes a project."],
        criteria="correctness",
        model="llama3",
        client=client,  # type: ignore[arg-type]
    )

    assert result["score"] == 8.5
    assert result["reasoning"] == "Correct and grounded."
    assert result["criteria"] == "correctness"
    assert client.calls and client.calls[0]["model"] == "llama3"


def test_evaluate_answer_set_aggregates_criteria_scores() -> None:
    client = _FakeJudgeClient(output="Score: 6\nReasoning: Adequate.")
    rows = evaluate_answer_set(
        questions=[
            {
                "question_id": "q1",
                "question": "What is WBS?",
            }
        ],
        answers=["A hierarchical decomposition of work."],
        contexts=[["WBS is a decomposition of project scope."]],
        criteria=["correctness", "helpfulness"],
        client=client,  # type: ignore[arg-type]
    )

    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["avg_score"] == 6.0
    assert len(rows[0]["judgements"]) == 2
