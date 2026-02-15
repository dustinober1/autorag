"""LLM-as-a-judge evaluation helpers."""

from __future__ import annotations

import re
from typing import Literal

from autokg_rag.ollama import OllamaClient

JudgementCriteria = Literal["correctness", "helpfulness", "groundedness", "coherence"]
_SCORE_RE = re.compile(r"(?im)^score\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _build_judge_prompt(
    *,
    question: str,
    answer: str,
    context: list[str],
    criteria: JudgementCriteria,
) -> str:
    context_text = "\n\n".join(context)
    return (
        f"You are an expert evaluator. Judge the answer for {criteria}.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        f"Generated Answer:\n{answer}\n\n"
        "Scoring rubric:\n"
        "- 0 means very poor quality\n"
        "- 10 means excellent quality\n\n"
        "Return plain text with exactly two lines:\n"
        "Score: <0-10>\n"
        "Reasoning: <short explanation>"
    )


def _parse_score_reasoning(raw_text: str) -> tuple[float, str]:
    score = 5.0
    match = _SCORE_RE.search(raw_text)
    if match:
        score = float(match.group(1))
    score = max(0.0, min(10.0, score))

    reasoning = raw_text.strip()
    for line in raw_text.splitlines():
        if line.lower().startswith("reasoning:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                reasoning = candidate
            break
    return score, reasoning


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def evaluate_with_llm_judge(
    *,
    question: str,
    answer: str,
    context: list[str],
    criteria: JudgementCriteria = "correctness",
    model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434",
    timeout_seconds: float = 60.0,
    ollama_api_key: str = "",
    client: OllamaClient | None = None,
) -> dict[str, object]:
    """Evaluate one answer with an Ollama judge model."""

    resolved_client = client or OllamaClient(
        base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        api_key=ollama_api_key,
    )
    prompt = _build_judge_prompt(
        question=question,
        answer=answer,
        context=context,
        criteria=criteria,
    )
    response = resolved_client.generate(
        model=model,
        prompt=prompt,
        stream=False,
    )
    raw_output = str(response.get("response", "")).strip()
    score, reasoning = _parse_score_reasoning(raw_output)
    return {
        "criteria": criteria,
        "score": score,
        "reasoning": reasoning,
        "model": model,
        "raw_output": raw_output,
    }


def evaluate_answer_set(
    *,
    questions: list[dict[str, object]],
    answers: list[str],
    contexts: list[list[str]],
    criteria: list[JudgementCriteria] | None = None,
    model: str = "llama3",
    ollama_base_url: str = "http://localhost:11434",
    timeout_seconds: float = 60.0,
    ollama_api_key: str = "",
    client: OllamaClient | None = None,
) -> list[dict[str, object]]:
    """Evaluate an answer set across one or more judge criteria."""

    resolved_criteria = criteria or ["correctness", "helpfulness", "groundedness"]
    resolved_client = client or OllamaClient(
        base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        api_key=ollama_api_key,
    )

    rows: list[dict[str, object]] = []
    for question_row, answer, context in zip(questions, answers, contexts, strict=True):
        question_id = str(question_row.get("question_id", "unknown"))
        question_text = str(question_row.get("question", "")).strip()
        judgements: list[dict[str, object]] = []

        for criterion in resolved_criteria:
            judgement = evaluate_with_llm_judge(
                question=question_text,
                answer=answer,
                context=context,
                criteria=criterion,
                model=model,
                ollama_base_url=ollama_base_url,
                timeout_seconds=timeout_seconds,
                client=resolved_client,
            )
            judgements.append(judgement)

        total = 0.0
        for judgement in judgements:
            total += _coerce_float(judgement.get("score", 0.0))
        average = total / float(len(judgements)) if judgements else 0.0
        rows.append(
            {
                "question_id": question_id,
                "question": question_text,
                "answer": answer,
                "avg_score": average,
                "judgements": judgements,
            }
        )

    return rows
