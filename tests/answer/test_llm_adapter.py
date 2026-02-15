from __future__ import annotations

from pytest import MonkeyPatch

from autokg_rag.answer.llm_adapter import OllamaSentenceAdapter
from autokg_rag.exceptions import RetrievalError


def test_ollama_sentence_adapter_falls_back_on_retrieval_error(
    monkeypatch: MonkeyPatch,
) -> None:
    adapter = OllamaSentenceAdapter()

    def _raise_error(**_: object) -> object:
        raise RetrievalError("ollama unavailable")

    monkeypatch.setattr("autokg_rag.answer.ollama_adapter.get_ollama_adapter", _raise_error)

    sentences = adapter.generate_sentences(
        question="What is scope control?",
        evidence_texts=["Scope control aligns approved changes."],
        max_sentences=2,
    )

    assert sentences
    assert "scope control" in sentences[0].lower()


def test_ollama_sentence_adapter_falls_back_when_llm_returns_empty(
    monkeypatch: MonkeyPatch,
) -> None:
    adapter = OllamaSentenceAdapter()

    class _FakeAdapter:
        def generate(self, *_: object, **__: object) -> str:
            return "   "

    monkeypatch.setattr(
        "autokg_rag.answer.ollama_adapter.get_ollama_adapter",
        lambda **_: _FakeAdapter(),
    )

    sentences = adapter.generate_sentences(
        question="What is risk response?",
        evidence_texts=["Risk response selects mitigation and acceptance tactics."],
        max_sentences=1,
    )

    assert sentences
    assert "risk response" in sentences[0].lower()
