from __future__ import annotations

from pytest import MonkeyPatch

from autokg_rag.answer.ollama_adapter import OllamaAdapter


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        stream: bool = False,
        options: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": options,
            }
        )
        return {"message": {"content": self.content}}


def test_ollama_adapter_generate_uses_chat_payload(monkeypatch: MonkeyPatch) -> None:
    adapter = OllamaAdapter(model="llama3", temperature=0.3, max_tokens=128)
    fake = _FakeClient(content="Grounded answer.")
    monkeypatch.setattr(adapter, "_client", lambda: fake)

    answer = adapter.generate("What is a charter?", system_prompt="You are grounded.")

    assert answer == "Grounded answer."
    assert fake.calls
    assert fake.calls[0]["model"] == "llama3"
    assert fake.calls[0]["stream"] is False


def test_ollama_adapter_generate_with_context_returns_citation_rows(
    monkeypatch: MonkeyPatch,
) -> None:
    adapter = OllamaAdapter(model="llama3")
    fake = _FakeClient(content="A charter authorizes the project manager [1].")
    monkeypatch.setattr(adapter, "_client", lambda: fake)

    answer, citations = adapter.generate_with_context(
        "What is the purpose of a charter?",
        ["The charter authorizes the project and PM authority."],
    )

    assert "charter" in answer.lower()
    assert citations and citations[0]["index"] == 1
