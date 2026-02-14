"""Ollama-backed answer generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autokg_rag.ollama import OllamaClient

OllamaModel = Literal["llama3", "llama3:8b", "mistral", "llava-llama3", "mixtral"]


@dataclass
class OllamaAdapter:
    """Adapter for local Ollama inference."""

    model: str = "llama3"
    temperature: float = 0.2
    max_tokens: int = 512
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 60.0

    def _client(self) -> OllamaClient:
        return OllamaClient(base_url=self.base_url, timeout_seconds=self.timeout_seconds)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        images: list[str] | None = None,
    ) -> str:
        """Generate text from prompt (optionally multimodal)."""

        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        user_message: dict[str, object] = {
            "role": "user",
            "content": prompt,
        }
        if images:
            user_message["images"] = images
        messages.append(user_message)

        response = self._client().chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        response_text = response.get("response")
        if isinstance(response_text, str) and response_text.strip():
            return response_text.strip()
        return ""

    def generate_with_context(
        self,
        question: str,
        context_chunks: list[str],
        *,
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        """Generate an answer from retrieved chunks and return lightweight citations."""

        context = "\n\n".join(
            f"[{idx}] {chunk}" for idx, chunk in enumerate(context_chunks, start=1)
        )
        prompt = (
            "Use only the provided context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Respond with a detailed grounded answer and keep "
            "citation markers like [1], [2] inline."
        )
        if system_prompt is None:
            system_prompt = "You are a grounded assistant that answers from provided context only."

        answer = self.generate(prompt, system_prompt=system_prompt)
        citations = [
            {
                "index": idx,
                "context": chunk,
            }
            for idx, chunk in enumerate(context_chunks, start=1)
        ]
        return answer, citations


_default_adapter: OllamaAdapter | None = None


def get_ollama_adapter(
    *,
    model: OllamaModel | str = "llama3",
    temperature: float = 0.2,
    max_tokens: int = 512,
    base_url: str = "http://localhost:11434",
    timeout_seconds: float = 60.0,
) -> OllamaAdapter:
    """Get a reusable Ollama adapter instance."""

    global _default_adapter
    if _default_adapter is None:
        _default_adapter = OllamaAdapter(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        return _default_adapter

    # Keep the singleton in sync with explicit overrides.
    _default_adapter.model = str(model)
    _default_adapter.temperature = float(temperature)
    _default_adapter.max_tokens = int(max_tokens)
    _default_adapter.base_url = base_url
    _default_adapter.timeout_seconds = float(timeout_seconds)
    return _default_adapter
