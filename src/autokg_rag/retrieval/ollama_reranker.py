"""Optional Ollama-backed reranker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autokg_rag.ollama import OllamaClient
from autokg_rag.retrieval.rerank import (
    RerankResult,
    reorder_hits_by_chunk_id,
    with_sequential_ranks,
)


class _RerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_chunk_ids: list[str] = Field(min_length=1)


class OllamaReranker:
    """Call Ollama `generate` and reorder retrieval hits by returned chunk IDs."""

    def __init__(self, *, model: str = "llama3:8b", client: OllamaClient) -> None:
        self._model = model
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def rerank(
        self,
        *,
        question: str,
        hits: list[Any],
        chunk_text_by_id: Mapping[str, str] | None = None,
    ) -> RerankResult:
        """Return reranked hits or deterministic original-order fallback."""

        original = with_sequential_ranks(list(hits))
        if not original:
            return RerankResult(hits=[], parse_status="no_candidates")

        prompt = self._build_prompt(
            question=question,
            hits=original,
            chunk_text_by_id=chunk_text_by_id or {},
        )
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()

        try:
            raw_output = self._generate(prompt=prompt)
        except Exception as exc:  # pragma: no cover - exercised through integration tests
            return RerankResult(
                hits=original,
                parse_status="client_error",
                prompt_hash=prompt_hash,
                error=str(exc),
            )

        parsed_ids: list[str]
        try:
            payload = json.loads(raw_output)
            parsed = _RerankResponse.model_validate(payload)
            parsed_ids = parsed.ranked_chunk_ids
        except json.JSONDecodeError:
            return RerankResult(
                hits=original,
                parse_status="invalid_json",
                prompt_hash=prompt_hash,
                raw_output=raw_output,
            )
        except ValidationError:
            return RerankResult(
                hits=original,
                parse_status="invalid_schema",
                prompt_hash=prompt_hash,
                raw_output=raw_output,
            )

        valid_ids = {hit.chunk_id for hit in original}
        ranked_chunk_ids: list[str] = []
        seen: set[str] = set()
        ignored_count = 0
        for chunk_id in parsed_ids:
            if chunk_id not in valid_ids:
                ignored_count += 1
                continue
            if chunk_id in seen:
                continue
            ranked_chunk_ids.append(chunk_id)
            seen.add(chunk_id)

        # Keep deterministic completion ordering for missing IDs.
        for hit in original:
            if hit.chunk_id in seen:
                continue
            ranked_chunk_ids.append(hit.chunk_id)
            seen.add(hit.chunk_id)

        reranked = reorder_hits_by_chunk_id(hits=original, ranked_chunk_ids=ranked_chunk_ids)
        return RerankResult(
            hits=reranked,
            parse_status="ok" if ignored_count == 0 else "partial_ids",
            prompt_hash=prompt_hash,
            raw_output=raw_output,
        )

    def _build_prompt(
        self,
        *,
        question: str,
        hits: list[Any],
        chunk_text_by_id: Mapping[str, str],
    ) -> str:
        lines = [
            "You are a retrieval reranker.",
            "Return ONLY valid JSON as one object with exactly one key: ranked_chunk_ids.",
            "The value must include every candidate chunk_id exactly once.",
            "Do not add any prose or markdown.",
            f"Question: {question}",
            "Candidates:",
        ]

        for hit in hits:
            snippet = chunk_text_by_id.get(hit.chunk_id, "").strip().replace("\n", " ")
            if len(snippet) > 320:
                snippet = f"{snippet[:320]}..."
            lines.append(
                json.dumps(
                    {
                        "chunk_id": hit.chunk_id,
                        "doc_id": hit.doc_id,
                        "page": hit.page,
                        "section": hit.section,
                        "score": hit.score,
                        "text": snippet,
                    },
                    ensure_ascii=True,
                )
            )

        lines.append(
            json.dumps(
                {"ranked_chunk_ids": [hit.chunk_id for hit in hits]},
                ensure_ascii=True,
            )
        )
        return "\n".join(lines)

    def _generate(self, *, prompt: str) -> str:
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            format="json",
        )

        response_text = response.get("response")
        if isinstance(response_text, str):
            return response_text
        raise RuntimeError("Ollama generate response did not contain a text payload.")


__all__ = ["OllamaReranker"]
