"""Minimal Ollama HTTP client wrappers."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request
from urllib.parse import urljoin

from autokg_rag.exceptions import RetrievalError


class OllamaClient:
    """HTTP client for Ollama JSON APIs using stdlib urllib."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        api_key: str = "",
    ) -> None:
        normalized_base = base_url.strip()
        if not normalized_base:
            raise RetrievalError("Ollama base URL must not be empty.")

        if not normalized_base.endswith("/"):
            normalized_base = f"{normalized_base}/"

        timeout = float(timeout_seconds)
        if timeout <= 0:
            raise RetrievalError("Ollama timeout_seconds must be greater than zero.")

        self.base_url = normalized_base
        self.timeout_seconds = timeout
        self.api_key = api_key

    def _url_for(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._url_for(path)
        body = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            url=url,
            data=body,
            headers=headers,
            method=method.upper(),
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            raw_body = b""
            try:
                raw_body = exc.read()
            except OSError:
                raw_body = b""
            detail = raw_body.decode("utf-8", errors="replace").strip()
            suffix = f": {detail[:240]}" if detail else ""
            raise RetrievalError(
                f"Ollama request failed with HTTP {exc.code} for {url}{suffix}"
            ) from exc
        except TimeoutError as exc:
            raise RetrievalError(
                f"Ollama request timed out after {self.timeout_seconds}s for {url}."
            ) from exc
        except error.URLError as exc:
            reason = str(exc.reason)
            lowered = reason.lower()
            if "timed out" in lowered:
                raise RetrievalError(
                    f"Ollama request timed out after {self.timeout_seconds}s for {url}."
                ) from exc
            raise RetrievalError(f"Ollama request failed for {url}: {reason}") from exc

        payload_text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(payload_text) if payload_text.strip() else {}
        except json.JSONDecodeError as exc:
            snippet = payload_text[:240].replace("\n", "\\n")
            raise RetrievalError(
                f"Ollama response for {url} was not valid JSON: {snippet}"
            ) from exc

        if not isinstance(parsed, dict):
            raise RetrievalError(
                f"Ollama response for {url} must be a JSON object, got {type(parsed).__name__}."
            )

        return parsed

    def get_json(self, *, path: str) -> dict[str, Any]:
        """GET and decode a JSON object response."""

        return self._request_json(method="GET", path=path)

    def post_json(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON payload and return decoded JSON object."""

        return self._request_json(method="POST", path=path, payload=payload)

    def list_tags(self) -> dict[str, Any]:
        """Return Ollama model tags response from `/api/tags`."""

        return self.get_json(path="/api/tags")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool = False,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Call `/api/generate` and return JSON payload."""

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if format is not None:
            payload["format"] = format
        return self.post_json(path="/api/generate", payload=payload)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call `/api/chat` and return JSON payload."""

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        return self.post_json(path="/api/chat", payload=payload)
