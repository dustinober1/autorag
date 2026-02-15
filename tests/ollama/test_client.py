from __future__ import annotations

import io
from urllib import error as url_error
from urllib import request

import pytest

from autokg_rag.exceptions import RetrievalError
from autokg_rag.ollama.client import OllamaClient


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_post_json_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = req.data.decode("utf-8")
        return _FakeResponse(b'{"ok": true, "embeddings": [[0.1, 0.2]]}')

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=5)
    payload = client.post_json(path="/api/embed", payload={"model": "test", "input": ["hello"]})

    assert payload["ok"] is True
    assert captured["url"] == "http://localhost:11434/api/embed"
    assert captured["timeout"] == 5.0
    assert captured["payload"] == '{"model": "test", "input": ["hello"]}'


def test_post_json_http_error_raises_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        raise url_error.HTTPError(
            url=req.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"backend unavailable"),
        )

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=2)

    with pytest.raises(RetrievalError, match="HTTP 503"):
        client.post_json(path="/api/embed", payload={"model": "test", "input": ["hello"]})


def test_post_json_timeout_raises_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        raise url_error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1.5)

    with pytest.raises(RetrievalError, match="timed out"):
        client.post_json(path="/api/embed", payload={"model": "test", "input": ["hello"]})


def test_post_json_invalid_json_raises_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"not-json")

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=2)

    with pytest.raises(RetrievalError, match="valid JSON"):
        client.post_json(path="/api/embed", payload={"model": "test", "input": ["hello"]})


def test_post_json_non_object_json_raises_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"[1, 2, 3]")

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=2)

    with pytest.raises(RetrievalError, match="JSON object"):
        client.post_json(path="/api/embed", payload={"model": "test", "input": ["hello"]})


def test_client_sends_authorization_header_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OllamaClient should send Authorization header with Bearer token when api_key is set."""
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        captured["headers"] = dict(req.headers)
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout_seconds=5,
        api_key="test-secret-key",
    )
    client.post_json(path="/api/test", payload={})

    assert "Authorization" in captured["headers"], "Authorization header should be present"
    assert captured["headers"]["Authorization"] == "Bearer test-secret-key", \
        "Authorization header should use Bearer token format"


def test_client_omits_authorization_header_when_api_key_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OllamaClient should omit Authorization header when api_key is empty."""
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout: float) -> _FakeResponse:
        captured["headers"] = dict(req.headers)
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr("autokg_rag.ollama.client.request.urlopen", fake_urlopen)

    # Test with empty string (default)
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout_seconds=5,
        api_key="",
    )
    client.post_json(path="/api/test", payload={})

    assert "Authorization" not in captured["headers"], \
        "Authorization header should not be present when api_key is empty"
