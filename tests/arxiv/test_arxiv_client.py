from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from autokg_rag.arxiv import client
from autokg_rag.schemas.api import ArxivPaper


def test_search_arxiv_returns_structured_papers(monkeypatch: MonkeyPatch) -> None:
    class _FakeSearch:
        def __init__(self, *, query: str, max_results: int, sort_by: object) -> None:
            _ = sort_by
            self.query = query
            self.max_results = max_results

    class _FakeClient:
        def __init__(self, *, page_size: int, delay_seconds: int, num_retries: int) -> None:
            _ = page_size
            _ = delay_seconds
            _ = num_retries

        def results(self, search: _FakeSearch) -> list[object]:
            assert search.query == "graph rag"
            assert search.max_results == 5
            return [
                SimpleNamespace(
                    entry_id="https://arxiv.org/abs/2401.01234v2",
                    title="Graph RAG Systems",
                    authors=[SimpleNamespace(name="A. Author"), SimpleNamespace(name="B. Author")],
                    summary="A paper about graph-augmented retrieval.",
                    pdf_url="https://arxiv.org/pdf/2401.01234v2.pdf",
                    published=datetime(2024, 1, 5, tzinfo=UTC),
                )
            ]

    fake_sdk = SimpleNamespace(
        Search=_FakeSearch,
        Client=_FakeClient,
        SortCriterion=SimpleNamespace(Relevance="relevance"),
    )
    monkeypatch.setattr(client, "_load_arxiv_sdk", lambda: fake_sdk)

    papers = client.search_arxiv("graph rag", max_results=5)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.01234v2"
    assert papers[0].title == "Graph RAG Systems"
    assert papers[0].authors == ["A. Author", "B. Author"]


def test_download_papers_writes_pdf_files(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    paper = ArxivPaper(
        arxiv_id="2401.01234v2",
        title="Graph RAG Systems",
        authors=["A. Author"],
        abstract="",
        pdf_url="https://arxiv.org/pdf/2401.01234v2.pdf",
        published=datetime(2024, 1, 5, tzinfo=UTC),
    )

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            _ = exc_type
            _ = exc
            _ = tb
            return False

        def read(self) -> bytes:
            return b"%PDF-1.7\nfake\n%%EOF"

    monkeypatch.setattr(client.request, "urlopen", lambda url, timeout=60.0: _FakeResponse())

    paths = client.download_papers([paper], tmp_path)
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".pdf"

