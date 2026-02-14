from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from autokg_rag.config import Settings
from autokg_rag.schemas.api import AnswerPayload


def _load_render_app() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_path = repo_root / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("m6_streamlit_app", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_app


class _FakeContext:
    def __enter__(self) -> _FakeContext:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakeSidebar:
    def __init__(self, parent: _FakeStreamlit) -> None:
        self._parent = parent

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        _ = unsafe_allow_html
        self._parent.sidebar_markdown_calls.append(body)

    def selectbox(
        self,
        label: str,
        options: list[str],
        index: int = 0,
        disabled: bool = False,
    ) -> str:
        _ = label
        _ = disabled
        selected_run = self._parent.sidebar_run_override
        if selected_run is not None and selected_run in options:
            return selected_run
        if not options:
            return ""
        return options[index]

    def radio(
        self,
        label: str,
        options: list[str],
        index: int = 0,
        horizontal: bool = False,
    ) -> str:
        _ = label
        _ = horizontal
        selected_mode = self._parent.sidebar_mode_override
        if selected_mode is not None:
            return selected_mode
        return options[index]

    def slider(
        self,
        label: str,
        min_value: int,
        max_value: int,
        value: int,
        step: int = 1,
    ) -> int:
        _ = label
        _ = min_value
        _ = max_value
        _ = step
        selected_top_k = self._parent.sidebar_top_k_override
        if selected_top_k is not None:
            return selected_top_k
        return value


class _FakeStreamlit:
    def __init__(
        self,
        *,
        button_sequence: list[bool] | None = None,
        text_value: str | None = None,
    ) -> None:
        self.session_state: dict[str, Any] = {}
        self.sidebar = _FakeSidebar(self)
        self.markdown_calls: list[str] = []
        self.sidebar_markdown_calls: list[str] = []
        self.tab_labels: list[str] = []
        self.dataframe_calls = 0
        self.expander_labels: list[str] = []
        self.spinner_messages: list[str] = []
        self._button_sequence = list(button_sequence or [])
        self._text_value = text_value
        self.sidebar_run_override: str | None = None
        self.sidebar_mode_override: str | None = None
        self.sidebar_top_k_override: int | None = None

    def set_page_config(self, **kwargs: object) -> None:
        _ = kwargs

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        _ = unsafe_allow_html
        self.markdown_calls.append(body)

    def columns(self, specs: list[int]) -> list[_FakeContext]:
        _ = specs
        return [_FakeContext(), _FakeContext()]

    def text_input(
        self,
        label: str,
        value: str = "",
        placeholder: str = "",
        label_visibility: str = "visible",
    ) -> str:
        _ = label
        _ = placeholder
        _ = label_visibility
        if self._text_value is not None:
            return self._text_value
        return value

    def button(self, label: str, use_container_width: bool = False) -> bool:
        _ = label
        _ = use_container_width
        if not self._button_sequence:
            return False
        return self._button_sequence.pop(0)

    def spinner(self, message: str) -> _FakeContext:
        self.spinner_messages.append(message)
        return _FakeContext()

    def container(self) -> _FakeContext:
        return _FakeContext()

    def tabs(self, labels: list[str]) -> list[_FakeContext]:
        self.tab_labels = list(labels)
        return [_FakeContext(), _FakeContext(), _FakeContext()]

    def dataframe(
        self,
        data: object,
        *,
        use_container_width: bool = False,
        width: str | None = None,
    ) -> None:
        _ = data
        _ = use_container_width
        _ = width
        self.dataframe_calls += 1

    def expander(self, label: str, expanded: bool = False) -> _FakeContext:
        _ = expanded
        self.expander_labels.append(label)
        return _FakeContext()


def _sample_payload() -> AnswerPayload:
    return AnswerPayload.model_validate(
        {
            "answer": {
                "question_id": "q1",
                "answer_text": (
                    "Mitigation reduces probability, acceptance tolerates residual risk."
                ),
                "citations": [
                    {
                        "chunk_id": "c1",
                        "doc_id": "docA",
                        "page": 3,
                        "section": "Risk Responses",
                    }
                ],
            },
            "hits": [
                {
                    "question_id": "q1",
                    "rank": 1,
                    "score": 0.9212,
                    "vector_score": 0.8823,
                    "graph_score": 0.7401,
                    "chunk_id": "c1",
                    "doc_id": "docA",
                    "page": 3,
                    "section": "Risk Responses",
                }
            ],
            "citation_trace": [
                {
                    "answer_sentence_id": "s1",
                    "citation": {
                        "chunk_id": "c1",
                        "doc_id": "docA",
                        "page": 3,
                        "section": "Risk Responses",
                    },
                    "support_score": 0.93,
                }
            ],
        }
    )


def _contains_fragment(markdown_calls: list[str], fragment: str) -> bool:
    return any(fragment in entry for entry in markdown_calls)


def test_streamlit_renders_no_runs_empty_state(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit()

    render_app(fake, settings=Settings(artifact_root=tmp_path / "artifacts"), run_ids=[])

    assert _contains_fragment(fake.markdown_calls, "No run artifacts available")
    assert not fake.tab_labels


def test_streamlit_renders_no_query_state_with_runs(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit()

    render_app(fake, settings=Settings(artifact_root=tmp_path / "artifacts"), run_ids=["m6"])

    assert _contains_fragment(fake.markdown_calls, "Run:")
    assert _contains_fragment(fake.markdown_calls, "Ask a question to begin")
    assert not fake.tab_labels


def test_streamlit_persists_latest_result_across_rerenders(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit(button_sequence=[True, False], text_value="How do risk responses differ?")
    payload = _sample_payload()
    call_count = 0

    def _query_handler(**kwargs: object) -> AnswerPayload:
        nonlocal call_count
        _ = kwargs
        call_count += 1
        return payload

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )
    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    assert call_count == 1
    assert fake.tab_labels == ["Citations", "Retrieval Hits", "Grounding Trace"]
    assert fake.dataframe_calls >= 4
    assert _contains_fragment(fake.markdown_calls, "Answered in")


def test_streamlit_shows_cached_result_context_when_controls_change(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit(button_sequence=[True, False], text_value="How do risk responses differ?")
    payload = _sample_payload()

    def _query_handler(**kwargs: object) -> AnswerPayload:
        _ = kwargs
        return payload

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    fake.sidebar_mode_override = "vector"
    fake.sidebar_top_k_override = 5

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    assert _contains_fragment(fake.markdown_calls, "Showing latest result from Run: m6")
    assert _contains_fragment(fake.markdown_calls, "Submit again to refresh with current controls.")


def test_streamlit_shows_cached_result_context_when_question_changes(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit(button_sequence=[True, False], text_value="Question one")
    payload = _sample_payload()

    def _query_handler(**kwargs: object) -> AnswerPayload:
        _ = kwargs
        return payload

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    fake._text_value = "Question two"

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    assert _contains_fragment(fake.markdown_calls, "Showing latest result from Run: m6")
    assert _contains_fragment(fake.markdown_calls, "Submit again to refresh with current controls.")


def test_streamlit_renders_styled_error_card(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit(button_sequence=[True], text_value="What failed?")

    def _query_handler(**kwargs: object) -> AnswerPayload:
        _ = kwargs
        raise RuntimeError("simulated backend failure")

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    assert _contains_fragment(fake.markdown_calls, "Query failed")
    assert _contains_fragment(fake.markdown_calls, "simulated backend failure")


def test_streamlit_error_state_preserves_and_labels_cached_result(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit(button_sequence=[True, True], text_value="How do risk responses differ?")
    payload = _sample_payload()
    call_count = 0

    def _query_handler(**kwargs: object) -> AnswerPayload:
        nonlocal call_count
        _ = kwargs
        call_count += 1
        if call_count == 1:
            return payload
        raise RuntimeError("backend timeout")

    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )
    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
        query_handler=_query_handler,
    )

    assert _contains_fragment(fake.markdown_calls, "Query failed")
    assert _contains_fragment(fake.markdown_calls, "backend timeout")
    assert _contains_fragment(fake.markdown_calls, "Showing the latest successful result below.")
