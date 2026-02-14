from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from autokg_rag.config import Settings


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


class _FakeStreamlit:
    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.markdown_calls: list[str] = []

    def set_page_config(self, **kwargs: object) -> None:
        _ = kwargs

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        _ = unsafe_allow_html
        self.markdown_calls.append(body)

    def title(self, body: str) -> None:
        _ = body

    def caption(self, body: str) -> None:
        _ = body

    def warning(self, body: str) -> None:
        _ = body

    def error(self, body: str) -> None:
        _ = body

    def info(self, body: str) -> None:
        _ = body

    def subheader(self, body: str) -> None:
        self.subheaders.append(body)

    def selectbox(self, label: str, options: list[str], index: int = 0) -> str:
        _ = label
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
        return options[index]

    def text_input(self, label: str, value: str = "") -> str:
        _ = label
        return value

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
        return value

    def button(self, label: str) -> bool:
        _ = label
        return False


def test_streamlit_renders_query_and_citation_panels(tmp_path: Path) -> None:
    render_app = _load_render_app()
    fake = _FakeStreamlit()
    render_app(
        fake,
        settings=Settings(artifact_root=tmp_path / "artifacts"),
        run_ids=["m6"],
    )

    assert "Query Panel" in fake.subheaders
    assert "Citation Viewer" in fake.subheaders
    assert fake.markdown_calls
