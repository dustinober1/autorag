from __future__ import annotations

from autokg_rag.ingest.header_footer_filter import (
    is_header_footer_line,
    remove_header_footer_from_text,
    remove_repeated_lines_across_pages,
)


def test_is_header_footer_line_matches_common_pmbok_boilerplate() -> None:
    assert is_header_footer_line("PMBOK Guide")
    assert is_header_footer_line("Project Management Institute")
    assert is_header_footer_line("Page 7 of 742")
    assert not is_header_footer_line("Project charter authorizes the initiative.")


def test_remove_header_footer_from_text_strips_known_lines() -> None:
    text = (
        "PMBOK Guide\n"
        "Project Management Institute\n"
        "Project charter authorizes the project manager.\n"
        "Page 1 of 700\n"
    )
    cleaned = remove_header_footer_from_text(text)
    assert "PMBOK Guide" not in cleaned
    assert "Project Management Institute" not in cleaned
    assert "Page 1 of 700" not in cleaned
    assert "Project charter authorizes the project manager." in cleaned


def test_remove_repeated_lines_across_pages_filters_repetition() -> None:
    pages = [
        "PMBOK Guide\nUnique line A",
        "PMBOK Guide\nUnique line B",
        "PMBOK Guide\nUnique line C",
    ]
    cleaned = remove_repeated_lines_across_pages(pages, threshold=0.66)
    assert all("PMBOK Guide" not in page for page in cleaned)
    assert cleaned[0] == "Unique line A"
