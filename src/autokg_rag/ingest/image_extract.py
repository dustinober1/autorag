"""Image metadata extraction helpers for PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autokg_rag.exceptions import IngestError


def _load_pdfplumber() -> Any:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise IngestError(
            "pdfplumber is required for image extraction. Install with `uv sync --extra vision`."
        ) from exc
    return pdfplumber


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def extract_images_from_pdf(pdf_path: Path) -> list[dict[str, object]]:
    """Extract image metadata rows from each page of a PDF."""

    if not pdf_path.exists():
        raise IngestError(f"PDF path does not exist: {pdf_path}")

    pdfplumber = _load_pdfplumber()
    images: list[dict[str, object]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for idx, image in enumerate(page.images, start=1):
                width = _as_float(image.get("width"))
                height = _as_float(image.get("height"))
                images.append(
                    {
                        "page_num": page_num,
                        "image_idx": idx,
                        "name": str(image.get("name", f"img_{page_num}_{idx}")),
                        "x0": _as_float(image.get("x0")),
                        "y0": _as_float(image.get("top")),
                        "x1": _as_float(image.get("x1")),
                        "y1": _as_float(image.get("bottom")),
                        "width": width,
                        "height": height,
                        "area": width * height,
                    }
                )

    return images


def extract_diagram_regions(
    pdf_path: Path,
    *,
    min_size: int = 100,
    max_aspect_ratio: float = 8.0,
) -> list[dict[str, object]]:
    """Extract likely diagram regions by size and aspect-ratio heuristics."""

    if min_size < 1:
        raise IngestError("min_size must be >= 1.")

    candidates = extract_images_from_pdf(pdf_path)
    diagrams: list[dict[str, object]] = []
    for row in candidates:
        width = _as_float(row.get("width"))
        height = _as_float(row.get("height"))
        if width < min_size or height < min_size:
            continue

        smaller = max(1.0, min(width, height))
        ratio = max(width, height) / smaller
        if ratio > max_aspect_ratio:
            continue

        diagrams.append(row)

    return diagrams
