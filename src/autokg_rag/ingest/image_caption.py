"""Image captioning helpers backed by Ollama vision models."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from autokg_rag.exceptions import RetrievalError
from autokg_rag.ingest.image_extract import extract_diagram_regions
from autokg_rag.ollama import OllamaClient

DEFAULT_CAPTION_PROMPT = (
    "Describe this project-management diagram. Include key entities, labels, "
    "relationships, directional flow, and practical meaning."
)


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file as base64 text."""

    if not image_path.exists():
        raise RetrievalError(f"Image path does not exist: {image_path}")
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _resolve_caption_text(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    response_text = response.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    raise RetrievalError("Ollama vision response did not include caption text.")


def _coerce_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def generate_image_caption(
    image_path: Path,
    *,
    prompt: str | None = None,
    model: str = "llava-llama3",
    ollama_base_url: str = "http://localhost:11434",
    timeout_seconds: float = 60.0,
    ollama_api_key: str = "",
    client: OllamaClient | None = None,
) -> str:
    """Generate a caption for one image using Ollama chat vision input."""

    prompt_text = (
        prompt.strip()
        if isinstance(prompt, str) and prompt.strip()
        else DEFAULT_CAPTION_PROMPT
    )
    image_b64 = encode_image_to_base64(image_path)
    resolved_client = client or OllamaClient(
        base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        api_key=ollama_api_key,
    )
    response = resolved_client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt_text,
                "images": [image_b64],
            }
        ],
        stream=False,
        options={
            "temperature": 0.2,
            "num_predict": 256,
        },
    )
    return _resolve_caption_text(response)


def batch_caption_images(
    image_paths: list[Path],
    *,
    prompt: str | None = None,
    model: str = "llava-llama3",
    ollama_base_url: str = "http://localhost:11434",
    timeout_seconds: float = 60.0,
    ollama_api_key: str = "",
) -> list[dict[str, object]]:
    """Caption multiple images and return success/error rows."""

    client = OllamaClient(
        base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        api_key=ollama_api_key,
    )
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        try:
            caption = generate_image_caption(
                image_path,
                prompt=prompt,
                model=model,
                client=client,
                ollama_base_url=ollama_base_url,
                timeout_seconds=timeout_seconds,
            )
            rows.append(
                {
                    "image_path": str(image_path),
                    "caption": caption,
                    "success": True,
                }
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            rows.append(
                {
                    "image_path": str(image_path),
                    "caption": "",
                    "success": False,
                    "error": str(exc),
                }
            )
    return rows


def extract_and_caption_diagrams(
    pdf_path: Path,
    *,
    output_dir: Path,
    min_size: int = 200,
) -> list[dict[str, object]]:
    """Extract diagram candidates and return caption-ready metadata.

    Note: this helper identifies diagram regions. Persisting/cropping actual image
    files is left to caller-specific tooling.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    diagrams = extract_diagram_regions(pdf_path, min_size=min_size)
    rows: list[dict[str, object]] = []
    for idx, diagram in enumerate(diagrams, start=1):
        page_num = _coerce_int(diagram.get("page_num", 0))
        rows.append(
            {
                "diagram_id": f"diagram_p{page_num}_{idx}",
                "page_num": page_num,
                "region": diagram,
                "caption": (
                    f"Diagram candidate on page {page_num} with size "
                    f"{diagram.get('width', 0)}x{diagram.get('height', 0)}."
                ),
            }
        )
    return rows
