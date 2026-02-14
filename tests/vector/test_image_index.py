from __future__ import annotations

from pathlib import Path

import numpy as np

from autokg_rag.vector.image_index import ImageIndex


class _FakeEmbeddingProvider:
    model_name = "fake"
    dim = 3

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            token_count = float(len(text.split()))
            has_risk = 1.0 if "risk" in text.lower() else 0.0
            rows.append([token_count, has_risk, 0.5])
        return np.asarray(rows, dtype=np.float32)


def test_image_index_search_returns_ranked_rows(tmp_path: Path) -> None:
    provider = _FakeEmbeddingProvider()
    index = ImageIndex.from_captions(
        captions=["Risk matrix diagram", "WBS hierarchy chart"],
        image_refs=["img_a", "img_b"],
        page_nums=[10, 12],
        embedding_provider=provider,  # type: ignore[arg-type]
    )
    index.save(tmp_path)

    reloaded = ImageIndex.load(
        artifact_dir=tmp_path,
        embedding_provider=provider,  # type: ignore[arg-type]
    )
    hits = reloaded.search("Risk matrix", top_k=1)

    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "img_0"
    assert float(hits[0]["score"]) >= 0.0
