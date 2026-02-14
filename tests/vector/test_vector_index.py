from __future__ import annotations

import numpy as np

from autokg_rag.vector.index import search_top_k


def test_vector_index_roundtrip_and_topk_ordering() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.7, 0.7],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    query_vector = np.array([1.0, 0.0], dtype=np.float32)

    hits = search_top_k(query_vector=query_vector, embeddings=embeddings, top_k=3)

    assert len(hits) == 3
    assert hits[0][0] == 0

    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
