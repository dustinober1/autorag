"""Vector similarity indexing primitives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cosine_scores(
    query_vector: npt.NDArray[np.float32],
    embeddings: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute cosine similarity between one query and an embedding matrix."""

    if embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)

    query = query_vector.astype(np.float32)
    matrix = embeddings.astype(np.float32)

    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(matrix, axis=1)

    denom = np.maximum(doc_norms * query_norm, 1e-8)
    numer = matrix @ query

    return np.asarray(numer / denom, dtype=np.float32)


def search_top_k(
    query_vector: npt.NDArray[np.float32],
    embeddings: npt.NDArray[np.float32],
    top_k: int,
) -> list[tuple[int, float]]:
    """Return top-k `(row_idx, score)` tuples sorted by descending score."""

    scores = cosine_scores(query_vector=query_vector, embeddings=embeddings)
    if scores.size == 0:
        return []

    top_k = max(1, min(top_k, int(scores.shape[0])))
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [(int(row_idx), float(scores[row_idx])) for row_idx in ranked_indices]
