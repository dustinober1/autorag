"""Canonicalization helpers for ontology extraction."""

from __future__ import annotations

import hashlib
import re

_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize_entity_name(name: str) -> str:
    """Normalize an entity mention into canonical lowercase form."""

    cleaned = _NON_ALNUM_RE.sub(" ", name.lower())
    compact = _SPACE_RE.sub(" ", cleaned).strip()
    return compact


def make_node_id(canonical_name: str) -> str:
    """Build stable node identifier from canonical entity name."""

    digest = hashlib.sha1(canonical_name.encode("utf-8")).hexdigest()[:12]
    return f"node_{digest}"


def make_edge_id(source_node_id: str, relation: str, target_node_id: str) -> str:
    """Build stable edge identifier from source/relation/target triple."""

    payload = f"{source_node_id}|{relation}|{target_node_id}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]
    return f"edge_{digest}"
