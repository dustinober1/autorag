"""Shared coercion helpers for loosely typed values."""

from __future__ import annotations


def coerce_float(value: object, *, default: float = 0.0) -> float:
    """Best-effort float coercion with an explicit default fallback."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return default
    return default


def coerce_non_negative_int(value: object, *, default: int = 0) -> int:
    """Best-effort integer coercion clamped at zero."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str) and value.strip():
        try:
            return max(0, int(value))
        except ValueError:
            return max(0, default)
    return max(0, default)
