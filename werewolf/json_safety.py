"""Small helpers for safely consuming untrusted persisted JSON values."""
from __future__ import annotations

import math
from typing import Any


def as_mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def nonnegative_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


__all__ = ["as_mapping", "nonnegative_finite_number", "nonnegative_int"]
