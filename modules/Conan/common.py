"""Small shared Conan helpers that stay dependency-light and cycle-safe."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def first_present(mapping: Optional[Mapping[str, Any]], *keys: str, default=None):
    if not isinstance(mapping, Mapping):
        return default
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


__all__ = ["first_present"]
