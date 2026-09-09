from __future__ import annotations

import numpy as np

from .exceptions import ValidationError


def require_positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be finite and greater than zero")


def require_non_negative_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be finite and >= 0")


def require_neighbor_bounds(
    max_neighbors: int,
    min_neighbors: int,
    *,
    max_neighbors_ceiling: int | None = None,
) -> None:
    """Shared max_neighbors/min_neighbors bound check (idw/kriging/cross_validate)."""
    if max_neighbors <= 0:
        raise ValidationError("max_neighbors must be greater than zero")
    if max_neighbors_ceiling is not None and max_neighbors > max_neighbors_ceiling:
        raise ValidationError(f"max_neighbors must be <= {max_neighbors_ceiling}")
    if min_neighbors <= 0 or min_neighbors > max_neighbors:
        raise ValidationError("min_neighbors must be between 1 and max_neighbors")


__all__ = [
    "require_neighbor_bounds",
    "require_non_negative_finite",
    "require_positive_finite",
]
