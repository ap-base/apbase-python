from __future__ import annotations

import numpy as np

from .exceptions import ValidationError


def require_positive_finite(value: float, name: str) -> None:
    """Validate that a scalar value is finite and strictly positive.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Human-readable parameter name used in the error message.

    Raises
    ------
    ValidationError
        If ``value`` is not finite or is less than or equal to zero.
    """
    if not np.isfinite(value) or value <= 0.0:
        raise ValidationError(f"{name} must be finite and greater than zero")


def require_non_negative_finite(value: float, name: str) -> None:
    """Validate that a scalar value is finite and non-negative.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Human-readable parameter name used in the error message.

    Raises
    ------
    ValidationError
        If ``value`` is not finite or is negative.
    """
    if not np.isfinite(value) or value < 0.0:
        raise ValidationError(f"{name} must be finite and >= 0")


def require_neighbor_bounds(
    max_neighbors: int,
    min_neighbors: int,
    *,
    max_neighbors_ceiling: int | None = None,
) -> None:
    """Validate local-neighborhood bounds.

    Parameters
    ----------
    max_neighbors : int
        Maximum neighbors allowed in a local search.
    min_neighbors : int
        Minimum neighbors required to compute an estimate.
    max_neighbors_ceiling : int or None, default None
        Optional global ceiling applied to ``max_neighbors``.

    Raises
    ------
    ValidationError
        If the bounds are non-positive, inconsistent, or above the optional
        ceiling.
    """
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
