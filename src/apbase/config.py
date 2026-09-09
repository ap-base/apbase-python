"""Process-wide default overrides for apbase classes."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import overload

_ConfigValue = int | float


def _positive_int(key: str, value: object) -> int:
    int_value = int(value)  # type: ignore[call-overload]
    if int_value <= 0:
        raise ValueError(f"{key} must be greater than zero")
    return int_value


def _non_negative_int(key: str, value: object) -> int:
    int_value = int(value)  # type: ignore[call-overload]
    if int_value < 0:
        raise ValueError(f"{key} must be >= 0")
    return int_value


def _non_negative_float(key: str, value: object) -> float:
    float_value = float(value)  # type: ignore[arg-type]
    if float_value < 0.0:
        raise ValueError(f"{key} must be >= 0")
    return float_value


_VALIDATORS: dict[str, Callable[[str, object], _ConfigValue]] = {
    "n_threads": _positive_int,
    "variogram_n_lags": _positive_int,
    "variogram_max_pairs": _non_negative_int,
    "variogram_max_distance": _non_negative_float,
    "max_neighbors_ceiling": _positive_int,
    "variogram_max_pairs_ceiling": _positive_int,
    "local_mode_extent_km_ceiling": _non_negative_float,
    "grid_max_points_ceiling": _positive_int,
    "max_secondary_variables_ceiling": _positive_int,
    "max_lmc_structures_ceiling": _positive_int,
}


class _Config:
    """Global settings consulted when a class is constructed without an explicit value.

    Example
    -------
    >>> import apbase
    >>> apbase.config["n_threads"] = 4
    >>> apbase.Kriging().n_threads
    4
    """

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, _ConfigValue] = {}

    def __setitem__(self, key: str, value: object) -> None:
        validator = _VALIDATORS.get(key)
        if validator is None:
            raise KeyError(f"unknown apbase.config key: {key!r}")
        self._values[key] = validator(key, value)

    def __getitem__(self, key: str) -> _ConfigValue:
        try:
            return self._values[key]
        except KeyError:
            raise KeyError(f"{key!r} is not set; valid keys: {sorted(_VALIDATORS)}") from None

    @overload
    def get(self, key: str) -> _ConfigValue | None: ...
    @overload
    def get(self, key: str, default: _ConfigValue) -> _ConfigValue: ...

    def get(self, key: str, default: _ConfigValue | None = None) -> _ConfigValue | None:
        return self._values.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __repr__(self) -> str:
        return f"apbase.config({self._values!r})"


config = _Config()


def resolve_n_threads() -> int:
    """Resolve n_threads: apbase.config > APBASE_N_THREADS env var > 1."""
    configured = config.get("n_threads")
    if configured is not None:
        return int(configured)

    env_value = os.environ.get("APBASE_N_THREADS")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            warnings.warn(
                f"APBASE_N_THREADS={env_value!r} is not a valid integer; ignoring it and falling back to 1",
                UserWarning,
                stacklevel=2,
            )

    return 1


def resolve_variogram_profile() -> tuple[int, int, float]:
    """Resolve the shared ``(n_lags, max_pairs, max_distance)`` variogram profile.

    This is the internal variogram fit used to derive a default search radius
    across :class:`~apbase.kriging.Kriging`, :class:`~apbase.idw.IDW`,
    :func:`~apbase.cross_validate.cross_validate`, and
    :func:`~apbase.mapping.create_map`/:class:`~apbase.mapping.Map` when
    ``radius``/``model_values`` are not supplied explicitly -- kept in one
    place so the four stay consistent with each other by construction,
    instead of by four separately maintained copies. ``apbase.config``
    overrides the default for each key independently; unset keys fall back to
    ``(50, 100_000, 0.0)``.

    Not used by :class:`~apbase.filtering.SpatialFilter`, which fits its own,
    intentionally different (lighter/faster) variogram profile internally.
    """
    n_lags = config.get("variogram_n_lags", 50)
    max_pairs = config.get("variogram_max_pairs", 100_000)
    max_distance = config.get("variogram_max_distance", 0.0)
    return int(n_lags), int(max_pairs), float(max_distance)


def resolve_max_neighbors_ceiling() -> int:
    """Resolve the hard cap on ``max_neighbors`` for Kriging/cross_validate.

    Not a fixed native buffer size -- the underlying Fortran arrays are
    allocated dynamically at this size -- so raising it is safe, just
    slower/more memory per target point (local kriging cost grows with
    ``max_neighbors`` roughly quadratically to cubically: a bigger linear
    system solved per target). Defaults to ``500``; override with
    ``apbase.config["max_neighbors_ceiling"]``. Not enforced by ``IDW`` or
    ``SpatialFilter``, which have no ceiling today.
    """
    return int(config.get("max_neighbors_ceiling", 500))


def resolve_variogram_max_pairs_ceiling() -> int:
    """Resolve the hard cap on ``Variogram.max_pairs``.

    Not a fixed native buffer size -- the underlying Fortran pair-sampling
    array is allocated dynamically at this size (its own internal default,
    when ``max_pairs<=0``, is 1,000,000) -- so raising it is safe, just
    slower to fit. Defaults to ``200_000``; override with
    ``apbase.config["variogram_max_pairs_ceiling"]``.
    """
    return int(config.get("variogram_max_pairs_ceiling", 200_000))


def resolve_local_mode_extent_km_ceiling() -> float:
    """Resolve the extent ceiling (bounding-box diagonal, km) for ``geographic_mode="local"``.

    :func:`~apbase.common.coordinates.prepare_metric_xy` and
    :func:`~apbase.common.coordinates.reproject_geographic_xy` approximate
    distance with a flat equirectangular tangent plane anchored on the
    input's own mean lon/lat when ``geographic_mode="local"``; the
    approximation error grows with distance from that anchor, so both
    functions reject a ``"local"`` call whose input extent exceeds this
    ceiling and ask for ``geographic_mode="utm"`` instead. Defaults to
    ``50.0`` km; override with
    ``apbase.config["local_mode_extent_km_ceiling"]`` (``0`` disables
    ``"local"`` entirely, forcing ``"utm"`` for any geographic input).
    """
    return float(config.get("local_mode_extent_km_ceiling", 50.0))


def resolve_grid_max_points_ceiling() -> int:
    """Resolve the hard cap on ``Grid``'s candidate axis-grid size (``nx * ny``).

    Checked before the chunked ``x``/``y`` candidate arrays are built, so a
    ``resolution``/``boundary`` mismatch fails instantly with a clear error
    instead of silently chewing through minutes of ``np.tile``/``np.repeat``
    chunk generation and eventually raising ``MemoryError`` (or exhausting
    system RAM first) -- the most common trigger is geographic input
    auto-detected and reprojected to UTM meters while ``resolution`` was
    still sized for a much coarser local unit, turning a modest grid into a
    multi-billion-cell one. Defaults to ``100_000_000``; override with
    ``apbase.config["grid_max_points_ceiling"]`` for legitimate very large
    rasters.
    """
    return int(config.get("grid_max_points_ceiling", 100_000_000))


def resolve_max_secondary_variables_ceiling() -> int:
    """Resolve the hard cap on the number of secondary variables (K) for cokriging.

    ICM/LMC bin and jointly fit every one of the (K+1)(K+2)/2 pairs among the
    K secondaries and the primary (icm_fit.f90/lmc_fit.f90), and the
    per-target block-kriging system grows with K too -- both costs climb
    quadratically-ish with K, the same risk profile
    ``max_neighbors_ceiling`` already mitigates for neighbor counts.
    Defaults to ``20``; override with
    ``apbase.config["max_secondary_variables_ceiling"]``.
    """
    return int(config.get("max_secondary_variables_ceiling", 20))


def resolve_max_lmc_structures_ceiling() -> int:
    """Resolve the hard cap on the number of nested LMC structures (S).

    The LMC coregionalization vector scales as ``S*(K+1)(K+2)/2`` (plus S
    ranges) -- unbounded S grows both the fit and per-target solve cost
    without limit, same rationale as the other ceilings in this module.
    Defaults to ``5``; override with
    ``apbase.config["max_lmc_structures_ceiling"]``. Not enforced before
    Milestone 4 (LMC).
    """
    return int(config.get("max_lmc_structures_ceiling", 5))


__all__ = [
    "config",
    "resolve_grid_max_points_ceiling",
    "resolve_local_mode_extent_km_ceiling",
    "resolve_max_lmc_structures_ceiling",
    "resolve_max_neighbors_ceiling",
    "resolve_max_secondary_variables_ceiling",
    "resolve_n_threads",
    "resolve_variogram_max_pairs_ceiling",
    "resolve_variogram_profile",
]
