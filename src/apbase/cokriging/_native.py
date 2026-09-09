"""Thin Python wrappers around the cokriging native extension.

``"collocated"``, ``"icm"``, and ``"lmc"`` all have native kernels now.
Each method's fitted-model shape is genuinely different (a flat
``[model_id, nugget, partial_sill, range, sse, rho_1..rho_K]`` vector for
collocated; a scalar (model_id, range) plus two (K+1)x(K+1) matrices for
ICM; a scalar model_id, S-1 ranges, and an (S, K+1, K+1) coefficient array
for LMC), so there is no single generic ``run_cokriging(method, ...)``
dispatcher here -- ``_interpolation.py`` calls the method-specific functions
directly.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = (
    "collocated_cokriging_local",
    "fit_icm_model",
    "co_kriging_icm_local",
    "fit_lmc_model",
    "co_kriging_lmc_local",
)


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_int32_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.int32 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.int32)


def load_cokriging_module() -> ModuleType:
    """Return the loaded cokriging native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"cokriging extension is unavailable: missing {', '.join(missing)}")
    return module


def run_collocated_cokriging(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    model_info: np.ndarray,
    secondary_x: np.ndarray,
    secondary_y: np.ndarray,
    secondary_z: np.ndarray,
    secondary_offset: np.ndarray,
    radius: float,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[np.ndarray, int]:
    """Run collocated cokriging (MM1) for the provided target coordinates.

    ``secondary_x``/``secondary_y``/``secondary_z`` are the K standardized
    secondary source arrays concatenated end to end; ``secondary_offset``
    (length K+1) gives CSR-style boundaries into them (see
    ``cokriging/_interpolation.py`` for how these are assembled).
    """
    module = load_cokriging_module()
    try:
        estimates, status = module.collocated_cokriging_local(
            _as_float64_contiguous(x0),
            _as_float64_contiguous(y0),
            _as_float64_contiguous(z0),
            _as_float64_contiguous(target_x),
            _as_float64_contiguous(target_y),
            _as_float64_contiguous(model_info),
            _as_float64_contiguous(secondary_x),
            _as_float64_contiguous(secondary_y),
            _as_float64_contiguous(secondary_z),
            _as_int32_contiguous(secondary_offset),
            float(radius),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cokriging extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native cokriging execution failed: {exc}") from exc
    return np.asarray(estimates, dtype=np.float64), int(status)


def run_fit_icm_model(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    secondary_x: np.ndarray,
    secondary_y: np.ndarray,
    secondary_z: np.ndarray,
    secondary_offset: np.ndarray,
    n_lags: int,
    max_pairs: int,
    max_distance: float,
    max_colocation_dist: float,
    n_threads: int,
) -> tuple[float, float, np.ndarray, np.ndarray, int]:
    """Jointly fit the ICM coregionalization model (shared shape/range, one B matrix).

    Returns ``(model_id, model_range, nugget_matrix, sill_matrix, status)``;
    ``nugget_matrix``/``sill_matrix`` are full (K+1)x(K+1) symmetric arrays
    (variable 0 = primary, 1..K = secondaries in ``secondary_offset`` order).
    """
    module = load_cokriging_module()
    try:
        model_id, model_range, nugget_matrix, sill_matrix, status = module.fit_icm_model(
            _as_float64_contiguous(x0),
            _as_float64_contiguous(y0),
            _as_float64_contiguous(z0),
            _as_float64_contiguous(secondary_x),
            _as_float64_contiguous(secondary_y),
            _as_float64_contiguous(secondary_z),
            _as_int32_contiguous(secondary_offset),
            int(n_lags),
            int(max_pairs),
            float(max_distance),
            float(max_colocation_dist),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cokriging extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error fitting ICM model: {exc}") from exc
    return (
        float(model_id),
        float(model_range),
        np.asarray(nugget_matrix, dtype=np.float64),
        np.asarray(sill_matrix, dtype=np.float64),
        int(status),
    )


def run_co_kriging_icm(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    model_id: float,
    model_range: float,
    nugget_matrix: np.ndarray,
    sill_matrix: np.ndarray,
    secondary_x: np.ndarray,
    secondary_y: np.ndarray,
    secondary_z: np.ndarray,
    secondary_offset: np.ndarray,
    radius: float,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[np.ndarray, int]:
    """Run ICM cokriging for the provided target coordinates."""
    module = load_cokriging_module()
    try:
        estimates, status = module.co_kriging_icm_local(
            _as_float64_contiguous(x0),
            _as_float64_contiguous(y0),
            _as_float64_contiguous(z0),
            _as_float64_contiguous(target_x),
            _as_float64_contiguous(target_y),
            float(model_id),
            float(model_range),
            _as_float64_contiguous(nugget_matrix),
            _as_float64_contiguous(sill_matrix),
            _as_float64_contiguous(secondary_x),
            _as_float64_contiguous(secondary_y),
            _as_float64_contiguous(secondary_z),
            _as_int32_contiguous(secondary_offset),
            float(radius),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cokriging extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native cokriging execution failed: {exc}") from exc
    return np.asarray(estimates, dtype=np.float64), int(status)


def run_fit_lmc_model(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    secondary_x: np.ndarray,
    secondary_y: np.ndarray,
    secondary_z: np.ndarray,
    secondary_offset: np.ndarray,
    n_structures: int,
    n_lags: int,
    max_pairs: int,
    max_distance: float,
    max_colocation_dist: float,
    n_threads: int,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Jointly fit the LMC coregionalization model via Goulard & Voltz (1992).

    Returns ``(model_id, structure_ranges, coefficient_matrices, status)``;
    ``coefficient_matrices`` has shape ``(n_structures, K+1, K+1)`` --
    index 0 is the nugget structure (no range), indices 1..n_structures-1
    are the shared continuous shape at ``structure_ranges[l-1]``.
    """
    module = load_cokriging_module()
    try:
        model_id, structure_ranges, coefficient_matrices, status = module.fit_lmc_model(
            _as_float64_contiguous(x0),
            _as_float64_contiguous(y0),
            _as_float64_contiguous(z0),
            _as_float64_contiguous(secondary_x),
            _as_float64_contiguous(secondary_y),
            _as_float64_contiguous(secondary_z),
            _as_int32_contiguous(secondary_offset),
            int(n_structures),
            int(n_lags),
            int(max_pairs),
            float(max_distance),
            float(max_colocation_dist),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cokriging extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error fitting LMC model: {exc}") from exc
    return (
        float(model_id),
        np.asarray(structure_ranges, dtype=np.float64),
        np.asarray(coefficient_matrices, dtype=np.float64),
        int(status),
    )


def run_co_kriging_lmc(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    model_id: float,
    structure_ranges: np.ndarray,
    coefficient_matrices: np.ndarray,
    secondary_x: np.ndarray,
    secondary_y: np.ndarray,
    secondary_z: np.ndarray,
    secondary_offset: np.ndarray,
    radius: float,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[np.ndarray, int]:
    """Run LMC cokriging for the provided target coordinates."""
    module = load_cokriging_module()
    try:
        estimates, status = module.co_kriging_lmc_local(
            _as_float64_contiguous(x0),
            _as_float64_contiguous(y0),
            _as_float64_contiguous(z0),
            _as_float64_contiguous(target_x),
            _as_float64_contiguous(target_y),
            float(model_id),
            _as_float64_contiguous(structure_ranges),
            _as_float64_contiguous(coefficient_matrices),
            _as_float64_contiguous(secondary_x),
            _as_float64_contiguous(secondary_y),
            _as_float64_contiguous(secondary_z),
            _as_int32_contiguous(secondary_offset),
            float(radius),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cokriging extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native cokriging execution failed: {exc}") from exc
    return np.asarray(estimates, dtype=np.float64), int(status)


__all__ = [
    "load_cokriging_module",
    "run_co_kriging_icm",
    "run_co_kriging_lmc",
    "run_collocated_cokriging",
    "run_fit_icm_model",
    "run_fit_lmc_model",
]
