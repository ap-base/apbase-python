"""Thin Python wrapper around the cross-validation native extension.

The module keeps all F2PY loading in one place, normalizes NumPy arrays to
contiguous buffers, and converts native status values into Python types for
the high-level API.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = ("cross_validate_local", "cokriging_cross_validate_local")


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_int32_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.int32 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.int32)


def load_cross_validate_module() -> ModuleType:
    """Return the loaded cross-validation native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"cross_validate extension is unavailable: missing {', '.join(missing)}")
    return module


def run_cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sample_idx: np.ndarray,
    radius: float,
    power: float,
    model_info: np.ndarray,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, int, float, float, float, int, int]:
    """Run leave-one-out cross-validation for IDW and ordinary kriging together."""
    module = load_cross_validate_module()
    try:
        (
            idw_pred,
            krig_pred,
            idw_mae,
            idw_rmse,
            idw_r2,
            idw_n_valid,
            krig_mae,
            krig_rmse,
            krig_r2,
            krig_n_valid,
            status,
        ) = module.cross_validate_local(
            _as_float64_contiguous(x),
            _as_float64_contiguous(y),
            _as_float64_contiguous(z),
            _as_int32_contiguous(sample_idx),
            float(radius),
            float(power),
            _as_float64_contiguous(model_info),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cross_validate extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native cross_validate execution failed: {exc}") from exc

    return (
        np.asarray(idw_pred, dtype=np.float64),
        np.asarray(krig_pred, dtype=np.float64),
        float(idw_mae),
        float(idw_rmse),
        float(idw_r2),
        int(idw_n_valid),
        float(krig_mae),
        float(krig_rmse),
        float(krig_r2),
        int(krig_n_valid),
        int(status),
    )


def run_cokriging_cross_validate(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    sample_idx: np.ndarray,
    secondary_x: np.ndarray,
    secondary_y: np.ndarray,
    secondary_z: np.ndarray,
    secondary_offset: np.ndarray,
    collocated_model: np.ndarray,
    icm_model_id: float,
    icm_range: float,
    icm_nugget: np.ndarray,
    icm_sill: np.ndarray,
    lmc_model_id: float,
    lmc_ranges: np.ndarray,
    lmc_coefficients: np.ndarray,
    radius: float,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    float, float, float, int,
    float, float, float, int,
    float, float, float, int,
    int,
]:
    """Run leave-one-out cross-validation for collocated/ICM/LMC cokriging together.

    All three methods share one primary neighbor search and evaluate the
    same fixed coregionalization models (fitted once beforehand) -- only the
    interpolation step is leave-one-out, matching :func:`run_cross_validate`.
    """
    module = load_cross_validate_module()
    try:
        (
            collocated_pred, icm_pred, lmc_pred,
            collocated_mae, collocated_rmse, collocated_r2, collocated_n_valid,
            icm_mae, icm_rmse, icm_r2, icm_n_valid,
            lmc_mae, lmc_rmse, lmc_r2, lmc_n_valid,
            status,
        ) = module.cokriging_cross_validate_local(
            _as_float64_contiguous(x0),
            _as_float64_contiguous(y0),
            _as_float64_contiguous(z0),
            _as_int32_contiguous(sample_idx),
            _as_float64_contiguous(secondary_x),
            _as_float64_contiguous(secondary_y),
            _as_float64_contiguous(secondary_z),
            _as_int32_contiguous(secondary_offset),
            1,
            _as_float64_contiguous(collocated_model),
            1,
            float(icm_model_id),
            float(icm_range),
            _as_float64_contiguous(icm_nugget),
            _as_float64_contiguous(icm_sill),
            1,
            float(lmc_model_id),
            _as_float64_contiguous(lmc_ranges),
            _as_float64_contiguous(lmc_coefficients),
            float(radius),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native cross_validate extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native cokriging cross-validation execution failed: {exc}") from exc

    return (
        np.asarray(collocated_pred, dtype=np.float64),
        np.asarray(icm_pred, dtype=np.float64),
        np.asarray(lmc_pred, dtype=np.float64),
        float(collocated_mae), float(collocated_rmse), float(collocated_r2), int(collocated_n_valid),
        float(icm_mae), float(icm_rmse), float(icm_r2), int(icm_n_valid),
        float(lmc_mae), float(lmc_rmse), float(lmc_r2), int(lmc_n_valid),
        int(status),
    )


__all__ = [
    "load_cross_validate_module",
    "run_cokriging_cross_validate",
    "run_cross_validate",
]
