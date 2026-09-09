"""Thin Python wrappers around the variogram native extension."""

from __future__ import annotations

from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = (
    "evaluate_variogram",
    "fit_variogram_models",
    "select_variogram",
)


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def load_variogram_module() -> ModuleType:
    """Return the loaded variogram native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"variogram extension is unavailable: missing {', '.join(missing)}")
    return module


def run_variogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_lags: int,
    max_pairs: int,
    max_distance: float,
    n_threads: int,
) -> np.ndarray:
    """Fit candidate variogram models with the native implementation."""
    module = load_variogram_module()
    try:
        result = module.fit_variogram_models(
            _as_float64_contiguous(x),
            _as_float64_contiguous(y),
            _as_float64_contiguous(z),
            int(n_lags),
            int(max_pairs),
            float(max_distance),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native variogram extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error fitting variogram: {exc}") from exc
    return np.asarray(result, dtype=np.float64)


def run_select_variogram(model_info: np.ndarray) -> tuple[np.ndarray, int]:
    """Select the best variogram model from native model values."""
    module = load_variogram_module()
    try:
        params_array, status = module.select_variogram(_as_float64_contiguous(model_info))
    except AttributeError:
        raise NativeExtensionError("native variogram extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error selecting variogram: {exc}") from exc
    return np.asarray(params_array, dtype=np.float64), int(status)


def run_evaluate_variogram(
    model_info: np.ndarray,
    distance: ArrayLike,
    n_threads: int,
) -> tuple[np.ndarray, int]:
    """Evaluate a fitted variogram model at one or more distances."""
    module = load_variogram_module()
    distance_array = np.asarray(distance, dtype=np.float64)
    try:
        gamma, status = module.evaluate_variogram(
            _as_float64_contiguous(model_info),
            _as_float64_contiguous(distance_array.reshape(-1)),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native variogram extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"error evaluating variogram: {exc}") from exc
    return np.asarray(gamma, dtype=np.float64).reshape(distance_array.shape), int(status)


__all__ = [
    "load_variogram_module",
    "run_evaluate_variogram",
    "run_select_variogram",
    "run_variogram",
]
