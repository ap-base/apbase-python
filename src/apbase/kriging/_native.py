"""Thin Python wrappers around the kriging native extension.

The module keeps all F2PY loading in one place, normalizes NumPy arrays to
contiguous ``float64`` buffers, and converts native status values into Python
types for the high-level classes.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = ("ordinary_kriging_local",)


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def load_kriging_module() -> ModuleType:
    """Return the loaded kriging native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"kriging extension is unavailable: missing {', '.join(missing)}")
    return module


def run_kriging(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    model_info: np.ndarray,
    radius: float,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[np.ndarray, int]:
    """Run local ordinary kriging for the provided target coordinates."""
    module = load_kriging_module()
    try:
        estimates, status = module.ordinary_kriging_local(
            _as_float64_contiguous(x),
            _as_float64_contiguous(y),
            _as_float64_contiguous(z),
            _as_float64_contiguous(target_x),
            _as_float64_contiguous(target_y),
            _as_float64_contiguous(model_info),
            float(radius),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native kriging extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native kriging execution failed: {exc}") from exc
    return np.asarray(estimates, dtype=np.float64), int(status)


__all__ = [
    "load_kriging_module",
    "run_kriging",
]
