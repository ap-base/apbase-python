"""Thin Python wrappers around the IDW native extension.

The module centralizes F2PY loading, converts NumPy inputs to contiguous
``float64`` buffers, and returns native status values to the high-level API.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = (
    "idw_local",
)


def _as_float64_contiguous(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def load_idw_module() -> ModuleType:
    """Return the loaded IDW native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"IDW extension is unavailable: missing {', '.join(missing)}")
    return module


def run_idw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    radius: float,
    power: float,
    max_neighbors: int,
    min_neighbors: int,
    n_threads: int,
) -> tuple[np.ndarray, int]:
    """Run IDW for arbitrary target coordinates."""
    module = load_idw_module()
    try:
        estimates, status = module.idw_local(
            _as_float64_contiguous(x),
            _as_float64_contiguous(y),
            _as_float64_contiguous(z),
            _as_float64_contiguous(target_x),
            _as_float64_contiguous(target_y),
            float(radius),
            float(power),
            int(max_neighbors),
            int(min_neighbors),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native IDW extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native IDW execution failed: {exc}") from exc

    return np.asarray(estimates, dtype=np.float64), int(status)


__all__ = [
    "load_idw_module",
    "run_idw",
]
