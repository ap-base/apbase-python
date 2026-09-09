"""Thin Python wrappers around the spatial filtering native extension."""

from __future__ import annotations

from types import ModuleType

import numpy as np

from apbase._native import lib as _shared_lib
from apbase.common.exceptions import NativeExecutionError, NativeExtensionError

REQUIRED_SYMBOLS = (
    "viz_filter_mask",
    "viz_zstats_radius_gridagg",
)


def load_filter_module() -> ModuleType:
    """Return the loaded spatial filtering native extension."""
    module = _shared_lib.load()
    missing = [symbol for symbol in REQUIRED_SYMBOLS if not hasattr(module, symbol)]
    if missing:
        raise NativeExtensionError(f"spatial filter extension is unavailable: missing {', '.join(missing)}")
    return module


def run_local_zstats(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radius: float,
    cell_factor: int,
    max_neighbors: int,
    candidate_fraction: float,
    dedup_radius: float,
    dedup_z_tol: float,
    n_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run local z-statistics for source points."""
    module = load_filter_module()
    try:
        pct_diff, local_z, local_prob, status = module.viz_zstats_radius_gridagg(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(z, dtype=np.float64),
            float(radius),
            int(cell_factor),
            int(max_neighbors),
            float(candidate_fraction),
            float(dedup_radius),
            float(dedup_z_tol),
            int(n_threads),
        )
    except AttributeError:
        raise NativeExtensionError("native spatial filter extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native spatial filter execution failed: {exc}") from exc
    return (
        np.asarray(pct_diff, dtype=np.float64),
        np.asarray(local_z, dtype=np.float64),
        np.asarray(local_prob, dtype=np.float64),
        int(status),
    )


def run_filter_mask(
    pct_diff: np.ndarray,
    local_prob: np.ndarray,
    local_z: np.ndarray,
    filter_level: int,
) -> tuple[np.ndarray, int]:
    """Apply the native threshold mask to local statistics."""
    module = load_filter_module()
    try:
        mask, status = module.viz_filter_mask(
            np.ascontiguousarray(pct_diff, dtype=np.float64),
            np.ascontiguousarray(local_prob, dtype=np.float64),
            np.ascontiguousarray(local_z, dtype=np.float64),
            int(filter_level),
        )
    except AttributeError:
        raise NativeExtensionError("native spatial filter extension must be rebuilt") from None
    except Exception as exc:
        raise NativeExecutionError(f"native spatial filter mask failed: {exc}") from exc
    return np.asarray(mask != 0, dtype=bool), int(status)


__all__ = [
    "load_filter_module",
    "run_filter_mask",
    "run_local_zstats",
]
