"""APBase: high-performance automatic map generation.

The primary entry point is :class:`Map`: pass metric ``x``/``y`` coordinates,
values, and a resolution, and APBase filters, builds the grid, fits one
shared variogram, cross-validates IDW against ordinary kriging, selects the
statistically better method via :func:`select_best_model`, and interpolates
the map. :func:`create_map` is the functional shortcut around the same
pipeline.
"""

from __future__ import annotations

from importlib.metadata import version as _version

from .cokriging import CoKriging, co_kriging
from .common import (
    ApbaseError,
    CoordinateTransform,
    CrossValidationError,
    CrossVariogramError,
    InterpolationError,
    NativeExecutionError,
    NativeExtensionError,
    SpatialFilterError,
    ValidationError,
    VariogramError,
    apply_metric_transform,
    prepare_metric_xy,
    reproject_geographic_xy,
    restore_original_xy,
)
from .config import config
from .cross_validate import (
    BestModelResult,
    CokrigingCrossValidationResult,
    CrossValidationResult,
    MethodCrossValidation,
    MultiMethodBestModelResult,
    cross_validate,
    cross_validate_cokriging,
    select_best_model,
)
from .cross_variogram import CrossVariogram, screen_secondary_variables
from .filtering import SpatialFilter
from .grid import Grid
from .idw import IDW
from .kriging import Kriging
from .mapping import Map, MapResult, create_map
from .variogram import Variogram

__version__ = _version("apbase")

__all__ = [
    "IDW",
    "BestModelResult",
    "CoKriging",
    "CokrigingCrossValidationResult",
    "CoordinateTransform",
    "CrossValidationResult",
    "CrossVariogram",
    "Kriging",
    "Grid",
    "Map",
    "MapResult",
    "MethodCrossValidation",
    "MultiMethodBestModelResult",
    "SpatialFilter",
    "Variogram",
    "__version__",
    "apply_metric_transform",
    "co_kriging",
    "config",
    "create_map",
    "cross_validate",
    "cross_validate_cokriging",
    "prepare_metric_xy",
    "reproject_geographic_xy",
    "restore_original_xy",
    "screen_secondary_variables",
    "select_best_model",
    "ApbaseError",
    "CrossValidationError",
    "CrossVariogramError",
    "InterpolationError",
    "NativeExecutionError",
    "NativeExtensionError",
    "SpatialFilterError",
    "ValidationError",
    "VariogramError",
]
