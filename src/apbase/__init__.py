"""APBase: high-performance automatic map generation.

The public top-level API is intentionally small: use :class:`Map` or
:func:`create_map` to create maps, and :data:`config` for process-wide
runtime settings. Lower-level geostatistical building blocks remain available
from their dedicated submodules for advanced workflows.
"""

from __future__ import annotations

from importlib.metadata import version as _version

from .config import config
from .mapping import Map, MapResult, create_map

__version__ = _version("apbase")

__all__ = [
    "Map",
    "MapResult",
    "__version__",
    "config",
    "create_map",
]
