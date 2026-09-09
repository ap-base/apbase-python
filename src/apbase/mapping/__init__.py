"""Automatic high-performance map generation.

This package exports :class:`Map`, the primary API for orchestrating
``SpatialFilter``, ``Grid``, ``Variogram``, ``cross_validate``, ``IDW``, and
``Kriging`` into one map-building pipeline. :func:`create_map` is the
functional shortcut around the same pipeline.
"""

from __future__ import annotations

from ._mapping import Map, MapResult, create_map

__all__ = ["Map", "MapResult", "create_map"]
