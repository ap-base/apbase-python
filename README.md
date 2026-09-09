# APBase — Intelligent Map-Generation Pipeline

[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![Kernel: Fortran](https://img.shields.io/badge/kernel-Fortran-734f96.svg)](#high-performance)
[![Parallel: OpenMP](https://img.shields.io/badge/parallel-OpenMP-orange.svg)](#high-performance)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](pyproject.toml)
[![CI](https://github.com/LeonardoAgricola/projeto-base-apbase-process/actions/workflows/ci.yml/badge.svg)](https://github.com/LeonardoAgricola/projeto-base-apbase-process/actions/workflows/ci.yml)
[![Build Wheels](https://github.com/LeonardoAgricola/projeto-base-apbase-process/actions/workflows/build.yml/badge.svg)](https://github.com/LeonardoAgricola/projeto-base-apbase-process/actions/workflows/build.yml)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg)]()

<p align="center">
  <img src="assets/logo_en.svg" alt="APBase — Python" width="660"/>
</p>

**The high-performance Fortran core for AI Agents and MCP servers in
precision agriculture.**

APBase is the geospatial computation layer built to be consumed by AI
Agents and MCP servers: it turns raw agricultural data — yield, soil,
sensors — into reliable maps.

**The real differentiator is the pipeline's intelligence.** APBase filters
the data, fits the variogram, cross-validates IDW, kriging, and co-kriging
against each other, and automatically picks the mathematical model with the
lowest error for each dataset — no manual intervention, always the most
accurate map possible for the data you have.

Critical numerical routines run as compiled native Fortran, exposed to
Python via F2PY/NumPy/OpenMP. This isn't a generic geostatistics package:
it's production-ready infrastructure for Agent and MCP pipelines that need
low latency, reproducible results, and scale.

## What this package does

The main entry point is `apbase.Map`.

Given `x`, `y`, `z`, and a resolution, the package automatically runs:

1. validation and filtering of the source points;
2. removal of invalid data, global outliers, and local spatial outliers;
3. fitting of an automated variogram;
4. leave-one-out cross-validation between models;
5. automatic selection of the method with the lowest RMSE;
6. final interpolation of the map.

## Key features

- **`Map`: high-performance automated pipeline for map creation.**
- `SpatialFilter`: spatial filter with global IQR and local statistics.
- `Grid`: regular grid generation within a boundary or concave hull.
- `Variogram`: fitting and evaluation of spherical, exponential, and gaussian models.
- `cross_validate`: leave-one-out comparison between IDW, kriging, and co-kriging.
- `IDW`: local inverse-distance-weighted interpolation.
- `Kriging`: local ordinary kriging with bounded neighborhoods.
- `screen_secondary_variables`: statistical screening of candidate secondary
  variables before co-kriging.
- `CoKriging`: local co-kriging (collocated, ICM, LMC) using secondary
  variables correlated with the primary one.
- `prepare_metric_xy`: preparation of metric coordinates from lon/lat
  or projected coordinates.

## High performance

APBase is built on a numerical kernel written in Fortran: the critical
routines run as precompiled native extensions, integrated with Python via
F2PY, NumPy, and OpenMP.

This is what makes APBase suitable for AI Agents and MCP servers:
low-latency responses, predictable CPU cost, and deterministic results —
without relying on Python loops or reimplementing geostatistics in every
agent. The package avoids full distance matrices and works with local
neighborhoods, neighbor limits, and internal spatial structures to keep
cost and memory under control on dense agricultural data.

## Installation

```bash
pip install apbase
```

For the latest development version, install directly from the repository:

```bash
pip install git+https://github.com/ap-base/apbase-python.git
```

For local development:

```bash
git clone https://github.com/ap-base/apbase-python.git
cd apbase-python
pip install -e .
```

## Quick usage

```python
import numpy as np

import apbase
from apbase import Map

apbase.config["n_threads"] = 4

x = np.ascontiguousarray(x_metric, dtype=np.float64)
y = np.ascontiguousarray(y_metric, dtype=np.float64)
z = np.ascontiguousarray(values, dtype=np.float64)

result = Map(
    resolution=10.0,
)(x, y, z)

result.method                  # "idw" or "kriging"
result.x, result.y, result.z   # compact points within the boundary

X, Y, Z = result.to_raster()   # (ny, nx) arrays, NaN outside the boundary
```

Equivalent functional shortcut:

```python
import apbase

result = apbase.create_map(x, y, z, resolution=10.0)
```

## Manual control

Use the low-level classes when you need to inspect intermediate steps
or force a specific method.

```python
from apbase.cross_validate import cross_validate
from apbase.filtering import SpatialFilter
from apbase.grid import Grid
from apbase.idw import IDW
from apbase.kriging import Kriging

spatial_filter = SpatialFilter(filter_level="light").fit(x, y, z)
x_filtered, y_filtered, z_filtered = spatial_filter.filter()

targets = Grid(resolution=10.0).generate(x_filtered, y_filtered)
cv = cross_validate(x_filtered, y_filtered, z_filtered)

kriging = Kriging().fit(
    x_filtered, y_filtered, z_filtered
)
z_kriging = kriging.interpolate(targets)

idw = IDW().fit(
    x_filtered, y_filtered, z_filtered
)
z_idw = idw.interpolate(targets)
```

## Data requirements

- `x`, `y`, and `z` must be one-dimensional arrays.
- For metric distance, use projected coordinates, such as UTM.
- lon/lat coordinates must be converted before interpolating.
- `resolution`, `radius`, and `bounds` must use the same unit as `x` and `y`.
- For best performance, use contiguous `float64` arrays.

## Global configuration

The number of OpenMP threads is a process-level setting. Set it before
creating instances:

```python
import apbase

apbase.config["n_threads"] = 4
```

## APBase ecosystem

APBase's vision goes beyond this package: to be the numerical foundation
that AI Agents and MCP servers for precision agriculture query for data
from farm machinery, sensors, soil, climate, operations, and remote sensing.

This repository is the high-performance geospatial core of that ecosystem.
Climate pipelines, satellite imagery, platform APIs, Agents, and MCP
servers should consume this package as a dependency, living in separate
modules or services.

## License

Apache License 2.0. See [LICENSE](LICENSE).
