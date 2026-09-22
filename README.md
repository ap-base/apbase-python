# APbase

[![Version](https://img.shields.io/badge/version-0.1.1-informational.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![Kernel: Fortran](https://img.shields.io/badge/kernel-Fortran-734f96.svg)](#performance)
[![Parallel: OpenMP](https://img.shields.io/badge/parallel-OpenMP-orange.svg)](#performance)
[![CI](https://img.shields.io/github/actions/workflow/status/ap-base/apbase-python/ci.yml?branch=main&label=ci)](https://github.com/ap-base/apbase-python/actions/workflows/ci.yml)

<p align="center">
  <img src="assets/main_en.png" alt="APbase Python" width="660"/>
</p>

APbase is an intelligent mapping pipeline for irregular field data, including
yield monitor, soil samples, and sensor grids. It automates the map workflow:
filtering source data, handling geographic coordinates when needed, building
the target grid, comparing interpolation methods, and selecting a ready-to-use
map without requiring users to choose a mathematical model by hand.

Full documentation: https://apbase.io

## Install

```bash
pip install apbase
```

Development version:

```bash
pip install git+https://github.com/ap-base/apbase-python.git
```

## Quick Use

```python
import numpy as np

import apbase
from apbase import Map

apbase.config["n_threads"] = 4

x = np.ascontiguousarray(x_metric, dtype=np.float64)
y = np.ascontiguousarray(y_metric, dtype=np.float64)
z = np.ascontiguousarray(values, dtype=np.float64)

result = Map(resolution=10.0)(x, y, z)

result.method                # "idw" or "kriging"
result.x, result.y, result.z # compact output points

X, Y, Z = result.to_raster() # dense raster, NaN outside the boundary
```

Functional shortcut:

```python
result = apbase.create_map(x, y, z, resolution=10.0)
```

## Main API

- `apbase.Map`: automated map pipeline.
- `apbase.create_map`: one-shot map creation.
- `apbase.config`: runtime configuration, including OpenMP thread count.

Advanced modules are available under `apbase.filtering`, `apbase.grid`,
`apbase.variogram`, `apbase.idw`, `apbase.kriging`, `apbase.cokriging`, and
`apbase.cross_validate`.

## Data Requirements

- `x`, `y`, and `z` must be one-dimensional arrays.
- Projected metric coordinates, such as UTM, are used directly.
- Geographic lon/lat input is converted internally before distance operations;
  distance settings such as `resolution` are meters.
- For best performance, use contiguous `float64` arrays.

## Performance

- Native Fortran kernels exposed through F2PY/NumPy.
- OpenMP parallelism for critical numerical work.
- Local neighborhoods and neighbor limits instead of full distance matrices.
- Compact output by default; dense rasters are materialized only with
  `result.to_raster()`.

Set threads before creating maps:

```python
import apbase

apbase.config["n_threads"] = 4
```

## Links

- Documentation: https://apbase.io
- Source: https://github.com/ap-base/apbase-python
- Issues: https://github.com/ap-base/apbase-python/issues

## License

Apache License 2.0. See [LICENSE](LICENSE).
