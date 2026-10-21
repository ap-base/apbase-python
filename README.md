# APbase

[![Version](https://img.shields.io/badge/version-0.2.0-informational.svg)](pyproject.toml)
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

PyPI has wheels for Linux (x86_64), macOS (arm64 and x86_64) and Windows
(x86_64), for Python 3.12 to 3.14.

Development version, Linux x86_64 with Python 3.14 only (the repository carries
the compiled native module for that platform alone):

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

## Cokriging

When secondary variables correlated with the target one are available, such as
elevation, distance to a river or a sensor reading at the soil samples,
`apbase.cokriging` uses them in the estimate. Three methods are available:
collocated (Markov-1, the default), ICM (intrinsic coregionalization model) and
LMC (linear model of coregionalization). `apbase.cross_variogram` screens the
candidates first: a Fisher z-test for the correlation and a Cauchy-Schwarz check
that the cross-variogram is admissible.

```python
from apbase.cokriging import co_kriging
from apbase.cross_variogram import screen_secondary_variables

# candidates: {"name": (x, y, values), ...} of possible secondary variables.
screening = screen_secondary_variables(x, y, z, candidates)

estimates = co_kriging(x, y, z, screening.selected, targets, method="collocated")
```

A secondary that does not share the primary's sample locations, such as a
raster (NDVI, elevation, apparent electrical conductivity), needs
`max_colocation_dist`: each primary sample is paired with its nearest secondary
point within that distance (half the cell diagonal for a grid) to estimate the
cross-covariances. For dense secondaries, `max_secondary_neighbors` (ICM and
LMC) keeps only the nearest ones; 8 to 16 is usually enough. Cokriging assumes
a roughly linear relation between the variables, so transform a secondary that
does not have one (for example, the square root of a distance).

```python
estimates = co_kriging(
    x, y, z, {"ndvi": (raster_x, raster_y, ndvi)}, targets,
    method="icm", max_colocation_dist=28.3, max_secondary_neighbors=8,  # 40 m raster
)
```

`Map` chooses between IDW and ordinary kriging only; cokriging is called
directly. The test suite runs it on three published geostatistics datasets,
Jura, Meuse and Walker Lake, including the Meuse 40 m grid as a raster
secondary (see `tests/data/README.md`).

## Main API

- `apbase.Map`: automated map pipeline.
- `apbase.create_map`: one-shot map creation.
- `apbase.config`: runtime configuration, including OpenMP thread count.

Advanced modules are available under `apbase.filtering`, `apbase.grid`,
`apbase.variogram`, `apbase.idw`, `apbase.kriging`, `apbase.cokriging`,
`apbase.cross_variogram`, and `apbase.cross_validate`.

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

## Contributing

See [CONTRIBUTING.md](https://github.com/ap-base/apbase-python/blob/main/CONTRIBUTING.md). Questions and ideas go to
[Discussions](https://github.com/ap-base/apbase-python/discussions); security
problems follow [SECURITY.md](https://github.com/ap-base/apbase-python/blob/main/SECURITY.md).

## Citation

If you use APbase in your work, please cite it with the metadata in
[CITATION.cff](https://github.com/ap-base/apbase-python/blob/main/CITATION.cff) (the "Cite this repository" button on GitHub).

## Links

- Documentation: https://apbase.io
- Source: https://github.com/ap-base/apbase-python
- Issues: https://github.com/ap-base/apbase-python/issues
- Changelog: https://github.com/ap-base/apbase-python/blob/main/CHANGELOG.md

## License

Apache License 2.0. See [LICENSE](LICENSE).
