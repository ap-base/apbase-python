# Changelog

All notable changes to the `apbase` package are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-21

### Added

- Cokriging section in the README and `apbase.cross_variogram` in the module
  list; `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`
  and issue templates.
- `test` extra (`pip install "apbase[test]"`) with the test dependencies
  (pytest, geopandas).
- Package keywords and classifiers (Python versions, operating systems,
  topics).
- `CoKriging`, `co_kriging` and `cross_validate_cokriging` options
  `max_colocation_dist` (pairing distance for the cross-covariances of
  heterotopic secondaries) and `max_secondary_neighbors` (secondary neighbor
  limit of ICM and LMC; `None` keeps `max_neighbors`).

### Fixed

- ICM and LMC cokriging estimated no cross-covariance for a secondary that
  does not share the primary's sample locations (for example, a raster): the
  fit silently set it to zero, and dense secondaries then left the block system
  singular (NaN estimates) or unstable. The new `max_colocation_dist` option
  pairs each primary sample with its nearest secondary point within that
  distance; without it, every method raises a `ValueError` naming the
  secondary instead of fitting a zero cross-covariance.
- With the gaussian model, ICM, LMC and collocated cokriging (and their
  cross-validation) give each variable a nugget of at least 0.01% of its sill.
  A gaussian model without nugget over close sources produced exploding
  estimates; models with a larger nugget, and the other models, are unchanged.

### Changed

- **Behavior change:** ICM and LMC cokriging (and `cross_validate_cokriging`)
  raise a `ValueError` for a secondary without samples paired with the primary,
  where they used to fit a zero cross-covariance; pass `max_colocation_dist` for
  heterotopic secondaries (see Fixed).
- The README states that installing from git works only on Linux x86_64 with
  Python 3.14; other platforms install the wheels from PyPI.

## [0.1.3] - 2026-10-17

### Changed

- Results no longer depend on the unit of the values: variogram fits (now
  Cressie-weighted) and the kriging and cokriging systems are scale-invariant.
- Covariate screening tolerates a 10% excess over the Cauchy-Schwarz bound of
  the cross-variogram.
- ICM cokriging projects its coregionalization matrices onto the
  positive-semidefinite cone; LMC reports projection and allocation failures
  as errors instead of returning an invalid fit.
- `cross_validate_cokriging` leaves out of `by_method` and `model_params` a
  method whose model cannot be fitted, instead of failing the call; the first
  fit error is raised only when no method can be fitted.

### Fixed

- Collocated cokriging raises a `ValueError` naming the secondary when it has
  too few samples at the primary locations to estimate its correlation, instead
  of producing NaN estimates.
- The native loader no longer returns the compiled extension instead of its
  facade after the first import.

## [0.1.2] - 2026-10-15

### Changed

- The default search radius (no `radius` given) is the variogram `range` below
  10 000 points and `range / 3` from there on (was always `range / 3`).
- When fewer than `min_neighbors` sources lie within the search radius, IDW,
  kriging, cokriging and cross-validation use the `max_neighbors` nearest, so
  every target inside the boundary gets a value.

## [0.1.1] - 2026-09-21

### Added

- First release on PyPI, with wheels for Linux (x86_64), macOS (arm64 and
  x86_64) and Windows (x86_64), for CPython 3.12, 3.13 and 3.14.

### Changed

- The top-level namespace exposes only `Map`, `create_map` and `config`; the
  geostatistical building blocks are imported from their submodules
  (`apbase.idw`, `apbase.kriging`, `apbase.cokriging`, ...).
- Shorter PyPI README with links to the documentation at https://apbase.io;
  package metadata links the documentation, source and issues.
- Expanded docstrings and input validation across the public API.

## [0.1.0] - 2026-09-09

### Added

- First release, on GitHub: `Map` and `create_map`, an automated map pipeline that filters
  the source data, converts geographic coordinates when needed, builds the
  target grid, fits a shared variogram, cross-validates IDW against ordinary
  kriging and keeps the better map.
- Modules `filtering`, `grid`, `variogram`, `idw`, `kriging`, `cokriging`
  (collocated, ICM and LMC), `cross_variogram` and `cross_validate`, backed by
  native Fortran kernels with OpenMP.

[Unreleased]: https://github.com/ap-base/apbase-python/compare/0.2.0...HEAD
[0.2.0]: https://github.com/ap-base/apbase-python/compare/0.1.3...0.2.0
[0.1.3]: https://github.com/ap-base/apbase-python/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/ap-base/apbase-python/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/ap-base/apbase-python/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/ap-base/apbase-python/releases/tag/0.1.0
