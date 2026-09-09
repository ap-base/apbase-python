# Reference datasets for cokriging validation

Real-world fixtures used by `tests/test_cokriging_reference_datasets.py` to
validate `apbase.cokriging` against published geostatistics benchmarks, in
addition to the synthetic NumPy-reference tests in `tests/test_cokriging.py`.

## jura_prediction.csv / jura_validation.csv

Goovaerts (1997), *Geostatistics for Natural Resources Evaluation*, Oxford
University Press — the canonical cokriging benchmark: 7 heavy metals (Cd, Co,
Cr, Cu, Ni, Pb, Zn) in the topsoil of the Swiss Jura, split into 259 training
points (`prediction.dat`) and 100 held-out points (`validation.dat`). `Xloc`/
`Yloc` are local grid coordinates in km.

- Source: `prediction.dat` / `validation.dat` objects inside
  `https://raw.githubusercontent.com/cran/gstat/master/data/jura.rda`
  (CRAN mirror of the `gstat` R package, GPL (>= 2)).
- Regenerate: `rdata.conversion.convert(rdata.parser.parse_file("jura.rda"))["prediction.dat"|"validation.dat"]`
  with the `rdata` PyPI package (pure Python, no R required), then
  `DataFrame.to_csv(..., index=False)`.

## meuse.csv

Classic Meuse river dataset (Burrough & McDonnell, via the `sp`/`gstat` R
packages): 155 points, heavy metal concentrations plus continuous covariates
`elev` (elevation) and `dist` (distance to the Meuse river). Coordinates
`x`/`y` are already in the Dutch RD projected grid (meters).

- Source: `meuse` object inside
  `https://raw.githubusercontent.com/cran/sp/master/data/meuse.rda`
  (CRAN mirror of the `sp` R package, GPL (>= 2)).
- Regenerate: same `rdata` procedure as above.

## walker_sample.csv / walker_exhaustive.csv

Walker Lake dataset from Isaaks & Srivastava (1989), *Applied Geostatistics*,
Oxford University Press — 470 sparse samples of `v`/`u` (`u` missing at 195
locations, encoded as empty) plus a 78,000-point exhaustive grid with the true
`u`/`v` values, used to check cokriging accuracy (predict `v` from `u`)
against known ground truth rather than only a held-out subset.

- Source: `opengeostat/pygslib` (MIT License) —
  `pygslib/data/WalkerLake/walker.dat` (GSLIB text format, header stripped)
  and `pygslib/data/WalkerLake/Exhaustive_set.csv`. That repo's own readme
  traces the data to the ai-geostats/52North `WalkerLake_01.zip` teaching
  dataset.
- Regenerate: `walker.dat` parses with
  `pandas.read_csv(sep=r"\s+", skiprows=8, names=["id","x","y","v","u","t"])`;
  values `>= 1e30` are the GSLIB missing-data sentinel and are mapped to NA.
  `Exhaustive_set.csv` is plain CSV, only lower-cased here.

## Notes

These CSVs are prepared once, offline, and committed as static fixtures so
`pytest` runs deterministically without a network dependency. `rdata` is only
a one-off conversion tool (see above) — it is **not** a runtime or test
dependency of `apbase`.
