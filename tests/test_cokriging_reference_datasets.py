"""Cokriging validation against published, real-world geostatistics benchmarks.

Complements the synthetic NumPy-reference tests in test_cokriging.py, which
only validate the linear-algebra solver against hand-derived reference
formulas. These tests exercise the same public API against real, noisy,
non-Gaussian data with known geostatistics literature results -- see
tests/data/README.md for dataset provenance.

RMSE assertions here are regression freezes (np.isclose against an observed
value, not a hardcoded "cokriging must win"): the well-documented cokriging
"screening effect" (Goovaerts 1997) means collocated/ICM cokriging do not
reliably beat plain ordinary kriging when the secondary is only sampled at
(or near) the same locations as the primary, which is the case for Jura,
Walker Lake and the Meuse samples. The Jura/LMC/Pb combination and the Meuse
raster secondary (sqrt(dist) on the official 40 m grid) below demonstrate a
genuine (if modest) improvement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apbase.cokriging import co_kriging
from apbase.cross_variogram import screen_secondary_variables
from apbase.kriging import ordinary_kriging

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> np.ndarray:
    return np.genfromtxt(_DATA_DIR / name, delimiter=",", names=True, dtype=None, encoding="utf-8")


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))


_JURA_RADIUS = 1.5


def _jura_prediction_and_targets() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    pred = _load("jura_prediction.csv")
    val = _load("jura_validation.csv")
    x0 = pred["Xloc"].astype(np.float64)
    y0 = pred["Yloc"].astype(np.float64)
    z0 = pred["Cd"].astype(np.float64)
    targets = np.column_stack([val["Xloc"], val["Yloc"]]).astype(np.float64)
    truth = val["Cd"].astype(np.float64)
    return pred, x0, y0, z0, targets, truth


def test_jura_lmc_cokriging_regression_for_cadmium() -> None:
    """Regression freeze, not a superiority claim: with Cressie-weighted
    variogram fits, plain kriging of Cd beats LMC cokriging with Pb on the
    validation set. Pb is sampled at the same 259 locations as Cd (no denser
    secondary coverage), so the "screening effect" leaves cokriging little
    to gain over a well-fitted kriging model.
    """
    pred, x0, y0, z0, targets, truth = _jura_prediction_and_targets()
    pb = pred["Pb"].astype(np.float64)

    ok_pred = ordinary_kriging(x0, y0, z0, targets, radius=_JURA_RADIUS, max_neighbors=40, min_neighbors=3)
    ck_pred = co_kriging(
        x0, y0, z0, {"Pb": (x0, y0, pb)}, targets,
        method="lmc", radius=_JURA_RADIUS, max_neighbors=40, min_neighbors=3,
    )

    assert not np.isnan(ok_pred).any()
    assert not np.isnan(ck_pred).any()

    ok_rmse = _rmse(ok_pred, truth)
    ck_rmse = _rmse(ck_pred, truth)

    assert ok_rmse == pytest.approx(0.6828821799917915, rel=0.05)
    assert ck_rmse == pytest.approx(0.716357113623874, rel=0.05)


def test_jura_collocated_and_icm_do_not_beat_ordinary_kriging() -> None:
    """Regression freeze, not a superiority claim: with Ni+Zn as secondaries,
    collocated (MM1, a single fixed rho) is noticeably worse than plain OK
    here, and ICM is roughly on par. A future change to the coregionalization
    fit that meaningfully shifts these numbers should be a deliberate,
    reviewed change -- not a silent regression.
    """
    pred, x0, y0, z0, targets, truth = _jura_prediction_and_targets()
    secondaries = {
        "Ni": (x0, y0, pred["Ni"].astype(np.float64)),
        "Zn": (x0, y0, pred["Zn"].astype(np.float64)),
    }

    collocated_pred = co_kriging(
        x0, y0, z0, secondaries, targets,
        method="collocated", radius=_JURA_RADIUS, max_neighbors=40, min_neighbors=3,
    )
    icm_pred = co_kriging(
        x0, y0, z0, secondaries, targets,
        method="icm", radius=_JURA_RADIUS, max_neighbors=40, min_neighbors=3,
    )

    assert not np.isnan(collocated_pred).any()
    assert not np.isnan(icm_pred).any()
    assert _rmse(collocated_pred, truth) == pytest.approx(0.9773576533793037, rel=0.05)
    assert _rmse(icm_pred, truth) == pytest.approx(0.7435857058009598, rel=0.05)


def test_meuse_screening_and_cokriging_pipeline_reproduces_observed_zinc() -> None:
    """End-to-end smoke test of screen_secondary_variables -> co_kriging.

    Meuse has no published train/validation split, so this checks the
    interpolator-exactness property instead of held-out RMSE: kriging and
    cokriging must reproduce the observed value when predicting back at a
    source point's own location.
    """
    meuse = _load("meuse.csv")
    x = meuse["x"].astype(np.float64)
    y = meuse["y"].astype(np.float64)
    zinc = meuse["zinc"].astype(np.float64)
    elev = meuse["elev"].astype(np.float64)
    dist = meuse["dist"].astype(np.float64)

    screening = screen_secondary_variables(x, y, zinc, {"elev": (x, y, elev), "dist": (x, y, dist)})
    assert set(screening.selected) == {"elev", "dist"}

    targets = np.column_stack([x, y])
    pred = co_kriging(
        x, y, zinc, screening.selected, targets, radius=1000.0, max_neighbors=40, min_neighbors=3
    )

    assert not np.isnan(pred).any()
    assert np.allclose(pred, zinc, rtol=1e-6, atol=1e-6)


_WALKER_RADIUS = 100.0
_WALKER_N_TARGETS = 1500


def _walker_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    walker = _load("walker_sample.csv")
    x0 = walker["x"].astype(np.float64)
    y0 = walker["y"].astype(np.float64)
    v0 = walker["v"].astype(np.float64)
    u_mask = ~np.isnan(walker["u"])
    xu = walker["x"][u_mask].astype(np.float64)
    yu = walker["y"][u_mask].astype(np.float64)
    uu = walker["u"][u_mask].astype(np.float64)
    return x0, y0, v0, xu, yu, uu


def _walker_exhaustive_targets() -> tuple[np.ndarray, np.ndarray]:
    exhaustive = _load("walker_exhaustive.csv")
    rng = np.random.default_rng(0)
    idx = rng.choice(exhaustive["x"].size, size=_WALKER_N_TARGETS, replace=False)
    targets = np.column_stack([exhaustive["x"][idx], exhaustive["y"][idx]]).astype(np.float64)
    truth = exhaustive["v"][idx].astype(np.float64)
    return targets, truth


def test_walker_lake_cokriging_regression_against_exhaustive_grid() -> None:
    """Regression freeze against the known-true exhaustive grid.

    None of the three methods beat plain OK for V~U on the full sample
    (U is only moderately correlated with V, rho~0.55) -- same
    "screening effect" outcome as the Jura Ni/Zn case above, on a
    different dataset. Still a valuable fixture: real, noisy,
    non-Gaussian data checked against actual ground truth.
    """
    x0, y0, v0, xu, yu, uu = _walker_sample()
    targets, truth = _walker_exhaustive_targets()
    secondaries = {"u": (xu, yu, uu)}

    ok_pred = ordinary_kriging(x0, y0, v0, targets, radius=_WALKER_RADIUS, max_neighbors=40, min_neighbors=3)
    assert not np.isnan(ok_pred).any()
    assert _rmse(ok_pred, truth) == pytest.approx(146.2269792078347, rel=0.05)

    expected_rmse = {
        "collocated": 167.54214520983322,
        "icm": 150.61500586082025,
        "lmc": 151.42962473836496,
    }
    for method, expected in expected_rmse.items():
        pred = co_kriging(
            x0, y0, v0, secondaries, targets,
            method=method, radius=_WALKER_RADIUS, max_neighbors=40, min_neighbors=3,
        )
        assert not np.isnan(pred).any()
        assert _rmse(pred, truth) == pytest.approx(expected, rel=0.05)


def test_walker_lake_icm_on_sparse_subsample_matches_ordinary_kriging() -> None:
    """Regression freeze on a sparse real-data subsample.

    A 120-point random subsample of the V sample makes ICM's independent
    per-pair fits slightly inadmissible: the fitted sill matrix implies a
    correlation just above 1. That is within ICM's tolerance, so the matrix is
    projected onto the PSD cone (here onto its rank-1 boundary) instead of
    rejected, and the resulting model predicts as well as plain kriging.
    """
    x0_full, y0_full, v0_full, xu, yu, uu = _walker_sample()
    targets, truth = _walker_exhaustive_targets()

    rng = np.random.default_rng(0)
    sparse_idx = rng.choice(x0_full.size, size=120, replace=False)
    x0, y0, v0 = x0_full[sparse_idx], y0_full[sparse_idx], v0_full[sparse_idx]

    ok_pred = ordinary_kriging(x0, y0, v0, targets, radius=_WALKER_RADIUS, max_neighbors=40, min_neighbors=3)
    icm_pred = co_kriging(
        x0, y0, v0, {"u": (xu, yu, uu)}, targets,
        method="icm", radius=_WALKER_RADIUS, max_neighbors=40, min_neighbors=3,
    )

    assert not np.isnan(icm_pred).any()
    assert _rmse(ok_pred, truth) == pytest.approx(222.0783738847038, rel=0.05)
    assert _rmse(icm_pred, truth) == pytest.approx(225.72232223151371, rel=0.05)


_MEUSE_COLOCATION_DIST = 28.3  # half the diagonal of a 40 m cell
_MEUSE_RADIUS = 400.0


def test_meuse_cokriging_with_raster_secondary_regression() -> None:
    """Heterotopic cokriging with a raster secondary.

    log(zinc) at 110 Meuse samples, sqrt(dist) on the official 40 m grid
    (3103 cells, none at a sample location) paired within half the cell
    diagonal, 45 held-out samples. Regression freeze at rel=0.05; on this
    split every method stays below ordinary kriging.
    """
    meuse = _load("meuse.csv")
    grid = _load("meuse_grid.csv")
    x = meuse["x"].astype(np.float64)
    y = meuse["y"].astype(np.float64)
    z = np.log(meuse["zinc"].astype(np.float64))
    train = np.random.default_rng(0).choice(z.size, size=110, replace=False)
    test = np.setdiff1d(np.arange(z.size), train)
    targets = np.column_stack([x[test], y[test]])
    secondaries = {
        "sqrt_dist": (
            grid["x"].astype(np.float64),
            grid["y"].astype(np.float64),
            np.sqrt(grid["dist"].astype(np.float64)),
        )
    }

    ok_pred = ordinary_kriging(
        x[train], y[train], z[train], targets, radius=_MEUSE_RADIUS, max_neighbors=40, min_neighbors=3
    )
    ok_rmse = _rmse(ok_pred, z[test])
    assert ok_rmse == pytest.approx(0.3215238376477924, rel=0.05)

    for method, max_secondary_neighbors, expected in (
        ("collocated", None, 0.31105033389900477),
        ("icm", None, 0.2948833593039993),
        ("icm", 8, 0.2843351846133008),
        ("lmc", None, 0.29474084618830976),
        ("lmc", 8, 0.28243526180634404),
    ):
        pred = co_kriging(
            x[train], y[train], z[train], secondaries, targets,
            method=method, radius=_MEUSE_RADIUS, max_neighbors=40, min_neighbors=3,
            max_colocation_dist=_MEUSE_COLOCATION_DIST, max_secondary_neighbors=max_secondary_neighbors,
        )
        assert np.isfinite(pred).all(), method
        value = _rmse(pred, z[test])
        assert value == pytest.approx(expected, rel=0.05), (method, max_secondary_neighbors, value)
        assert value < ok_rmse, (method, max_secondary_neighbors)


def test_meuse_icm_with_raster_secondary_covers_the_whole_grid() -> None:
    """Default radius on the full grid: every cell gets a value (dense raster
    neighborhoods used to leave the block system singular) and the fitted
    cross-covariance carries the negative zinc/distance correlation."""
    from apbase.cokriging import CoKriging

    meuse = _load("meuse.csv")
    grid = _load("meuse_grid.csv")
    gx, gy = grid["x"].astype(np.float64), grid["y"].astype(np.float64)
    cokriging = CoKriging(method="icm", max_colocation_dist=_MEUSE_COLOCATION_DIST).fit(
        meuse["x"].astype(np.float64),
        meuse["y"].astype(np.float64),
        np.log(meuse["zinc"].astype(np.float64)),
        {"sqrt_dist": (gx, gy, np.sqrt(grid["dist"].astype(np.float64)))},
    )

    pred = cokriging.interpolate(np.column_stack([gx, gy]))

    assert np.isfinite(pred).all()
    assert cokriging.model_params["sill_matrix"][0, 1] < 0.0


@pytest.mark.parametrize("radius", [100.0, 400.0])
def test_gaussian_cokriging_on_noise_free_raster_stays_within_the_data(radius: float) -> None:
    """A gaussian model without nugget on dense raster pixels used to leave the
    ICM/LMC block systems singular (NaN) or give them exploding weights
    (estimates in the thousands for log(zinc) between 4.7 and 7.5). With the
    kernels' gaussian nugget floor every estimate stays within the data range.
    The model is the one the fit produced before heterotopic pairing existed:
    no cross-covariance and no secondary nugget."""
    from apbase.cokriging._native import run_co_kriging_icm, run_co_kriging_lmc

    meuse = _load("meuse.csv")
    grid = _load("meuse_grid.csv")
    x = meuse["x"].astype(np.float64)
    y = meuse["y"].astype(np.float64)
    z = np.log(meuse["zinc"].astype(np.float64))
    gx, gy = grid["x"].astype(np.float64), grid["y"].astype(np.float64)
    dist = grid["dist"].astype(np.float64)
    secondary_z = z.mean() + (dist - dist.mean()) * (z.std() / dist.std())
    offset = np.array([0, gx.size], dtype=np.int32)
    nugget = np.array([[0.22388, 0.0], [0.0, 0.0]])
    sill = np.array([[0.37619, 0.0], [0.0, 0.63062]])

    icm, icm_status = run_co_kriging_icm(
        x, y, z, gx, gy, 3.0, 966.3674, nugget, sill, gx, gy, secondary_z, offset, radius, 40, 40, 3, 1
    )
    lmc, lmc_status = run_co_kriging_lmc(
        x, y, z, gx, gy, 3.0, np.array([966.3674]), np.stack([nugget, sill]),
        gx, gy, secondary_z, offset, radius, 40, 40, 3, 1,
    )

    for status, pred in ((icm_status, icm), (lmc_status, lmc)):
        assert status == 0
        assert np.isfinite(pred).all()
        assert z.min() - 0.5 < pred.min() and pred.max() < z.max() + 0.5
