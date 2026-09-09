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
(or near) the same locations as the primary, which is the case for every
dataset available here. Only the Jura/LMC/Pb combination below demonstrates
a genuine (if modest) improvement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apbase.cokriging import co_kriging
from apbase.common.exceptions import CoKrigingNonPSDCoregionalizationError
from apbase.cross_variogram import screen_secondary_variables
from apbase.kriging import ordinary_kriging

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> np.ndarray:
    return np.genfromtxt(_DATA_DIR / name, delimiter=",", names=True, dtype=None, encoding="utf-8")


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))


# ---------------------------------------------------------------------------
# Jura (Goovaerts 1997): predict Cd from Ni/Zn/Pb, 259 training points
# (prediction.dat) + 100 held-out points (validation.dat) -- the canonical
# published cokriging benchmark, with its own train/validation split.
# ---------------------------------------------------------------------------

# The default radius (range/3 from the auto-fit primary variogram, ~0.16 km)
# starves most validation targets of the min_neighbors=3 required on this
# dataset's clustered sampling (359 points over ~5x5 km); 1.5 km covers the
# whole extent with room to spare.
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


def test_jura_lmc_cokriging_beats_ordinary_kriging_for_cadmium() -> None:
    """LMC cokriging (Cd ~ Pb) is the one real-data case here where
    cokriging demonstrably beats plain kriging, and only by a few percent:
    Pb is sampled at the same 259 locations as Cd (no denser secondary
    coverage), so the "screening effect" limits the achievable gain.
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

    assert ck_rmse < ok_rmse
    assert ok_rmse == pytest.approx(0.73658451360863, rel=0.05)
    assert ck_rmse == pytest.approx(0.718762064724412, rel=0.05)


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


# ---------------------------------------------------------------------------
# Meuse (sp/gstat): zinc ~ elevation + distance-to-river, 155 points.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Walker Lake (Isaaks & Srivastava 1989): V ~ U, 470/275 points plus a
# 78,000-point exhaustive grid with known true values -- checked against
# actual ground truth rather than another held-out sample.
# ---------------------------------------------------------------------------

_WALKER_RADIUS = 100.0
_WALKER_N_TARGETS = 1500  # subsample of the 78k exhaustive grid, for test speed


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
    assert _rmse(ok_pred, truth) == pytest.approx(146.24500236850068, rel=0.05)

    expected_rmse = {
        "collocated": 168.03058021141942,
        "icm": 151.11274616648427,
        "lmc": 163.93028459231954,
    }
    for method, expected in expected_rmse.items():
        pred = co_kriging(
            x0, y0, v0, secondaries, targets,
            method=method, radius=_WALKER_RADIUS, max_neighbors=40, min_neighbors=3,
        )
        assert not np.isnan(pred).any()
        assert _rmse(pred, truth) == pytest.approx(expected, rel=0.05)


def test_walker_lake_icm_rejects_ill_conditioned_sparse_subsample() -> None:
    """ICM's PSD admissibility guard (icm_fit.f90), exercised with real data.

    ICM fits each variable pair's (nugget, sill) independently via weighted
    least squares, then requires the resulting nv x nv matrices to be
    jointly positive semidefinite (Goovaerts 1997 / Chiles & Delfiner 2012
    admissibility) before accepting the model -- it rejects an inadmissible
    fit rather than correcting or silently using it. The full 470-point V
    sample fits fine (see the test above); a 120-point random subsample of
    it is ill-conditioned enough that the independent per-pair fits are no
    longer jointly PSD, and the guard correctly raises instead of returning
    an unstable model.
    """
    x0_full, y0_full, v0_full, xu, yu, uu = _walker_sample()
    targets, _truth = _walker_exhaustive_targets()

    rng = np.random.default_rng(0)
    sparse_idx = rng.choice(x0_full.size, size=120, replace=False)
    x0, y0, v0 = x0_full[sparse_idx], y0_full[sparse_idx], v0_full[sparse_idx]

    with pytest.raises(CoKrigingNonPSDCoregionalizationError):
        co_kriging(
            x0, y0, v0, {"u": (xu, yu, uu)}, targets,
            method="icm", radius=_WALKER_RADIUS, max_neighbors=40, min_neighbors=3,
        )
