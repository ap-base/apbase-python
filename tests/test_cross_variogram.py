from __future__ import annotations

from math import erf, sqrt

import numpy as np

from apbase.cross_variogram import CrossVariogram, screen_secondary_variables
from apbase.cross_variogram._native import run_screen_covariate
from apbase.variogram import Variogram

MODEL_SPHERICAL = 1


def _smooth_field(
    x: np.ndarray, y: np.ndarray, seed: int, n_centers: int = 8, length_scale: float = 25.0
) -> np.ndarray:
    """Sum of randomly-placed Gaussian bumps: smooth and non-periodic.

    Deliberately avoids sin/cos-style periodic fields -- their semivariogram
    oscillates (a "hole effect"), which none of the 3 canonical monotonic
    shapes (spherical/exponential/gaussian) can represent well, and would
    make every fit in this file look artificially poor regardless of
    correctness.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.0, 100.0, (n_centers, 2))
    amps = rng.normal(0.0, 1.0, n_centers)
    field = np.zeros_like(x)
    for amp, (cx, cy) in zip(amps, centers, strict=True):
        field = field + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * length_scale**2))
    return field


def _with_exact_correlation(z0: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """Return a vector with Pearson correlation exactly ``rho`` with ``z0``.

    Standard Gram-Schmidt construction: orthogonalize a random vector against
    (mean-centered) z0, then combine so the result's correlation with z0 is
    exactly the requested rho (up to floating-point precision) -- gives exact
    control over the Fisher z-test input for tests 3 and 6 below, instead of
    hunting for a random seed that happens to produce a target correlation.
    """
    rng = np.random.default_rng(seed)
    centered = z0 - z0.mean()
    raw = rng.normal(0.0, 1.0, z0.size)
    raw_centered = raw - raw.mean()
    orthogonal = raw_centered - (raw_centered @ centered) / (centered @ centered) * centered
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    unit_z0 = centered / np.linalg.norm(centered)
    return rho * unit_z0 + sqrt(1.0 - rho**2) * orthogonal


def _fisher_p_value_reference(rho: float, n_pairs: int) -> float:
    """NumPy/math-only reference for the Fisher z-test two-sided p-value."""
    z_fisher = 0.5 * np.log((1.0 + rho) / (1.0 - rho))
    z_stat = z_fisher * sqrt(n_pairs - 3)
    normal_cdf = 0.5 * (1.0 + erf(abs(z_stat) / sqrt(2.0)))
    return 2.0 * (1.0 - normal_cdf)


def _pearson_reference(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.corrcoef(u, v)[0, 1])


def test_strongly_correlated_candidate_is_selected() -> None:
    x = np.random.default_rng(10).uniform(0.0, 100.0, 150)
    y = np.random.default_rng(11).uniform(0.0, 100.0, 150)
    latent = _smooth_field(x, y, seed=12)
    z0 = latent + np.random.default_rng(13).normal(0.0, 0.1, 150)
    z_corr = 1.5 * latent + np.random.default_rng(14).normal(0.0, 0.15, 150)

    result = screen_secondary_variables(x, y, z0, {"corr": (x, y, z_corr)}, n_lags=10)
    report = result.report["corr"]

    assert report.status == "ok"
    assert report.decision is True
    assert "corr" in result.selected
    assert np.isclose(report.rho, _pearson_reference(z0, z_corr), atol=1e-9)
    assert np.isclose(report.p_value, _fisher_p_value_reference(report.rho, report.n_pairs), atol=1e-6)


def test_pure_noise_candidate_is_rejected_by_p_value() -> None:
    x = np.random.default_rng(20).uniform(0.0, 100.0, 150)
    y = np.random.default_rng(21).uniform(0.0, 100.0, 150)
    z0 = _smooth_field(x, y, seed=22) + np.random.default_rng(23).normal(0.0, 0.1, 150)
    z_noise = np.random.default_rng(24).normal(0.0, 1.0, 150)

    result = screen_secondary_variables(x, y, z0, {"noise": (x, y, z_noise)}, n_lags=10)
    report = result.report["noise"]

    assert report.status == "ok"
    assert report.decision is False
    assert "noise" not in result.selected
    assert np.isclose(report.p_value, _fisher_p_value_reference(report.rho, report.n_pairs), atol=1e-6)
    assert report.p_value > 0.05


def test_high_rho_small_n_is_rejected_by_significance() -> None:
    # n_pairs=6 (>= the native n_pairs>=4 guard) with rho=0.8: high enough
    # correlation to clear min_abs_rho, but too few pairs for the Fisher
    # z-test's 1/sqrt(n-3) standard error to call it significant at alpha=0.05.
    model_00 = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 1.0, 10.0, 0.0])
    model_uu = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 1.0, 10.0, 0.0])
    model_0u = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 0.8, 10.0, 0.0])
    rho = 0.8
    n_pairs = 6

    decision, p_value, admissible_fraction, status = run_screen_covariate(
        model_00, model_uu, model_0u, rho, n_pairs, alpha=0.05, min_abs_rho=0.3, max_violation_fraction=0.1
    )

    reference_p = _fisher_p_value_reference(rho, n_pairs)
    assert status == 0
    assert np.isclose(p_value, reference_p, atol=1e-9)
    assert reference_p > 0.05  # confirms this scenario genuinely tests the significance path
    assert decision is False
    assert admissible_fraction == 1.0  # matched models, isolates the significance rejection


def test_cauchy_schwarz_violation_is_rejected_even_when_significant() -> None:
    # model_0u's partial_sill (5.0) vastly exceeds sqrt(model_00 * model_uu)'s
    # partial_sill (1.0) at every lag -- an inadmissible cross-variogram
    # injected directly, isolated from the fit step.
    model_00 = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 1.0, 10.0, 0.0])
    model_uu = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 1.0, 10.0, 0.0])
    model_0u = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 5.0, 10.0, 0.0])
    rho = 0.99
    n_pairs = 100

    decision, p_value, admissible_fraction, status = run_screen_covariate(
        model_00, model_uu, model_0u, rho, n_pairs, alpha=0.05, min_abs_rho=0.3, max_violation_fraction=0.1
    )

    assert status == 0
    assert p_value < 0.05  # significant on its own -- isolates the admissibility rejection
    assert admissible_fraction < 0.9  # exceeds max_violation_fraction=0.1
    assert decision is False


def test_insufficient_sample_reports_status_without_raising() -> None:
    model_00 = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 1.0, 10.0, 0.0])
    model_uu = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 1.0, 10.0, 0.0])
    model_0u = np.ascontiguousarray([float(MODEL_SPHERICAL), 0.0, 0.8, 10.0, 0.0])

    decision, p_value, admissible_fraction, status = run_screen_covariate(
        model_00, model_uu, model_0u, 0.9, 3, alpha=0.05, min_abs_rho=0.3, max_violation_fraction=0.1
    )

    assert status == 1
    assert decision is False

    # Both sides need enough of their own points for their individual
    # auto-variogram fits to succeed (>= 3 populated lag bins) -- what makes
    # the *pairing* insufficient is that only 3 coordinates are shared
    # between them (exact-match join with max_colocation_dist=None), not a
    # shortage of points on either side alone.
    shared_x = np.ascontiguousarray([0.0, 10.0, 0.0])
    shared_y = np.ascontiguousarray([0.0, 0.0, 10.0])
    primary_only = np.random.default_rng(50).uniform(20.0, 100.0, (27, 2))
    candidate_only = np.random.default_rng(51).uniform(20.0, 100.0, (27, 2))
    x0 = np.concatenate([shared_x, primary_only[:, 0]])
    y0 = np.concatenate([shared_y, primary_only[:, 1]])
    z0 = _smooth_field(x0, y0, seed=52) + np.random.default_rng(53).normal(0.0, 0.05, x0.size)
    xc = np.concatenate([shared_x, candidate_only[:, 0]])
    yc = np.concatenate([shared_y, candidate_only[:, 1]])
    zc = _smooth_field(xc, yc, seed=54) + np.random.default_rng(55).normal(0.0, 0.05, xc.size)

    result = screen_secondary_variables(x0, y0, z0, {"barely_overlapping": (xc, yc, zc)}, n_lags=5)
    report = result.report["barely_overlapping"]
    assert report.n_pairs == 3
    assert report.status == "amostra insuficiente"
    assert report.decision is False
    assert "barely_overlapping" not in result.selected


def test_bonferroni_correction_reduces_false_positive_vs_none() -> None:
    # One borderline candidate whose Fisher p-value sits between alpha/K and
    # alpha (K = len(candidates)): correction="none" tests it at alpha and
    # accepts it, correction="bonferroni" tests it at alpha/K and rejects
    # it -- same underlying data, same call, only `correction` differs.
    n = 40
    x = np.random.default_rng(30).uniform(0.0, 100.0, n)
    y = np.random.default_rng(31).uniform(0.0, 100.0, n)
    latent = _smooth_field(x, y, seed=32, length_scale=40.0)
    z0 = latent + np.random.default_rng(33).normal(0.0, 0.05, n)
    z_borderline = _with_exact_correlation(z0, rho=0.35, seed=34)

    candidates = {"borderline": (x, y, z_borderline)}
    for i in range(9):
        candidates[f"filler_noise_{i}"] = (x, y, np.random.default_rng(100 + i).normal(0.0, 1.0, n))

    reference_p = _fisher_p_value_reference(0.35, n)
    # sanity: the scenario actually straddles both thresholds
    assert 0.05 / len(candidates) < reference_p < 0.05

    result_none = screen_secondary_variables(x, y, z0, candidates, correction="none", n_lags=8)
    result_bonferroni = screen_secondary_variables(x, y, z0, candidates, correction="bonferroni", n_lags=8)

    assert result_none.report["borderline"].decision is True
    assert "borderline" in result_none.selected
    assert result_bonferroni.report["borderline"].decision is False
    assert "borderline" not in result_bonferroni.selected


def test_cross_variogram_matches_reference_correlation_and_reduces_to_auto_variogram_when_equal() -> None:
    # Fitting a variable against itself through the (unconstrained) cross
    # machinery must reproduce the same *shape* of semivariance as the
    # (non-negativity-clamped) auto-variogram fitter: sanity check that the
    # parallel-bracket cross estimator reduces correctly to the standard
    # auto-semivariogram estimator when u=v (guards the sign convention).
    x = np.random.default_rng(40).uniform(0.0, 100.0, 150)
    y = np.random.default_rng(41).uniform(0.0, 100.0, 150)
    z0 = _smooth_field(x, y, seed=42) + np.random.default_rng(43).normal(0.0, 0.05, 150)

    auto = Variogram(n_lags=10).fit(x, y, z0)
    cross_self = CrossVariogram(n_lags=10).fit(x, y, z0, x, y, z0)

    h = np.linspace(1.0, 60.0, 30)
    assert np.allclose(auto.evaluate(h), cross_self.evaluate(h), rtol=0.05, atol=0.05)
    assert np.isclose(cross_self.rho, 1.0, atol=1e-9)


def test_cross_variogram_survives_repeated_fits_with_many_duplicate_points() -> None:
    """Regression test for a real crash (segfault / OOB array write) found
    via a large real dataset in cross_semivariance.f90's rejection-sampling
    collector (collect_random_pair, used when there are more candidate
    pairs than max_pairs and max_distance<=0): a duplicate-point pair (or a
    non-finite gamma) advanced count_samples without writing
    sample_distance/sample_gamma at that slot, leaving a hole that later
    got read as uninitialized garbage by accumulate_sample -- occasionally
    overflowing int() during bin_local's computation and writing
    local_pair_count at a wildly out-of-bounds index. Needs enough points
    that C(n,2) exceeds max_pairs (so the rejection-sampling branch, not
    the "use every pair" branch, actually runs) and enough exact duplicates
    that collect_random_pair's degenerate (dist<=0) branch is hit
    repeatedly. Only manifested on a *second* native call in the same
    process (stale heap reuse made the garbage look like a plausible,
    in-range float on the first call) -- hence the repeated-call loop, not
    a single fit. Reproducing the exact segfault reliably needs the
    specific heap layout a large real dataset happened to produce
    (confirmed directly via ``gfortran -fcheck=all``, which caught it at
    the exact out-of-bounds line); this test cannot force that
    non-determinism from a plain build, but it exercises the previously
    -buggy code path (many degenerate draws across many repeated fits) as
    a standing smoke check.
    """
    rng = np.random.default_rng(77)
    n_unique = 500
    xu = rng.uniform(0.0, 100.0, n_unique)
    yu = rng.uniform(0.0, 100.0, n_unique)
    zu = _smooth_field(xu, yu, seed=78) + rng.normal(0.0, 0.1, n_unique)
    dup_count = 400
    dup_idx = rng.integers(0, n_unique, dup_count)
    x = np.concatenate([xu, xu[dup_idx]])
    y = np.concatenate([yu, yu[dup_idx]])
    z = np.concatenate([zu, zu[dup_idx] + rng.normal(0.0, 0.01, dup_count)])

    for _ in range(30):
        cross = CrossVariogram(n_lags=20, max_pairs=50_000, max_distance=0.0).fit(x, y, z, x, y, z + 1.0)
        assert cross.n_pairs > 0
        assert np.isfinite(cross.rho)


def test_cross_variogram_rejects_size_mismatch() -> None:
    x = np.ascontiguousarray([0.0, 1.0, 2.0])
    y = np.ascontiguousarray([0.0, 1.0, 2.0])
    z = np.ascontiguousarray([1.0, 2.0, 3.0])
    z_short = np.ascontiguousarray([1.0, 2.0])

    try:
        CrossVariogram().fit(x, y, z, x, y, z_short)
    except ValueError as exc:
        assert "size" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched xb/yb/zb sizes")


if __name__ == "__main__":
    test_strongly_correlated_candidate_is_selected()
    test_pure_noise_candidate_is_rejected_by_p_value()
    test_high_rho_small_n_is_rejected_by_significance()
    test_cauchy_schwarz_violation_is_rejected_even_when_significant()
    test_insufficient_sample_reports_status_without_raising()
    test_bonferroni_correction_reduces_false_positive_vs_none()
    test_cross_variogram_matches_reference_correlation_and_reduces_to_auto_variogram_when_equal()
    test_cross_variogram_survives_repeated_fits_with_many_duplicate_points()
    test_cross_variogram_rejects_size_mismatch()
