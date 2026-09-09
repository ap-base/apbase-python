from __future__ import annotations

import numpy as np

from apbase.cokriging import CoKriging, co_kriging

MODEL_SPHERICAL = 1
MODEL_EXPONENTIAL = 2
MODEL_GAUSSIAN = 3


def _variogram_shape(h: np.ndarray, model_id: int, model_range: float) -> np.ndarray:
    # ICM's joint fit picks whichever of the 3 shapes best fits the data, so
    # the reference solver (used against fitted, not hand-specified, models
    # in the ICM tests below) needs all 3, not just spherical/gaussian.
    ratio = h / model_range
    if model_id == MODEL_SPHERICAL:
        clipped = np.clip(ratio, 0.0, 1.0)
        return np.where(ratio >= 1.0, 1.0, 1.5 * clipped - 0.5 * clipped**3)
    if model_id == MODEL_EXPONENTIAL:
        return 1.0 - np.exp(-3.0 * ratio)
    if model_id == MODEL_GAUSSIAN:
        return 1.0 - np.exp(-3.0 * ratio**2)
    raise ValueError(f"unsupported model_id: {model_id}")


def _covariance(
    h: np.ndarray, model_id: int, nugget: float, partial_sill: float, model_range: float
) -> np.ndarray:
    sill = nugget + partial_sill
    gamma = nugget + partial_sill * _variogram_shape(h, model_id, model_range)
    return np.where(h <= 0.0, sill, sill - gamma)


def _standardize(z0: np.ndarray, zu: np.ndarray) -> np.ndarray:
    """Same formula as CoKriging.fit(): rescale zu to z0's mean/variance."""
    z0_mean, z0_std = float(z0.mean()), float(z0.std())
    zu_mean, zu_std = float(zu.mean()), float(zu.std())
    return z0_mean + (zu - zu_mean) * (z0_std / zu_std)


def _reference_collocated_cokriging(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    secondaries_xyz: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rhos: list[float],
    targets: np.ndarray,
    nugget: float,
    partial_sill: float,
    model_range: float,
    model_id: int = MODEL_SPHERICAL,
    search_radius: float | None = None,
    max_neighbors: int | None = None,
) -> np.ndarray:
    """Independent NumPy solver for standardized collocated cokriging (MM1).

    Mirrors collocated_cokriging_local.f90's system exactly: C_0u(h) =
    rho_u * C00(h) and C_uv(x0,x0) = rho_u*rho_v*C00(0) for standardized
    secondaries (COKRIGING_PLAN.md section 3/4c specialized to
    standardized inputs -- see the kernel's own derivation comment).

    ``search_radius``/``max_neighbors`` truncate the primary neighbor set
    per target exactly like the kernel's own radius/max_neighbors search
    (nearest ``max_neighbors`` within ``search_radius``, closest first);
    omit both (the default) to use every primary point unconditionally, as
    the K1/K2 tests below do.
    """
    n = x0.size
    k = len(secondaries_xyz)
    sill = nugget + partial_sill
    standardized = [_standardize(z0, zu) for (_, _, zu) in secondaries_xyz]

    out = np.empty(targets.shape[0], dtype=np.float64)
    for row, (tx, ty) in enumerate(targets):
        d2 = (x0 - tx) ** 2 + (y0 - ty) ** 2
        if search_radius is not None:
            within = np.flatnonzero(d2 <= search_radius * search_radius)
            order = within[np.argsort(d2[within])]
            if max_neighbors is not None:
                order = order[:max_neighbors]
        else:
            order = np.arange(n)
        xs, ys, zs = x0[order], y0[order], z0[order]
        nn = xs.size

        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        h_pp = np.sqrt(dx**2 + dy**2)
        c_pp = _covariance(h_pp, model_id, nugget, partial_sill, model_range)

        nsys = nn + k + 1
        a = np.zeros((nsys, nsys))
        b = np.zeros(nsys)

        a[:nn, :nn] = c_pp
        h0 = np.sqrt((xs - tx) ** 2 + (ys - ty) ** 2)
        c0 = _covariance(h0, model_id, nugget, partial_sill, model_range)
        b[:nn] = c0

        collocated_values = np.empty(k)
        for u in range(k):
            xu, yu, _ = secondaries_xyz[u]
            nearest = int(np.argmin((xu - tx) ** 2 + (yu - ty) ** 2))
            collocated_values[u] = standardized[u][nearest]

            a[:nn, nn + u] = rhos[u] * c0
            a[nn + u, :nn] = rhos[u] * c0
            a[nn + u, nn + u] = sill
            b[nn + u] = rhos[u] * sill
            for v in range(u + 1, k):
                a[nn + u, nn + v] = rhos[u] * rhos[v] * sill
                a[nn + v, nn + u] = rhos[u] * rhos[v] * sill

        a[:nn, nsys - 1] = 1.0
        a[nsys - 1, :nn] = 1.0
        a[nn : nn + k, nsys - 1] = 1.0
        a[nsys - 1, nn : nn + k] = 1.0
        a[nsys - 1, nsys - 1] = 0.0
        b[nsys - 1] = 1.0

        weights = np.linalg.solve(a, b)
        out[row] = float(np.dot(weights[:nn], zs) + np.dot(weights[nn : nn + k], collocated_values))
    return out


def _model_values(nugget: float, partial_sill: float, model_range: float, rhos: list[float]) -> np.ndarray:
    return np.ascontiguousarray([float(MODEL_SPHERICAL), nugget, partial_sill, model_range, 0.0, *rhos])


def test_collocated_cokriging_k1_matches_reference() -> None:
    x0 = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 4.0, 7.0])
    y0 = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 6.0, 2.0])
    z0 = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.5, 6.0])
    xu = np.ascontiguousarray([1.0, 9.0, 1.0, 9.0, 5.0, 6.0])
    yu = np.ascontiguousarray([1.0, 1.0, 9.0, 9.0, 5.0, 3.0])
    zu = np.ascontiguousarray([10.0, 14.0, 16.0, 20.0, 13.0, 12.0])
    grid = np.ascontiguousarray([[5.0, 5.0], [2.0, 8.0], [9.0, 1.0]])
    nugget, partial_sill, model_range, rho = 0.5, 4.0, 30.0, 0.9

    result = (
        CoKriging(
            method="collocated",
            model_values=_model_values(nugget, partial_sill, model_range, [rho]),
            radius=model_range,
            max_neighbors=10,
            min_neighbors=1,
        )
        .fit(x0, y0, z0, {"secondary": (xu, yu, zu)})
        .interpolate(grid)
    )
    expected = _reference_collocated_cokriging(
        x0, y0, z0, [(xu, yu, zu)], [rho], grid, nugget, partial_sill, model_range
    )

    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_collocated_cokriging_k2_matches_reference_with_correlated_secondaries() -> None:
    # K=2, with the two secondaries also correlated with each other (not
    # just each with the primary) -- exercises the secondary<->secondary
    # block of the system, per the plan's own testing note.
    rng = np.random.default_rng(7)
    n = 25
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    z0 = rng.uniform(1.0, 10.0, n)

    xu1 = rng.uniform(0.0, 100.0, n)
    yu1 = rng.uniform(0.0, 100.0, n)
    zu1 = rng.uniform(1.0, 10.0, n)
    xu2 = rng.uniform(0.0, 100.0, n)
    yu2 = rng.uniform(0.0, 100.0, n)
    # correlation with zu1 comes from shared rho_uv in the model, not the raw data
    zu2 = rng.uniform(1.0, 10.0, n)

    grid = np.ascontiguousarray(rng.uniform(10.0, 90.0, (8, 2)))
    nugget, partial_sill, model_range = 0.5, 4.0, 60.0
    rho1, rho2 = 0.8, 0.6
    # search_radius >> model_range: guarantees every one of the n=25 primary
    # points (and the nearest secondary point) is found for every target, so
    # the reference solver's "always include everyone" system matches the
    # kernel's radius-bounded search exactly (unlike model_range, which is
    # deliberately smaller and only shapes the covariance itself).
    search_radius = 1000.0

    result = (
        CoKriging(
            method="collocated",
            model_values=_model_values(nugget, partial_sill, model_range, [rho1, rho2]),
            radius=search_radius,
            max_neighbors=n,
            min_neighbors=1,
        )
        .fit(x0, y0, z0, [(xu1, yu1, zu1), (xu2, yu2, zu2)])
        .interpolate(grid)
    )
    expected = _reference_collocated_cokriging(
        x0, y0, z0, [(xu1, yu1, zu1), (xu2, yu2, zu2)], [rho1, rho2], grid, nugget, partial_sill, model_range
    )

    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_collocated_cokriging_truncates_primary_neighbors_correctly() -> None:
    # Regression test for a real bug found via cross_validate_cokriging's
    # manual-LOO cross-check (see COKRIGING_IMPLEMENTATION_CHECKLIST.md,
    # Milestone 5): collocated_cokriging_local.f90's primary neighbor push
    # tracked "worst kept distance" via a fresh linear scan on every insert,
    # but never wrote that value back into neighbor_d2_local(1) -- the slot
    # find_primary_neighbors' ring-pruning check assumes holds it (a
    # max-heap root). Once more candidates existed within radius than
    # max_neighbors, pruning compared against a stale/arbitrary slot-1 value
    # instead of the true worst, silently dropping true nearest neighbors.
    # K1/K2 above never catch this: both deliberately size max_neighbors >=
    # every candidate (see the K2 fixture's own comment), so truncation
    # never actually triggers. Here max_neighbors=8 with ~25-35 candidates
    # within radius per target forces it every time.
    rng = np.random.default_rng(11)
    n = 50
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    z0 = rng.uniform(1.0, 10.0, n)
    xu = rng.uniform(0.0, 100.0, n)
    yu = rng.uniform(0.0, 100.0, n)
    zu = rng.uniform(1.0, 10.0, n)
    grid = np.ascontiguousarray(rng.uniform(10.0, 90.0, (10, 2)))
    nugget, partial_sill, model_range, rho = 0.5, 4.0, 30.0, 0.7
    radius, max_neighbors = 40.0, 8

    result = (
        CoKriging(
            method="collocated",
            model_values=_model_values(nugget, partial_sill, model_range, [rho]),
            radius=radius,
            max_neighbors=max_neighbors,
            min_neighbors=1,
        )
        .fit(x0, y0, z0, {"secondary": (xu, yu, zu)})
        .interpolate(grid)
    )
    expected = _reference_collocated_cokriging(
        x0, y0, z0, [(xu, yu, zu)], [rho], grid, nugget, partial_sill, model_range,
        search_radius=radius, max_neighbors=max_neighbors,
    )

    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_collocated_cokriging_fitted_model_beats_plain_kriging_when_primary_sparse() -> None:
    # Cokriging's actual value proposition (COKRIGING_PLAN.md's own framing):
    # a SPARSE primary alongside a DENSE, correlated secondary -- not a dense
    # primary that plain kriging can already resolve well on its own (that
    # scenario showed no reliable improvement when tried, matching the
    # plan's own "screening effect" discussion: with enough primary
    # neighbors already available, one extra covariate barely moves RMSE).
    from apbase.kriging import Kriging

    rng = np.random.default_rng(11)
    n_dense = 150
    x = rng.uniform(0.0, 100.0, n_dense)
    y = rng.uniform(0.0, 100.0, n_dense)
    centers = rng.uniform(0.0, 100.0, (8, 2))
    amps = rng.normal(0.0, 1.0, 8)
    latent = np.zeros(n_dense)
    for amp, (cx, cy) in zip(amps, centers, strict=True):
        latent += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 25.0**2))
    z_secondary = 1.5 * latent + rng.normal(0.0, 0.05, n_dense)  # dense, low-noise, covers every target

    sparse_idx = rng.choice(n_dense, size=20, replace=False)
    x0_sparse, y0_sparse = x[sparse_idx], y[sparse_idx]
    z0_sparse = latent[sparse_idx] + rng.normal(0.0, 0.05, 20)

    remaining = np.setdiff1d(np.arange(n_dense), sparse_idx)
    holdout = rng.choice(remaining, size=20, replace=False)
    targets = np.column_stack([x[holdout], y[holdout]])
    actual = latent[holdout]  # noiseless signal: what both methods are trying to recover

    cokrig_est = (
        CoKriging(method="collocated", radius=50.0, max_neighbors=15, min_neighbors=2)
        .fit(x0_sparse, y0_sparse, z0_sparse, {"secondary": (x, y, z_secondary)})
        .interpolate(targets)
    )
    kriging_est = (
        Kriging(radius=50.0, max_neighbors=15, min_neighbors=2)
        .fit(x0_sparse, y0_sparse, z0_sparse)
        .interpolate(targets)
    )
    assert np.all(np.isfinite(cokrig_est)) and np.all(np.isfinite(kriging_est))

    cokrig_rmse = float(np.sqrt(np.mean((cokrig_est - actual) ** 2)))
    kriging_rmse = float(np.sqrt(np.mean((kriging_est - actual) ** 2)))
    assert cokrig_rmse < kriging_rmse


def test_co_kriging_free_function_matches_class() -> None:
    x0 = np.ascontiguousarray([0.0, 10.0, 0.0, 10.0, 4.0, 7.0])
    y0 = np.ascontiguousarray([0.0, 0.0, 10.0, 10.0, 6.0, 2.0])
    z0 = np.ascontiguousarray([5.0, 7.0, 8.0, 10.0, 6.5, 6.0])
    zu = np.ascontiguousarray([10.0, 14.0, 16.0, 20.0, 13.0, 12.0])
    grid = np.ascontiguousarray([[5.0, 5.0], [2.0, 8.0]])
    model_values = _model_values(0.5, 4.0, 30.0, [0.9])

    from_function = co_kriging(
        x0, y0, z0, {"secondary": (x0, y0, zu)}, grid,
        model_values=model_values, radius=30.0, max_neighbors=10, min_neighbors=1,
    )
    from_class = (
        CoKriging(model_values=model_values, radius=30.0, max_neighbors=10, min_neighbors=1)
        .fit(x0, y0, z0, {"secondary": (x0, y0, zu)})
        .interpolate(grid)
    )
    assert np.array_equal(from_function, from_class)


def test_cokriging_lmc_model_values_override_raises_not_implemented() -> None:
    # LMC's fit kernel is implemented; only the model_values override
    # (pre-fitted model bypass) remains unsupported -- collocated's flat-
    # vector format doesn't describe an LMC model (S nested (K+1)x(K+1)
    # matrices), same restriction already in place for ICM.
    try:
        CoKriging(method="lmc", model_values=[1.0, 0.0, 1.0, 10.0, 0.0, 0.5])
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for method='lmc' with model_values")


def _icm_covariance(u: int, v: int, h: float, model_id: int, model_range: float,
                     nugget_matrix: np.ndarray, sill_matrix: np.ndarray) -> float:
    if h <= 1e-6:
        return float(nugget_matrix[u, v] + sill_matrix[u, v])
    shape = _variogram_shape(np.asarray(h), model_id, model_range)
    return float(sill_matrix[u, v] * (1.0 - shape))


def _nearest_within(
    x: np.ndarray, y: np.ndarray, tx: float, ty: float, radius: float, max_neighbors: int
) -> np.ndarray:
    """Indices of up to max_neighbors nearest points within radius, closest first.

    NumPy equivalent of co_kriging_icm_local.f90's per-variable heap-bounded
    ring search: for non-tied random data, "smallest max_neighbors distances
    within radius" is the same set regardless of how it's found.
    """
    d2 = (x - tx) ** 2 + (y - ty) ** 2
    within = np.flatnonzero(d2 <= radius * radius)
    order = within[np.argsort(d2[within])]
    return order[:max_neighbors]


def _reference_icm_cokriging(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    secondaries_std: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_id: int,
    model_range: float,
    nugget_matrix: np.ndarray,
    sill_matrix: np.ndarray,
    targets: np.ndarray,
    radius: float,
    max_neighbors: int,
) -> np.ndarray:
    """Independent NumPy solver for the general ICM block-kriging system.

    Mirrors co_kriging_icm_local.f90: each of the K+1 variables searches its
    OWN neighborhood independently, up to max_neighbors nearest points within
    radius (see _nearest_within) -- not "use everyone", so this stays
    tractable for the larger, more realistic fit datasets ICM's joint fit
    actually needs (a handful of points per pair is too sparse to populate
    3+ lag bins, see test_icm_cokriging_k1_matches_reference's docstring).
    Cross blocks use C_uv(h) = sill_uv*(1-shape(h)) for h>0, nugget_uv+sill_uv
    at h=0 -- the ICM specialization verified against a full-coverage version
    of this same reference during development (matched to 1e-9).
    """
    k = len(secondaries_std)
    xs = [x0] + [xu for xu, _, _ in secondaries_std]
    ys = [y0] + [yu for _, yu, _ in secondaries_std]
    zs = [z0] + [zu for _, _, zu in secondaries_std]

    out = np.empty(targets.shape[0], dtype=np.float64)
    for row, (tx, ty) in enumerate(targets):
        neighbor_idx = [_nearest_within(xs[u], ys[u], tx, ty, radius, max_neighbors) for u in range(k + 1)]
        sizes = [idx.size for idx in neighbor_idx]
        offsets = np.cumsum([0] + sizes)
        total = sum(sizes)
        nsys = total + 1
        a = np.zeros((nsys, nsys))
        b = np.zeros(nsys)

        for u in range(k + 1):
            for i, pi in enumerate(neighbor_idx[u]):
                gi = offsets[u] + i
                for v in range(k + 1):
                    for j, pj in enumerate(neighbor_idx[v]):
                        gj = offsets[v] + j
                        if gj <= gi:
                            continue
                        h = float(np.hypot(xs[u][pi] - xs[v][pj], ys[u][pi] - ys[v][pj]))
                        cov = _icm_covariance(u, v, h, model_id, model_range, nugget_matrix, sill_matrix)
                        a[gi, gj] = cov
                        a[gj, gi] = cov
                a[gi, gi] = _icm_covariance(u, u, 0.0, model_id, model_range, nugget_matrix, sill_matrix)
                h0 = float(np.hypot(xs[u][pi] - tx, ys[u][pi] - ty))
                b[gi] = _icm_covariance(u, 0, h0, model_id, model_range, nugget_matrix, sill_matrix)
                a[gi, nsys - 1] = 1.0
                a[nsys - 1, gi] = 1.0

        b[nsys - 1] = 1.0
        weights = np.linalg.solve(a, b)
        prediction = 0.0
        for u in range(k + 1):
            for i, pi in enumerate(neighbor_idx[u]):
                prediction += weights[offsets[u] + i] * zs[u][pi]
        out[row] = prediction
    return out


def _smooth_field(
    x: np.ndarray, y: np.ndarray, seed: int, n_centers: int = 8, length_scale: float = 25.0
) -> np.ndarray:
    """Sum of randomly-placed Gaussian bumps: smooth, non-periodic, enough
    lag-bin coverage for ICM's joint fit (unlike a handful of hand-picked
    points -- see the module docstring note in test_icm_cokriging_k1_matches_reference).
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.0, 100.0, (n_centers, 2))
    amps = rng.normal(0.0, 1.0, n_centers)
    field = np.zeros_like(x)
    for amp, (cx, cy) in zip(amps, centers, strict=True):
        field = field + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * length_scale**2))
    return field


def test_icm_cokriging_k1_matches_reference() -> None:
    # Isolates the INTERPOLATION kernel from the FIT: whatever model ICM's
    # joint fit happens to produce for this data, the kernel's per-target
    # estimate must match a direct NumPy solve of the same general block
    # system (radius/max_neighbors-truncated per variable) built from that
    # exact (model_id, range, nugget_matrix, sill_matrix). Uses a
    # moderately-sized smooth field, not a handful of hand-picked points --
    # ICM's joint fit needs enough colocated pairs to populate >=3 lag bins
    # per one of the (K+1)(K+2)/2 pairs at the default n_lags=50 profile,
    # which a 6-point fixture (as the collocated tests use, backed by an
    # explicit model_values override ICM doesn't support) cannot supply.
    # seed=6/1006 with this noise level lands on an admissible (PSD) per-pair
    # nugget fit -- see test_icm_cokriging_rejects_non_psd_coregionalization's
    # note: ICM's independent per-pair nugget WLS is inherently noisy when the
    # true cross-nugget is near zero, so this fixture is chosen (not
    # arbitrary) to avoid the degenerate near-zero-diagonal case exercised
    # by the rejection test below.
    rng = np.random.default_rng(6)
    n = 120
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    latent = _smooth_field(x0, y0, seed=1006)
    z0 = latent + rng.normal(0.0, 0.15, n)
    zu = 1.4 * latent + rng.normal(0.0, 0.25, n)
    grid = np.ascontiguousarray(rng.uniform(10.0, 90.0, (6, 2)))
    radius, max_neighbors = 40.0, 12

    ck = CoKriging(method="icm", radius=radius, max_neighbors=max_neighbors, min_neighbors=1)
    ck.fit(x0, y0, z0, {"secondary": (x0, y0, zu)})
    result = ck.interpolate(grid)

    zu_std = _standardize(z0, zu)
    expected = _reference_icm_cokriging(
        x0, y0, z0, [(x0, y0, zu_std)],
        int(round(ck._fit.model_id)), ck._fit.model_range, ck._fit.nugget_matrix, ck._fit.sill_matrix,
        grid, radius, max_neighbors,
    )
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_icm_cokriging_k2_matches_reference_with_correlated_secondaries() -> None:
    # K=2, both secondaries sharing the same latent signal as the primary
    # (so they are also correlated with EACH OTHER, not just each with the
    # primary) -- exercises the secondary<->secondary block of the system,
    # per the plan's own testing note.
    # seed=46/1046 lands on an admissible (PSD) per-pair nugget fit while
    # still exercising a nonzero secondary<->secondary block -- see the
    # note in test_icm_cokriging_k1_matches_reference above.
    rng = np.random.default_rng(46)
    n = 100
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    latent = _smooth_field(x0, y0, seed=1046)
    z0 = latent + rng.normal(0.0, 0.15, n)
    zu1 = 1.2 * latent + rng.normal(0.0, 0.25, n)
    zu2 = 0.8 * latent + rng.normal(0.0, 0.25, n)
    grid = np.ascontiguousarray(rng.uniform(10.0, 90.0, (5, 2)))
    radius, max_neighbors = 40.0, 12

    ck = CoKriging(method="icm", radius=radius, max_neighbors=max_neighbors, min_neighbors=1)
    ck.fit(x0, y0, z0, [(x0, y0, zu1), (x0, y0, zu2)])
    result = ck.interpolate(grid)

    zu1_std = _standardize(z0, zu1)
    zu2_std = _standardize(z0, zu2)
    expected = _reference_icm_cokriging(
        x0, y0, z0, [(x0, y0, zu1_std), (x0, y0, zu2_std)],
        int(round(ck._fit.model_id)), ck._fit.model_range, ck._fit.nugget_matrix, ck._fit.sill_matrix,
        grid, radius, max_neighbors,
    )
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)
    # Secondary<->secondary block actually exercised: off-diagonal entry nonzero.
    assert ck._fit.sill_matrix[1, 2] != 0.0 or ck._fit.nugget_matrix[1, 2] != 0.0


def test_icm_cokriging_rejects_non_psd_coregionalization() -> None:
    from apbase.common.exceptions import CoKrigingNonPSDCoregionalizationError

    def field(x: np.ndarray, y: np.ndarray, seed: int, n_centers: int, length_scale: float) -> np.ndarray:
        rng = np.random.default_rng(seed)
        centers = rng.uniform(0.0, 100.0, (n_centers, 2))
        amps = rng.normal(0.0, 1.0, n_centers)
        out = np.zeros_like(x)
        for amp, (cx, cy) in zip(amps, centers, strict=True):
            out = out + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * length_scale**2))
        return out

    # Deterministic construction (trial_seed=0) known to produce a fitted
    # sill_matrix that fails the PSD check: primary dominated by a
    # short-range structure, each secondary sharing that short-range
    # component but otherwise driven by a DIFFERENT-range structure of its
    # own -- forcing ICM's single shared range to fit genuinely mismatched
    # per-pair spatial structures.
    trial_seed = 0
    rng = np.random.default_rng(trial_seed)
    n = 250
    x = rng.uniform(0.0, 100.0, n)
    y = rng.uniform(0.0, 100.0, n)
    f_short = field(x, y, trial_seed * 3 + 1, 40, 4.0)
    f_med = field(x, y, trial_seed * 3 + 2, 10, 15.0)
    f_long = field(x, y, trial_seed * 3 + 3, 3, 70.0)

    z0 = f_short + rng.normal(0.0, 0.01, n)
    zu1 = 0.6 * f_short + f_med + rng.normal(0.0, 0.01, n)
    zu2 = 0.6 * f_short + f_long + rng.normal(0.0, 0.01, n)

    try:
        CoKriging(method="icm").fit(x, y, z0, {"medium_range": (x, y, zu1), "long_range": (x, y, zu2)})
    except CoKrigingNonPSDCoregionalizationError:
        pass
    else:
        raise AssertionError("expected CoKrigingNonPSDCoregionalizationError for a mismatched-structure fit")


def _lmc_covariance(u: int, v: int, h: float, model_id: int, structure_ranges: np.ndarray,
                     coefficient_matrices: np.ndarray) -> float:
    if h <= 1e-6:
        return float(coefficient_matrices[:, u, v].sum())
    cov = 0.0
    for structure_idx in range(1, coefficient_matrices.shape[0]):
        shape = float(_variogram_shape(np.asarray(h), model_id, structure_ranges[structure_idx - 1]))
        cov += coefficient_matrices[structure_idx, u, v] * (1.0 - shape)
    return cov


def _reference_lmc_cokriging(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    secondaries_std: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_id: int,
    structure_ranges: np.ndarray,
    coefficient_matrices: np.ndarray,
    targets: np.ndarray,
    radius: float,
    max_neighbors: int,
) -> np.ndarray:
    """Independent NumPy solver for the general LMC block-kriging system.

    Same structure as _reference_icm_cokriging, generalized to sum over
    every nested structure's contribution (see co_kriging_lmc_local.f90 and
    _lmc_covariance above) instead of a single (nugget, sill) pair.
    """
    k = len(secondaries_std)
    xs = [x0] + [xu for xu, _, _ in secondaries_std]
    ys = [y0] + [yu for _, yu, _ in secondaries_std]
    zs = [z0] + [zu for _, _, zu in secondaries_std]

    out = np.empty(targets.shape[0], dtype=np.float64)
    for row, (tx, ty) in enumerate(targets):
        neighbor_idx = [_nearest_within(xs[u], ys[u], tx, ty, radius, max_neighbors) for u in range(k + 1)]
        sizes = [idx.size for idx in neighbor_idx]
        offsets = np.cumsum([0] + sizes)
        total = sum(sizes)
        nsys = total + 1
        a = np.zeros((nsys, nsys))
        b = np.zeros(nsys)

        for u in range(k + 1):
            for i, pi in enumerate(neighbor_idx[u]):
                gi = offsets[u] + i
                for v in range(k + 1):
                    for j, pj in enumerate(neighbor_idx[v]):
                        gj = offsets[v] + j
                        if gj <= gi:
                            continue
                        h = float(np.hypot(xs[u][pi] - xs[v][pj], ys[u][pi] - ys[v][pj]))
                        cov = _lmc_covariance(u, v, h, model_id, structure_ranges, coefficient_matrices)
                        a[gi, gj] = cov
                        a[gj, gi] = cov
                a[gi, gi] = _lmc_covariance(u, u, 0.0, model_id, structure_ranges, coefficient_matrices)
                h0 = float(np.hypot(xs[u][pi] - tx, ys[u][pi] - ty))
                b[gi] = _lmc_covariance(u, 0, h0, model_id, structure_ranges, coefficient_matrices)
                a[gi, nsys - 1] = 1.0
                a[nsys - 1, gi] = 1.0

        b[nsys - 1] = 1.0
        weights = np.linalg.solve(a, b)
        prediction = 0.0
        for u in range(k + 1):
            for i, pi in enumerate(neighbor_idx[u]):
                prediction += weights[offsets[u] + i] * zs[u][pi]
        out[row] = prediction
    return out


def test_lmc_cokriging_k1_matches_reference() -> None:
    # Isolates the INTERPOLATION kernel from the FIT, same rationale as
    # test_icm_cokriging_k1_matches_reference.
    rng = np.random.default_rng(70)
    n = 120
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    latent = _smooth_field(x0, y0, seed=71)
    z0 = latent + rng.normal(0.0, 0.05, n)
    zu = 1.4 * latent + rng.normal(0.0, 0.1, n)
    grid = np.ascontiguousarray(rng.uniform(10.0, 90.0, (6, 2)))
    radius, max_neighbors = 40.0, 12

    ck = CoKriging(method="lmc", n_structures=2, radius=radius, max_neighbors=max_neighbors, min_neighbors=1)
    ck.fit(x0, y0, z0, {"secondary": (x0, y0, zu)})
    result = ck.interpolate(grid)

    zu_std = _standardize(z0, zu)
    expected = _reference_lmc_cokriging(
        x0, y0, z0, [(x0, y0, zu_std)],
        int(round(ck._fit.model_id)), ck._fit.structure_ranges, ck._fit.coefficient_matrices,
        grid, radius, max_neighbors,
    )
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_lmc_cokriging_k2_matches_reference_with_correlated_secondaries() -> None:
    rng = np.random.default_rng(72)
    n = 100
    x0 = rng.uniform(0.0, 100.0, n)
    y0 = rng.uniform(0.0, 100.0, n)
    latent = _smooth_field(x0, y0, seed=73)
    z0 = latent + rng.normal(0.0, 0.05, n)
    zu1 = 1.2 * latent + rng.normal(0.0, 0.1, n)
    zu2 = 0.8 * latent + rng.normal(0.0, 0.1, n)
    grid = np.ascontiguousarray(rng.uniform(10.0, 90.0, (5, 2)))
    radius, max_neighbors = 40.0, 12

    ck = CoKriging(method="lmc", n_structures=2, radius=radius, max_neighbors=max_neighbors, min_neighbors=1)
    ck.fit(x0, y0, z0, [(x0, y0, zu1), (x0, y0, zu2)])
    result = ck.interpolate(grid)

    zu1_std = _standardize(z0, zu1)
    zu2_std = _standardize(z0, zu2)
    expected = _reference_lmc_cokriging(
        x0, y0, z0, [(x0, y0, zu1_std), (x0, y0, zu2_std)],
        int(round(ck._fit.model_id)), ck._fit.structure_ranges, ck._fit.coefficient_matrices,
        grid, radius, max_neighbors,
    )
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)
    coeffs = ck._fit.coefficient_matrices
    assert coeffs[:, 1, 2].any()  # secondary<->secondary block actually exercised


def test_lmc_goulard_voltz_recovers_known_coregionalization() -> None:
    # Known-case convergence test (S=2, nugget + 1 structure): simulate
    # z0/zu from a coregionalization model with a KNOWN nugget/sill matrix
    # pair, then check Goulard&Voltz recovers matrices close to the ones
    # used to generate the data. Construction: for a (nv,nv) PSD matrix
    # B^l = C^l (C^l)^T, Z = sum_l C^l @ W^l where each W^l is nv
    # independent standard fields sharing structure l's spatial covariance
    # (i.i.d. per point for the nugget; Cholesky-correlated for the
    # continuous structure) reproduces Cov(Z_u(x), Z_v(y)) = B^l_uv * g^l(h)
    # exactly by construction -- see COKRIGING_PLAN.md section 4a.
    rng = np.random.default_rng(80)
    n = 220
    x = rng.uniform(0.0, 100.0, n)
    y = rng.uniform(0.0, 100.0, n)
    nv = 2
    true_range = 25.0
    true_nugget_b = np.array([[0.3, 0.05], [0.05, 0.3]])
    true_sill_b = np.array([[1.0, 0.7], [0.7, 1.0]])

    h = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    spatial_cov = _covariance(h, MODEL_SPHERICAL, 0.0, 1.0, true_range)
    spatial_cov += np.eye(n) * 1e-9  # numerical jitter for a clean Cholesky
    l_spatial = np.linalg.cholesky(spatial_cov)

    def structure_field(b_matrix: np.ndarray, spatially_correlated: bool) -> np.ndarray:
        c = np.linalg.cholesky(b_matrix + np.eye(nv) * 1e-12)
        w = rng.normal(size=(nv, n))
        if spatially_correlated:
            w = w @ l_spatial.T
        return c @ w  # (nv, n)

    z_nugget = structure_field(true_nugget_b, spatially_correlated=False)
    z_continuous = structure_field(true_sill_b, spatially_correlated=True)
    z = z_nugget + z_continuous  # (nv, n): row 0 = primary, row 1 = secondary
    z0 = z[0]
    zu = z[1]

    ck = CoKriging(method="lmc", n_structures=2, max_neighbors=30, min_neighbors=3)
    ck.fit(x, y, z0, {"secondary": (x, y, zu)})

    fitted_nugget = ck._fit.coefficient_matrices[0]
    fitted_sill = ck._fit.coefficient_matrices[1]
    assert np.allclose(fitted_sill, true_sill_b, rtol=0.35, atol=0.15)
    assert np.allclose(ck._fit.structure_ranges[0], true_range, rtol=0.5)
    # Recovered matrices stay PSD (Goulard&Voltz's own guarantee, not just
    # close to truth): both eigenvalue sets non-negative within tolerance.
    assert np.all(np.linalg.eigvalsh(fitted_nugget) >= -1e-6)
    assert np.all(np.linalg.eigvalsh(fitted_sill) >= -1e-6)


def test_cokriging_rejects_invalid_method() -> None:
    try:
        CoKriging(method="bogus")
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid method")


if __name__ == "__main__":
    test_collocated_cokriging_k1_matches_reference()
    test_collocated_cokriging_k2_matches_reference_with_correlated_secondaries()
    test_collocated_cokriging_fitted_model_beats_plain_kriging_when_primary_sparse()
    test_co_kriging_free_function_matches_class()
    test_cokriging_lmc_model_values_override_raises_not_implemented()
    test_icm_cokriging_k1_matches_reference()
    test_icm_cokriging_k2_matches_reference_with_correlated_secondaries()
    test_icm_cokriging_rejects_non_psd_coregionalization()
    test_lmc_cokriging_k1_matches_reference()
    test_lmc_cokriging_k2_matches_reference_with_correlated_secondaries()
    test_lmc_goulard_voltz_recovers_known_coregionalization()
    test_cokriging_rejects_invalid_method()
