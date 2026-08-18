import numpy as np

from rfest import ALD, ASD, Ridge


def _exact_negative_log_evidence(X, y, C, sigma):
    """-log N(y; 0, sigma^2 I + X C X').

    For the linear-Gaussian model this IS the evidence, in closed form. It is
    derived here independently of `EmpiricalBayes`, so it is a ground truth for
    `negative_log_evidence` rather than a restatement of it -- a consistency
    check of that method against its own sufficient statistics cannot
    distinguish the posterior covariance from its inverse in the quadratic term.
    """
    n = X.shape[0]
    S = sigma ** 2 * np.eye(n) + X @ C @ X.T
    # Cholesky rather than slogdet: |S| underflows to 0 for these sizes.
    L = np.linalg.cholesky(S)
    return 0.5 * (2 * np.sum(np.log(np.diag(L)))
                  + y @ np.linalg.solve(S, y)
                  + n * np.log(2 * np.pi))


def _smooth_rf_data(n_features=12, n_samples=400, sigma=0.5, random_seed=2046):
    rng = np.random.default_rng(random_seed)
    lag = np.arange(n_features)
    w_true = np.exp(-0.5 * (lag - n_features / 3.) ** 2 / 2. ** 2)
    X = rng.normal(size=(n_samples, n_features))
    y = X @ w_true + sigma * rng.normal(size=n_samples)
    return X, y, sigma


def _assert_matches_exact(model, params, X, y, sigma):
    C = np.asarray(model.update_C_prior(np.asarray(params))[0])
    got = float(model.negative_log_evidence(np.asarray(params)))
    want = _exact_negative_log_evidence(X, y, C, sigma)
    assert np.isclose(got, want, rtol=1e-5), (params, got, want)


def test_asd_negative_log_evidence_matches_gaussian_marginal():
    # cond(C) reaches 5e9 by a smoothness of 3 and 1e17 by 8, so the grid also
    # covers the regime where the prior covariance is numerically singular.
    X, y, sigma = _smooth_rf_data()
    model = ASD(X, y, dims=[X.shape[1]])
    for delta in [1., 1.5, 3., 8., 20.]:
        _assert_matches_exact(model, [sigma, 1., delta], X, y, sigma)


def test_ridge_negative_log_evidence_matches_gaussian_marginal():
    X, y, sigma = _smooth_rf_data()
    model = Ridge(X, y, dims=[X.shape[1]])
    for theta in [0.5, 2.]:
        _assert_matches_exact(model, [sigma, 1., theta], X, y, sigma)


def test_ald_negative_log_evidence_matches_gaussian_marginal():
    X, y, sigma = _smooth_rf_data()
    model = ALD(X, y, dims=[X.shape[1]])
    for params in ([sigma, 1., 2., 4., 1., 1.], [sigma, 1., 1., 4., 3., 1.]):
        _assert_matches_exact(model, params, X, y, sigma)


def test_negative_log_evidence_is_consistent_across_prior_scales():
    """The prior scale rho must not change how accurately the evidence is computed.

    A jitter that is absolute rather than relative to the prior variances
    regularizes a differing amount at each scale, which shows up here as an
    error that grows as rho shrinks.
    """
    X, y, sigma = _smooth_rf_data()
    model = ASD(X, y, dims=[X.shape[1]])
    for rho in [1e-4, 1e-2, 1., 1e2, 1e4]:
        _assert_matches_exact(model, [sigma, rho, 3.], X, y, sigma)
