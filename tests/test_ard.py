import numpy as np

from rfest import ARD, ARDFixedPoint
from rfest.generate_test_data import generate_data_2d_stim
from rfest.metrics import mse
from rfest.utils import uvec


def _fit_w_ard(X, y, dims):
    model = ARD(X, y, dims=list(dims))
    # ARD gives every coefficient its own hyperparameter, so p0 carries one
    # theta per pixel on top of sigma and rho.
    p0 = np.concatenate([[1.0, 1.0], np.ones(X.shape[1])])
    model.fit(p0=p0, num_iters=100, verbose=0)
    return np.asarray(model.optimized_C_post @ X.T @ y / model.optimized_params[0] ** 2)


def _fit_w_ard_fixed_point(X, y, dims):
    model = ARDFixedPoint(X, y, dims=list(dims))
    # Here the per-coefficient hyperparameter is a precision, not a variance.
    p0 = np.concatenate([[1.0], np.ones(X.shape[1])])
    model.fit(p0=p0, num_iters=100, verbose=False)
    return np.asarray(model.w_opt).flatten()


def test_ard_2d_stim():
    w_true, X, y, _, dims = generate_data_2d_stim(
        noise="white", rf_kind="complex_small", y_distr="none"
    )
    w_fit = _fit_w_ard(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01


def test_ard_2d_stim_spikes():
    w_true, X, y, _, dims = generate_data_2d_stim(
        noise="white", rf_kind="complex_small", y_distr="poisson"
    )
    w_fit = _fit_w_ard(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01


def test_ard_fixed_point_2d_stim():
    w_true, X, y, _, dims = generate_data_2d_stim(
        noise="white", rf_kind="complex_small", y_distr="none"
    )
    w_fit = _fit_w_ard_fixed_point(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01
