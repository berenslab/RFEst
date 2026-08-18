import numpy as np

from rfest import sARD
from rfest.generate_test_data import generate_data_2d_stim
from rfest.metrics import mse
from rfest.utils import uvec


def _get_df(dims):
    df = [int(np.maximum(np.ceil(dim / 2), 3)) for dim in list(dims)]
    return df


def _fit_w_sard(X, y, dims):
    model = sARD(X, y, dims=dims, df=_get_df(dims))
    p0 = np.concatenate([[1.0, 1.0], np.ones(model.n_b)])
    model.fit(p0=p0, num_iters=50, verbose=0)
    return np.asarray(model.w_opt).flatten()


def test_sard_2d_stim():
    w_true, X, y, _, dims = generate_data_2d_stim(
        noise="white", rf_kind="complex_small", y_distr="none"
    )
    w_fit = _fit_w_sard(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01


def test_sard_2d_stim_spikes():
    w_true, X, y, _, dims = generate_data_2d_stim(
        noise="white", rf_kind="complex_small", y_distr="poisson"
    )
    w_fit = _fit_w_sard(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01
