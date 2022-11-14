import numpy as np

from rfest import splineLNP
from rfest.metrics import mse
from rfest.simulate import generate_data_2d_stim, generate_data_3d_stim
from rfest.utils import uvec, split_data


def _get_df(dims):
    df = [int(np.maximum(np.ceil(dim / 2), 3)) for dim in list(dims)]
    return df


def _fit_w_splinelnp(X, y, dims, dt, Xy_dev=None):
    df = _get_df(dims)
    model = splineLNP(X, y, dims=dims, df=df, dt=dt)
    kwargs = dict()
    if Xy_dev is not None:
        kwargs['extra'] = {'X': Xy_dev[0], 'y': Xy_dev[1]}
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01, **kwargs)
    return model.w_opt


def test_splinelnp_2d_stim_mle():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    df = _get_df(dims)
    model = splineLNP(X, y, dims=dims, df=df, dt=dt, compute_mle=True)
    assert model.w_mle is not None
    assert mse(uvec(model.w_mle.flatten()), uvec(w_true.flatten())) < 0.01


def test_splinelnp_STC():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    df = _get_df(dims)
    model = splineLNP(X, y, dims=dims, df=df, dt=dt)
    model.fit_STC(n_repeats=100, verbose=10)
    assert model.w_stc is not None


def test_splinelnp_2d_stim_spikes():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    w_opt = _fit_w_splinelnp(X, y, dims, dt)
    assert mse(uvec(w_opt.flatten()), uvec(w_true.flatten())) < 0.1


def test_splinelnp_3d_stim_spikes():
    w_true, X, y, dt, dims = generate_data_3d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    w_opt = _fit_w_splinelnp(X, y, dims, dt)
    assert mse(uvec(w_opt.flatten()), uvec(w_true.flatten())) < 0.01


def test_splinelnp_2d_stim_split_spikes():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    (X_train, y_train), (X_dev, y_dev), (_, _) = split_data(X, y, dt, frac_train=0.8, frac_dev=0.2)
    w_opt = _fit_w_splinelnp(X_train, y_train, dims, dt, Xy_dev=(X_dev, y_dev))
    assert mse(uvec(w_opt.flatten()), uvec(w_true.flatten())) < 0.01
