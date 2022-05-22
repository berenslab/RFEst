from rfest.generate_test_data import generate_data_2d_stim
from rfest import ASD
from rfest.metrics import mse
from rfest.utils import uvec


def _fit_w_asd(X, y, dims):
    model = ASD(X, y, dims=dims)
    model.fit(p0=[1., 1., 6., 6., ], num_iters=10, verbose=0)
    w_fit = model.optimized_C_post @ X.T @ y / model.optimized_params[0] ** 2
    return w_fit


def test_asd_2d_stim():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='none')
    w_fit = _fit_w_asd(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01


def test_asd_2d_stim_spikes():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    w_fit = _fit_w_asd(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01
