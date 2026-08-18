from rfest.generate_test_data import generate_data_2d_stim
from rfest import ALD
from rfest.metrics import mse
from rfest.utils import uvec


def _fit_w_ald(X, y, dims):
    # nus is the centre of the locality window in coefficient index, so it has
    # to fall on the axis it belongs to: placing it off the end leaves every
    # prior variance at ~1e-37 and the prior carries no locality information.
    sigma0 = [1.3]
    rho0 = [0.8]
    params_t0 = [3., (dims[0] - 1) / 2, 1., 0.]  # taus, nus, tauf, nuf
    params_y0 = [3., (dims[1] - 1) / 2, 1., 0.]
    p0 = sigma0 + rho0 + params_t0 + params_y0
    model = ALD(X, y, dims=dims)
    model.fit(p0=p0, num_iters=30, verbose=0)
    w_fit = model.optimized_C_post @ X.T @ y / model.optimized_params[0]**2
    return w_fit


def test_asd_2d_stim():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='none')
    w_fit = _fit_w_ald(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01


def test_asd_2d_stim_spikes():
    w_true, X, y, dt, dims = generate_data_2d_stim(noise='white', rf_kind='complex_small', y_distr='poisson')
    w_fit = _fit_w_ald(X, y, dims)
    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 0.01
