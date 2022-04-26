from rfest.generate_data import generate_2d_rf_data
from rfest import ALD
from rfest.metrics import mse
from rfest.utils import uvec


def test_ald_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    sigma0 = [1.3]
    rho0 = [0.8]
    params_t0 = [3., 20., 3., 20.9]  # taus, nus, tauf, nuf
    params_y0 = [3., 20., 3., 20.9]
    p0 = sigma0 + rho0 + params_t0 + params_y0
    model = ALD(X, y, dims=dims)
    model.fit(p0=p0, num_iters=30, verbose=10)

    w_fit = model.optimized_C_post @ X.T @ y / model.optimized_params[0]**2

    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 1e-1

    