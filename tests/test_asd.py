from generate_data import generate_small_rf_and_data
from rfest import ASD
from rfest.utils import uvec
from rfest.metrics import mse


def test_asd_small_rf():
    w_true, X, y, dims, dt = generate_small_rf_and_data(noise='white')

    model = ASD(X, y, dims=dims)
    model.fit(p0=[1., 1., 6., 6., ], num_iters=10, verbose=10)

    w_fit = model.optimized_C_post @ X.T @ y / model.optimized_params[0]**2

    assert mse(uvec(w_fit), uvec(w_true.flatten())) < 1e-1

    