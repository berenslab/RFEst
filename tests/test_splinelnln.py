from generate_data import generate_small_rf_and_data
from rfest import splineLNLN
from rfest.utils import uvec
from rfest.metrics import mse


def test_splinelnln_small_rf():
    w_true, X, y, dims, dt = generate_small_rf_and_data(noise='white')

    df = [3, 4]
    model = splineLNLN(X, y, dims=dims, dt=dt, df=df)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert mse(uvec(model.w_spl), uvec(w_true.flatten())) < 1e-1


def test_splinelnln_mle_small_rf():
    w_true, X, y, dims, dt = generate_small_rf_and_data(noise='white')

    df = [3, 4]
    model = splineLNLN(X, y, dims=dims, dt=dt, df=df, compute_mle=True)

    assert mse(uvec(model.w_mle), uvec(w_true.flatten())) < 1e-1
