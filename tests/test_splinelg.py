from rfest.generate_data import generate_2d_rf_data
from rfest import splineLG
from rfest.metrics import mse
from rfest.utils import uvec


def test_splinelg_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    df = [3, 4]
    model = splineLG(X, y, dims=dims, dt=dt, df=df)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert mse(uvec(model.w_opt), uvec(w_true.flatten())) < 1e-1


def test_splinelg_mle_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    df = [3, 4]
    model = splineLG(X, y, dims=dims, dt=dt, df=df, compute_mle=True)

    assert mse(uvec(model.w_mle), uvec(w_true.flatten())) < 1e-1
