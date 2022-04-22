from generate_data import generate_small_rf_and_data
from rfest import LNLN


def test_splinelnln_small_rf():
    w_true, X, y, dims, dt = generate_small_rf_and_data(noise='white')

    model = LNLN(X, y, dims=dims, dt=dt)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert model.w_opt is not None


def test_splinelnln_mle_small_rf():
    w_true, X, y, dims, dt = generate_small_rf_and_data(noise='white')

    model = LNLN(X, y, dims=dims, dt=dt, compute_mle=True)

    assert model.w_mle is not None
