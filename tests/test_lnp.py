from rfest.generate_data import generate_2d_rf_data, generate_spike_train
from rfest import LNP
from rfest.metrics import mse
from rfest.utils import uvec, split_data


def test_lnp_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    model = LNP(X, y, dims=dims, dt=dt)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert mse(uvec(model.w_opt), uvec(w_true.flatten())) < 1e-1


def test_lnp_mle_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    model = LNP(X, y, dims=dims, dt=dt, compute_mle=True)

    assert mse(uvec(model.w_mle), uvec(w_true.flatten())) < 1e-1


def test_lnp_spikes():
    w_true, X, y, dims, dt = generate_spike_train(noise='white')

    model = LNP(X, y, dims=dims, dt=dt)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert model.w_opt.size == w_true.size


def test_lnp_spikes_split_data():
    w_true, X, y, dims, dt = generate_spike_train(noise='white')

    (X_train, y_train), (X_dev, y_dev), (_, _) = split_data(X, y, dt, frac_train=0.8, frac_dev=0.2)

    model = LNP(X_train, y_train, dims, dt=dt, nonlinearity='exponential')
    model.fit(extra={'X': X_dev, 'y': y_dev}, num_iters=50,
              metric='corrcoef', beta=0.01, verbose=0, tolerance=10)

    assert model.w_opt.size == w_true.size
