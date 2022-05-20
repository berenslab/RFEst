from rfest.generate_data import generate_2d_rf_data, generate_spike_train
from rfest import splineLNP
from rfest.metrics import mse
from rfest.utils import uvec, split_data


def test_splinelnp_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    df = [3, 4]
    model = splineLNP(X, y, dims=dims, dt=dt, df=df)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert mse(uvec(model.w_spl), uvec(w_true.flatten())) < 1e-1


def test_splinelnp_mle_small_rf():
    w_true, X, y, dims, dt = generate_2d_rf_data(noise='white')

    df = [3, 4]
    model = splineLNP(X, y, dims=dims, dt=dt, df=df, compute_mle=True)

    assert mse(uvec(model.w_mle), uvec(w_true.flatten())) < 1e-1


def test_splinelnp_spikes():
    w_true, X, y, dims, dt = generate_spike_train(noise='white')

    df = [3, 4]
    model = splineLNP(X, y, dims=dims, dt=dt, df=df)
    model.fit(metric='corrcoef', num_iters=100, verbose=0, tolerance=10, beta=0.01)

    assert model.w_spl.size == w_true.size


def test_splinelnp_spikes_split_data():
    w_true, X, y, dims, dt = generate_spike_train(noise='white')

    (X_train, y_train), (X_dev, y_dev), (_, _) = split_data(X, y, dt, frac_train=0.8, frac_dev=0.2)

    df = [3, 4]

    model = splineLNP(X_train, y_train, dims, df, dt=dt, output_nonlinearity='exponential')
    model.fit(extra={'X': X_dev, 'y': y_dev}, num_iters=50,
              metric='corrcoef', beta=0.01, verbose=0, tolerance=10)

    assert model.w_spl.size == w_true.size


def test_lnp_spikes_parametric_nonlinearity():
    w_true, X, y, dims, dt = generate_spike_train(noise='white')

    (X_train, y_train), (X_dev, y_dev), (_, _) = split_data(X, y, dt, frac_train=0.8, frac_dev=0.2)

    df = [3, 4]

    model = splineLNP(X_train, y_train, dims, df=df, dt=dt, output_nonlinearity='spline')
    model.initialize_parametric_nonlinearity(init_to='exponential', params_dict={'df': 11})
    model.fit(extra={'X': X_dev, 'y': y_dev}, num_iters=50,
              metric='corrcoef', beta=0.01, verbose=False, tolerance=10,
              fit_nonlinearity=True)

    assert model.output_nonlinearity == 'spline'
    assert model.nl_basis is not None
    assert model.nl_xrange is not None
    assert model.nl_params is not None
    assert model.fnl_fitted is not None


def test_lnp_spikes_history_filter():
    w_true, X, y, dims, dt = generate_spike_train(noise='white')

    (X_train, y_train), (X_dev, y_dev), (_, _) = split_data(X, y, dt, frac_train=0.8, frac_dev=0.2)

    df = [3, 4]

    model = splineLNP(X_train, y_train, dims, df=df, dt=dt)
    model.initialize_history_filter(dims[0], df[0])
    model.fit(extra={'X': X_dev, 'y': y_dev}, num_iters=50,
              metric='corrcoef', beta=0.01, verbose=False, tolerance=10,
              fit_history_filter=True, fit_nonlinearity=False)

    assert model.h_opt is not None
    assert model.h_spl is not None
