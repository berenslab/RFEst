import numpy as np

from rfest import GLM
from rfest.metrics import mse
from rfest.simulate import generate_data_3d_stim
from rfest.utils import split_data, uvec


def _get_df(dims):
    df = [int(np.maximum(np.ceil(dim / 2), 3)) for dim in list(dims)]
    return df


def _fit_glm(
        Xy_train, dims, dt, Xy_dev=None, fit_history_filter=False,
        init_method='mle', output_nonlinearity='none', num_subunits=1,
        optimize_hps=False, fit_R=False, fit_intercept=True
):
    df = _get_df(dims)

    model = GLM(distr='gaussian', output_nonlinearity=output_nonlinearity)

    # Stimulus
    model.add_design_matrix(
        Xy_train[0], dims=dims, df=df, smooth='cr', kind='train', filter_nonlinearity='none', name='stimulus')
    if Xy_dev is not None:
        model.add_design_matrix(Xy_dev[0], dims=dims, df=df, name='stimulus', kind='dev')

    # History
    if fit_history_filter:
        model.add_design_matrix(
            Xy_train[1], dims=[5], df=[3], smooth='cr', kind='train', filter_nonlinearity='none', name='history')
        if Xy_dev is not None:
            model.add_design_matrix(Xy_dev[1], dims=[5], df=[3], kind='dev', name='history')

    fit_y = {'train': Xy_train[1]}
    if Xy_dev is not None:
        fit_y['dev'] = Xy_dev[1]

    model.initialize(num_subunits=num_subunits, dt=dt, method=init_method,
                     random_seed=42, compute_ci=False, y=Xy_train[1],
                     fit_R=fit_R, fit_intercept=fit_intercept)
    if not optimize_hps:
        model.fit(y=fit_y, num_iters=300, verbose=0, step_size=0.03, beta=0.001, metric='mse')
    else:
        model.fit_hps(y=fit_y, num_iters=300, verbose=0, step_size=0.03, alphas=(0., 1.),
                      betas=(0., 0.001, 1.), metric='mse')
    return model


def test_glm_3d_stim_outputnl_intercept():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False,
                     output_nonlinearity='exponential', fit_intercept=True)
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01
    assert 'opt' in model.intercept
    assert 'intercept' in model.p0
    assert 'intercept' in model.p['opt']
    assert model.intercept['opt']['global'] != 0
    assert model.get_intercept(p=model.p['opt'])['global'] != 0


def test_glm_3d_stim_outputnl_no_intercept():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False,
                     output_nonlinearity='exponential', fit_intercept=False, init_method='mle')
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01
    assert 'opt' not in model.intercept
    assert 'intercept' not in model.p0
    assert 'intercept' not in model.p['opt']
    assert model.get_intercept(p=model.p0)['global'] == model.intercept[model.init_method]['global']


def test_glm_3d_stim_outputnl_rate():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False,
                     output_nonlinearity='exponential', fit_intercept=True, fit_R=True)
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01
    assert 'opt' in model.R
    assert 'R' in model.p0
    assert 'R' in model.p['opt']
    assert model.R['opt']['global'] != 1.
    assert model.get_R(p=model.p['opt'])['global'] != 1.


def test_glm_3d_stim_outputnl_no_rate():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False,
                     output_nonlinearity='exponential', fit_intercept=True, fit_R=False)
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01
    assert 'opt' not in model.R
    assert 'R' not in model.p0
    assert 'R' not in model.p['opt']
    assert model.get_R(p=model.p0)['global'] == model.R[model.init_method]['global']


def test_glm_3d_stim():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False)
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01


def test_glm_3d_stim_train_dev_test():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = split_data(X, y, dt, frac_train=0.6, frac_dev=0.2)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, Xy_dev=(X_dev, y_dev), fit_history_filter=False)
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert model.score(X_dev, y_dev, metric='corrcoef') > 0.3
    assert model.score(X_test, y_test, metric='corrcoef') > 0.2
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01


def test_glm_3d_stim_outputnl_and_fit_rate():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False,
                     output_nonlinearity='exponential')
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01


def test_glm_3d_stim_2subunits():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (_, _), (_, _) = split_data(X, y, dt, frac_train=1.0, frac_dev=0.0)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, fit_history_filter=False,
                     num_subunits=2)
    assert model.score(X_train, y_train, metric='corrcoef') > 0.4
    assert mse(uvec(model.w['opt']['stimulus_s0'] + model.w['opt']['stimulus_s1']), uvec(w_true.flatten())) < 0.01


def test_glm_3d_stim_history():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = split_data(X, y, dt, frac_train=0.6, frac_dev=0.2)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, Xy_dev=(X_dev, y_dev), fit_history_filter=True)
    assert model.score({"stimulus": X_train, 'history': y_train}, y_train, metric='corrcoef') > 0.4
    assert model.score({"stimulus": X_dev, 'history': y_dev}, y_dev, metric='corrcoef') > 0.3
    assert model.score({"stimulus": X_test, 'history': y_test}, y_test, metric='corrcoef') > 0.2
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01


def test_glm_3d_stim_hp_sets():
    w_true, X, y, dt, dims = generate_data_3d_stim(
        noise='white', rf_kind='complex_small', y_distr='none', design_matrix=False)
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = split_data(X, y, dt, frac_train=0.6, frac_dev=0.2)
    model = _fit_glm(Xy_train=(X_train, y_train), dims=dims, dt=dt, Xy_dev=(X_dev, y_dev), optimize_hps=True)
    assert model.score({"stimulus": X_train}, y_train, metric='corrcoef') > 0.4
    assert model.score({"stimulus": X_dev}, y_dev, metric='corrcoef') > 0.3
    assert model.score({"stimulus": X_test}, y_test, metric='corrcoef') > 0.2
    assert mse(uvec(model.w['opt']['stimulus']), uvec(w_true.flatten())) < 0.01
