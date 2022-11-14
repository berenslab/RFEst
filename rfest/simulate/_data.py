import numpy as np

from rfest import build_design_matrix
from rfest.simulate._response import get_response
from rfest.simulate._rf import rf_to_3d, gaussian2d, V1complex_2d
from rfest.simulate._stim import flickerbar, noise2d


def generate_data_2d_stim(noise='white', rf_kind='gauss', y_distr='none', design_matrix=True):
    if rf_kind == 'gauss':
        w_true = gaussian2d(dims=(10, 8), std=(2., 2.))
    elif rf_kind == 'complex_small':
        w_true = V1complex_2d()[23:30, 14:22]
    elif rf_kind == 'complex':
        w_true = V1complex_2d()
    else:
        raise NotImplementedError(rf_kind)

    beta = None if noise == 'white' else 1

    dims = w_true.shape
    dt = 1.

    stim = flickerbar(n_samples=1000, dims=dims, design_matrix=False, random_seed=2046, beta=beta)
    X = build_design_matrix(stim, dims[0], shift=0)
    y = get_response(X, w_true.flatten(), dt=dt, distr=y_distr)

    assert X.shape[0] == y.size

    if design_matrix:
        return w_true, X, y, dt, dims
    else:
        return w_true, stim, y, dt, dims


def generate_data_3d_stim(noise='white', rf_kind='gauss', y_distr='none', design_matrix=True):
    beta = None if noise == 'white' else 1

    if rf_kind == 'gauss':
        srf = gaussian2d(dims=(10, 8), std=(2., 2.))
    elif rf_kind == 'complex_small':
        srf = V1complex_2d()[23:30, 14:22]
    elif rf_kind == 'complex':
        srf = V1complex_2d()
    else:
        raise NotImplementedError(rf_kind)

    trf = np.array([-1, -0.5, 0.1, 0.5, 1, 0.1])
    w_true = rf_to_3d(trf, srf)

    dims = w_true.shape
    dt = 1.

    stim = noise2d(n_samples=1000, dims=dims, design_matrix=False, random_seed=2046, beta=beta)
    X = build_design_matrix(stim, dims[0], shift=0)
    y = get_response(X, w_true.flatten(), dt=dt, distr=y_distr)

    assert X.shape[0] == y.size

    if design_matrix:
        return w_true, X, y, dt, dims
    else:
        return w_true, stim, y, dt, dims
