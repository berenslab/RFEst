import numpy as np
from rfest.simulate import V1complex_2d, flickerbar, get_response


def generate_small_rf_and_data(noise='white'):
    beta = None if noise == 'white' else 1

    w_true = V1complex_2d()[23:30, 14:22]

    dims = w_true.shape
    dt = 1.

    X = flickerbar(n_samples=1000, dims=dims, design_matrix=True, random_seed=2046, beta=beta)
    y = get_response(X, w_true.flatten(), dt=dt)

    return w_true, X, y, dims, dt


def generate_spike_train(noise='white'):
    beta = None if noise == 'white' else 1

    w_true = V1complex_2d()[23:30, 14:22]
    dims = w_true.shape
    dt = 1.

    X = flickerbar(n_samples=1000, dims=dims, design_matrix=True, random_seed=2046, beta=beta)
    y = (np.random.uniform(0, 1, X.shape[0]) > 0.8).astype(int) +\
        (np.random.uniform(0, 1, X.shape[0]) > 0.8).astype(int)

    return w_true, X, y, dims, dt
