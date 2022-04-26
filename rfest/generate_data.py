import numpy as np

from rfest.simulate import V1complex_2d, flickerbar, noise2d, get_response


def generate_2d_rf_data(noise='white'):
    beta = None if noise == 'white' else 1

    w_true = V1complex_2d()[23:30, 14:22]

    dims = w_true.shape
    dt = 1.

    X = flickerbar(n_samples=1000, dims=dims, design_matrix=True, random_seed=2046, beta=beta)
    y = get_response(X, w_true.flatten(), dt=dt)

    return w_true, X, y, dims, dt


def generate_3d_rf_data(noise='white'):
    beta = None if noise == 'white' else 1

    wframe = V1complex_2d()[23:30, 14:22]

    w_true = np.stack(
        [-0.1 * wframe, -0.5 * wframe, 0.1 * wframe, 0.5 * wframe, 1 * wframe * 0.5 * wframe, 0.1 * wframe])
    w_true += np.random.normal(0, np.max(wframe) * 0.01, w_true.shape)

    dims = w_true.shape
    dt = 1.

    X = noise2d(n_samples=1000, dims=dims[1:], design_matrix=True, random_seed=2046, beta=beta)
    y = get_response(X, w_true, dt=dt)

    return w_true, X, y, dims, dt


def generate_spike_train(noise='white'):
    # TODO: so far the response data is completly independent from the stimulus
    beta = None if noise == 'white' else 1

    w_true = V1complex_2d()[23:30, 14:22]
    dims = w_true.shape
    dt = 1.

    X = flickerbar(n_samples=1000, dims=dims, design_matrix=True, random_seed=2046, beta=beta)
    y = (np.random.uniform(0, 1, X.shape[0]) > 0.8).astype(int) +\
        (np.random.uniform(0, 1, X.shape[0]) > 0.8).astype(int)

    return w_true, X, y, dims, dt
