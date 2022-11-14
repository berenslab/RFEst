import numpy as np

from rfest import nonlinearities


def get_response(X, w, intercept=0, dt=1, R=1, random_seed=None, distr='gaussian', nonlinearity='none'):
    r = dt * R * nonlinearities.apply_nonlinearity(X @ w.flatten() + intercept, kind=nonlinearity)
    r = add_noise(r=r, distr=distr, seed=random_seed, dt=dt)
    return r


def get_subunits_response(X, w, intercept=0, dt=1, R=1, random_seed=None, distr='gaussian', nl0='none', nl1='none'):
    np.random.seed(random_seed)

    filter_output = np.mean(nonlinearities.apply_nonlinearity(X @ w, kind=nl0), axis=1)
    r = R * nonlinearities.apply_nonlinearity(filter_output + intercept, kind=nl1)
    r = add_noise(r=r, distr=distr, seed=random_seed, dt=dt)
    return r


def add_noise(r, distr, seed, **params):
    np.random.seed(seed)

    if distr == 'gaussian':
        return np.random.normal(r)

    elif distr == 'add_gaussian':
        return r + np.random.normal(0, np.std(r) * 0.1, r.size)

    elif distr == 'poisson':
        r = np.maximum(params['dt'] * r, 1e-17)  # avoid 0.
        return np.random.poisson(r)

    elif distr == 'none':
        return r
