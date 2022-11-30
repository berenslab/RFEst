import numpy as np

from rfest import build_design_matrix


def flickerfield(n_samples, dims=None, shift=0, beta=None, noise='gaussian', design_matrix=False, random_seed=2046):
    """
    Full field flicker.
    """

    np.random.seed(random_seed)

    if noise == 'gaussian':
        X = np.random.randn(n_samples)[:, np.newaxis]
    elif noise == 'binary':
        X = np.random.choice([-1, 1], size=n_samples)[:, np.newaxis]
    else:
        raise NotImplementedError(noise)

    if beta is not None:
        X = colornoise1d(1, dims=n_samples, beta=beta, phi=X.flatten(), random_seed=2046)[:, np.newaxis]
        X = (X - X.mean()) / X.std()

    if design_matrix:

        if dims is None:
            raise ValueError('`dims` is needed for building stimulus design matrix.')
        X = build_design_matrix(X, dims, shift)

    return X


def flickerbar(n_samples, dims, shift=0, beta=None, noise='gaussian', design_matrix=False, random_seed=2046):
    """
    Flicker bar.
    """

    nt, nx = dims

    np.random.seed(random_seed)

    if noise == 'gaussian':
        X = np.random.randn(n_samples, nx)
    elif noise == 'binary':
        X = np.random.choice([-1, 1], size=[n_samples, nx])
    else:
        raise NotImplementedError(noise)

    if beta is not None:
        if noise != 'gaussian':
            raise ValueError('1/f noise only applis to Gaussian noise.')
        X = colornoise1d(n_samples=n_samples, dims=nx, beta=beta, phi=X, random_seed=random_seed)
        X = (X - X.mean()) / X.std()

    if design_matrix:
        X = build_design_matrix(X, nt, shift)

    return X


def noise2d(n_samples, dims, shift=0, beta=None, noise='gaussian', design_matrix=False, random_seed=2046):
    """
    2D noise. Gaussian white noise or checkerboard binary noise.
    """

    if len(dims) == 2:
        nt = None
    elif len(dims) == 3:
        nt = dims[0]
        dims = dims[1:]
    else:
        raise NotImplementedError(len(dims))

    if noise == 'gaussian':
        X = np.random.randn(n_samples, *dims)
    elif noise == 'binary':
        X = np.random.choice([-1, 1], size=[n_samples, *dims])
    else:
        raise NotImplementedError(noise)

    if beta is not None:
        if noise != 'gaussian':
            raise ValueError('1/f noise only applies to Gaussian noise.')

        X = colornoise2d(n_samples=n_samples, dims=dims, beta=beta, phi=X, random_seed=random_seed)
        X = (X - X.mean()) / X.std()

    if design_matrix:
        if nt is None:
            pass
        else:
            X = build_design_matrix(X, nt, shift)

    return X


def colornoise1d(n_samples, dims, beta=1, phi=None, random_seed=2046):
    import warnings
    warnings.filterwarnings("ignore")

    u = np.fft.fftfreq(dims)
    Sf = (u ** 2) ** (- beta / 2)
    Sf[np.isinf(Sf)] = 0

    np.random.seed(random_seed)
    phi = np.random.randn(n_samples, dims) if phi is None else phi

    f = np.cos(2 * np.pi * phi, dtype=complex)
    f.imag = np.sin(2 * np.pi * phi)

    x = np.fft.fft(Sf ** 0.5 * f)

    return x.real


def colornoise2d(n_samples, dims, beta=1, phi=None, random_seed=2046):
    import warnings
    warnings.filterwarnings("ignore")

    u = np.fft.fftfreq(dims[0])[:, np.newaxis]
    u = np.tile(u, dims[1])
    v = np.fft.fftfreq(dims[1])[:, np.newaxis]
    v = np.tile(v.T, (dims[0], 1))

    Sf = (u ** 2 + v ** 2) ** (- beta / 2)
    Sf[np.isinf(Sf)] = 0

    np.random.seed(random_seed)
    phi = np.random.randn(n_samples, *dims) if phi is None else phi

    f = np.cos(2 * np.pi * phi, dtype=complex)
    f.imag = np.sin(2 * np.pi * phi)

    x = np.fft.fft2(Sf ** 0.5 * f)

    return x.real