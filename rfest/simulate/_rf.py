import numpy as np
import scipy.stats

from rfest.utils import uvec


def rf_to_3d(trf, srf, norm=True):
    assert trf.ndim == 1, trf.ndim
    assert srf.ndim == 2, srf.ndim

    rf = np.kron(trf, srf.flat).reshape(trf.shape + srf.shape)

    if norm:
        rf = uvec(rf)
    return rf


def gaussian1d(dim=200, std=15., mean=0, dx_pixel=1., norm=True):
    x = (np.arange(0, dim) - dim / 2. + 0.5) / dx_pixel
    rf = scipy.stats.norm(mean, std).pdf(x)
    if norm:
        rf = uvec(rf)
    return rf


def gaussian2d(dims=(25, 25), std=None, cov=None, mean=(0, 0), dx_pixel=1., norm=True):
    assert (std is None) != (cov is None), 'Cannot either std or cov'

    if std is not None:
        assert len(std) == len(mean)
        cov = np.diag(std) ** 2
    elif cov is not None:
        cov = np.asarray(cov)

    x = (np.arange(0, dims[0]) - dims[0] / 2. + 0.5) * dx_pixel
    y = (np.arange(0, dims[1]) - dims[1] / 2. + 0.5) * dx_pixel

    xx, yy = np.meshgrid(x, y, indexing='ij')
    rf = scipy.stats.multivariate_normal(mean=mean, cov=cov, allow_singular=False).pdf(
        np.vstack([xx.flat, yy.flat]).T).reshape(dims)
    if norm:
        rf = uvec(rf)
    return rf


def gaussian3d(dims, std, srf_mean=(0, 0), dx_pixel=1., norm=True):
    trf = np.gradient(gaussian1d(dims[0], std[0]), norm=False)
    srf = gaussian2d(dims[1:], std[1:], mean=srf_mean, dx_pixel=dx_pixel, norm=False)
    rf = rf_to_3d(trf, srf)
    if norm:
        rf = uvec(rf)
    return rf


def mexicanhat1d(dim=200, std=15., mean=0, dx_pixel=1., a=0.8, w=0.65, norm=True):
    g0 = gaussian1d(dim=dim, std=std, mean=mean, dx_pixel=dx_pixel, norm=False)
    g1 = gaussian1d(dim=dim, std=std * a, mean=mean, dx_pixel=dx_pixel, norm=False)
    rf = g1 - w * g0
    if norm:
        rf = uvec(rf)
    return rf


def mexicanhat2d(dims=(25, 25), std=None, cov=None, mean=(0, 0), dx_pixel=1., a=0.55, w=0.65, norm=True):
    g0 = gaussian2d(dims, std=std, cov=cov, mean=mean, dx_pixel=dx_pixel, norm=False)
    g1 = gaussian2d(dims, mean=mean, dx_pixel=dx_pixel,
                    std=np.array(std) * a if std is not None else None,
                    cov=np.array(cov) * a**2 if cov is not None else None,
                    norm=False)
    rf = g1 - w * g0
    if norm:
        rf = uvec(rf)
    return rf


def mexicanhat3d(dims, std, a=0.3, w=0.65, srf_mean=(0, 0), dx_pixel=1., norm=False):
    trf = np.gradient(gaussian1d(dims[0], std[0]), norm=False)
    srf = mexicanhat2d(dims[1:], std[1:], a=a, w=w, mean=srf_mean, dx_pixel=dx_pixel, norm=False)
    rf = rf_to_3d(trf, srf)
    if norm:
        rf = uvec(rf)
    return rf


def subunits2d(num_subunits=5, dims=(25, 25), std=(3, 3), offset=(3, 3), kind='gaussian', random_seed=2046):
    if kind == 'gaussian':
        filter2d = gaussian2d
    elif kind == 'mexicanhat':
        filter2d = mexicanhat2d
    else:
        raise NotImplementedError(kind)

    w = np.zeros(list(dims) + [num_subunits])
    f = filter2d(dims, std)
    for i in range(num_subunits):

        if random_seed is not None:
            np.random.seed(random_seed + i + 5)
        h, v = np.random.randint(low=-offset[0], high=offset[1], size=2)
        w[:, :, i] = uvec(np.roll(np.roll(f, h, axis=0), v, axis=1))

    return w


def gabor2d(dims=(25, 25), omega=0.5, theta=np.pi / 6, func=np.cos, K=1.):
    radius = (int(dims[1] / 2.0), int(dims[0] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gabor = gauss * sinusoid

    return uvec(gabor)[:dims[0], :dims[1]]


def gabor3d(dims=(7, 20, 15), std=3, omega=0.5, theta=np.pi / 6, func=np.cos, K=np.pi):
    trf = np.gradient(gaussian1d(dims[0], std))
    srf = gabor2d(dims[1:], omega, theta, func, K)
    return rf_to_3d(trf, srf)


def V1complex_2d(dims=(30, 40), scale=(.025, .03)):
    dt = 1 / 60  # time bin size
    nt = dims[0]
    nx = dims[1]
    tt = np.arange(-nt * dt, 0, dt)

    kt1 = scipy.stats.gamma.pdf(-tt, dims[0] / 7.5, scale=scale[0])
    kt2 = scipy.stats.gamma.pdf(-tt, dims[1] / 6, scale=scale[1])
    kt1 /= np.linalg.norm(kt1)
    kt2 /= -np.linalg.norm(kt2)

    kt = np.vstack([kt1, kt2]).T

    xx = np.linspace(-2, 2, nx)

    kx1 = np.cos(2 * np.pi * xx / 2 + np.pi / 5) * np.exp(-1 / (2 * 0.35 ** 2) * xx ** 2)
    kx2 = np.sin(2 * np.pi * xx / 2 + np.pi / 5) * np.exp(-1 / (2 * 0.35 ** 2) * xx ** 2)

    kx1 /= np.linalg.norm(kx1)
    kx2 /= np.linalg.norm(kx2)

    kx = np.vstack([kx1, kx2])

    k = kt @ kx

    return uvec(k)
