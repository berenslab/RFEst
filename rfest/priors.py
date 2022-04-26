import jax.numpy as jnp
import numpy as np

__all__ = ["ridge_kernel", "sparsity_kernel", "smoothness_kernel", "locality_kernel", "realfftbasis"]


def ridge_kernel(params, ncoeff):
    """
    Prior for ridge regression.
    """

    theta = jnp.abs(params[0])
    C = jnp.eye(ncoeff) * theta
    C_inv = jnp.linalg.inv(C + jnp.eye(ncoeff) * 1e-07)

    return C, C_inv


def sparsity_kernel(params, ncoeff):
    """
    Sparse prior for ARD.

    See: Section 4 of Sahani & Linden (2003).
    
    """

    theta = jnp.abs(params)
    C = jnp.eye(ncoeff) * theta
    C_inv = jnp.linalg.inv(C + jnp.eye(ncoeff) * 1e-07)

    return C, C_inv


def smoothness_kernel(params, ncoeff):
    """

    1D Squared exponential (SE) covariance.
    See eq(10) in Sahani & Linden (2003).
    """

    delta = params[0]

    grid = jnp.arange(ncoeff)
    square_distance = (grid - grid.reshape(-1, 1)) ** 2  # pairwise squared distance
    C = jnp.exp(-.5 * square_distance / delta ** 2)
    C_inv = jnp.linalg.inv(C + jnp.eye(ncoeff) * 1e-07)

    return C, C_inv


def locality_kernel(params, ncoeff):
    """

    1D Locality prior covariance. 
    See eq(11, 12, 13) in Park & Pillow (2011).
    """

    chi = jnp.arange(ncoeff)

    taux = jnp.array(params[0])
    nux = jnp.array(params[1])
    tauf = jnp.array(params[2])
    nuf = jnp.array(params[3])

    (B, freq) = realfftbasis(ncoeff)
    B = jnp.array(B)
    freq = jnp.array(freq)

    CxSqrt = jnp.diag(jnp.exp(-0.25 * 1 / taux ** 2 * (chi - nux) ** 2))

    Cf = B.T @ jnp.diag(jnp.exp(-0.5 * (jnp.abs(tauf * freq) - nuf) ** 2)) @ B

    C = CxSqrt @ Cf @ CxSqrt
    C_inv = jnp.linalg.inv(C + jnp.eye(ncoeff) * 1e-07)

    return C, C_inv


def realfftbasis(nx):
    """
    Basis of sines+cosines for nn-point discrete fourier transform (DFT).
    
    Ported from MatLab code:
    https://github.com/leaduncker/SimpleEvidenceOpt/blob/master/util/realfftbasis.m
    
    """

    nn = nx

    ncos = np.ceil((nn + 1) / 2)
    nsin = np.floor((nn - 1) / 2)

    wvec = np.hstack([np.arange(start=0., stop=ncos), np.arange(start=-nsin, stop=0.)])

    wcos = wvec[wvec >= 0]
    wsin = wvec[wvec < 0]

    x = np.arange(nx)

    t0 = np.cos(np.outer(wcos * 2 * np.pi / nn, x))
    t1 = np.sin(np.outer(wsin * 2 * np.pi / nn, x))

    B = np.vstack([t0, t1]) / np.sqrt(nn * 0.5)

    return B, wvec
