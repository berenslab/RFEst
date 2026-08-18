import jax.numpy as jnp
import numpy as np

__all__ = [
    "ridge_kernel",
    "sparsity_kernel",
    "smoothness_kernel",
    "locality_kernel",
    "realfftbasis",
]

# Jitter applied to a prior covariance before inverting it, as a fraction of the
# largest prior variance. Relative rather than absolute: the prior covariance
# carries the units of the RF, so a fixed jitter regularizes an arbitrary
# amount -- nothing at all for a prior with large variances, and everything for
# one with small variances.
JITTER = 1e-7


def _jitter(variances):
    """`JITTER`, scaled to a prior with the given variances."""

    scale = jnp.max(variances)

    # `scale` is 0 only for a prior that has collapsed everywhere, which has no
    # scale to be relative to; fall back to an absolute jitter there.
    return JITTER * jnp.where(scale > 0, scale, 1.0)


def _diagonal_prior(variances, ncoeff):
    """Jittered (C, C_inv) for a diagonal prior covariance.

    Inverting elementwise is exact for a diagonal matrix, so the jitter only
    has to keep the variances away from zero and is applied as a floor rather
    than added throughout: well scaled variances are left untouched.
    """

    variances = jnp.broadcast_to(variances, (ncoeff,))
    variances = jnp.maximum(variances, _jitter(variances))

    return jnp.diag(variances), jnp.diag(1 / variances)


def _dense_prior(C):
    """Jittered (C, C_inv) for a dense prior covariance.

    The jittered C is returned rather than kept private to this function,
    so that C and C_inv are each other's inverse. The evidence uses both, and
    `log|C Lambda^-1|` relies on the cancellation between them to stay well
    conditioned when C is not -- pairing an unjittered C with a jittered
    C_inv costs over 100 nats for the SE kernel at moderate smoothness, where
    cond(C) exceeds 1e18 at only 20 coefficients.
    """

    C = C + jnp.eye(C.shape[0]) * _jitter(jnp.diag(C))

    return C, jnp.linalg.inv(C)


def ridge_kernel(params, ncoeff):
    """
    Prior for ridge regression.
    """

    theta = jnp.abs(params[0])

    return _diagonal_prior(theta, ncoeff)


def sparsity_kernel(params, ncoeff):
    """
    Sparse prior for ARD.

    See: Section 4 of Sahani & Linden (2003).

    """

    theta = jnp.abs(params)

    return _diagonal_prior(theta, ncoeff)


def smoothness_kernel(params, ncoeff):
    """

    1D Squared exponential (SE) covariance.
    See eq(10) in Sahani & Linden (2003).
    """

    delta = params[0]

    grid = jnp.arange(ncoeff)
    square_distance = (grid - grid.reshape(-1, 1)) ** 2  # pairwise squared distance
    C = jnp.exp(-0.5 * square_distance / delta**2)

    return _dense_prior(C)


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

    CxSqrt = jnp.diag(jnp.exp(-0.25 * 1 / taux**2 * (chi - nux) ** 2))

    Cf = B.T @ jnp.diag(jnp.exp(-0.5 * (jnp.abs(tauf * freq) - nuf) ** 2)) @ B

    C = CxSqrt @ Cf @ CxSqrt

    return _dense_prior(C)


def realfftbasis(nx):
    """
    Basis of sines+cosines for nn-point discrete fourier transform (DFT).

    Ported from MatLab code:
    https://github.com/leaduncker/SimpleEvidenceOpt/blob/master/util/realfftbasis.m

    """

    nn = nx

    ncos = np.ceil((nn + 1) / 2)
    nsin = np.floor((nn - 1) / 2)

    wvec = np.hstack(
        [np.arange(start=0.0, stop=ncos), np.arange(start=-nsin, stop=0.0)]
    )

    wcos = wvec[wvec >= 0]
    wsin = wvec[wvec < 0]

    x = np.arange(nx)

    t0 = np.cos(np.outer(wcos * 2 * np.pi / nn, x))
    t1 = np.sin(np.outer(wsin * 2 * np.pi / nn, x))

    B = np.vstack([t0, t1]) / np.sqrt(nn * 0.5)

    return B, wvec
