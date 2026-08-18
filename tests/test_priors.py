import numpy as np
import pytest

from rfest.priors import (
    locality_kernel,
    ridge_kernel,
    smoothness_kernel,
    sparsity_kernel,
)

NCOEFF = 20

# Hyperparameters chosen to make each prior covariance badly conditioned, which
# is the ordinary case: cond(C) for the SE kernel passes 1e12 at a smoothness
# of 3 bins over 20 coefficients.
KERNELS = {
    "ridge": (ridge_kernel, [1.0]),
    "sparsity": (sparsity_kernel, np.linspace(0.1, 1.0, NCOEFF)),
    "smoothness": (smoothness_kernel, [8.0]),
    "locality": (locality_kernel, [2.0, NCOEFF / 2, 2.0, 1.0]),
}


@pytest.mark.parametrize("name", list(KERNELS))
def test_kernel_returns_a_matched_pair(name):
    """C and C_inv must be each other's inverse, not each other's approximation.

    Regularizing C_inv while returning C unregularized leaves the two describing
    different priors. The evidence uses both, and `log|C Lambda^-1|` stays well
    conditioned only through the cancellation between them.
    """
    kernel, params = KERNELS[name]
    C, C_inv = (np.asarray(a) for a in kernel(params, NCOEFF))

    assert np.allclose(C @ C_inv, np.eye(NCOEFF), atol=1e-6)


@pytest.mark.parametrize("name", ["ridge", "sparsity"])
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_variance_kernel_is_scale_invariant(name, scale):
    """Scaling the prior variances must scale the prior precision inversely.

    An absolute jitter breaks this: it is negligible against a prior with large
    variances and dominant against one with small variances, so the same prior
    expressed in different units gets regularized by different amounts.
    """
    kernel, params = KERNELS[name]
    _, C_inv = kernel(params, NCOEFF)
    _, C_inv_scaled = kernel(np.asarray(params) * scale, NCOEFF)

    assert np.allclose(np.asarray(C_inv_scaled) * scale, np.asarray(C_inv), rtol=1e-6)


@pytest.mark.parametrize("name", ["ridge", "sparsity"])
def test_variance_kernel_survives_a_collapsed_variance(name):
    """A prior variance driven to zero must not take the precision to infinity.

    ARD prunes coefficients by driving their variance down, so this is reached
    in ordinary use rather than only by a degenerate call.
    """
    kernel, params = KERNELS[name]
    params = np.asarray(params, dtype=float) * np.ones(NCOEFF)
    params[0] = 0.0

    C, C_inv = (np.asarray(a) for a in kernel(params, NCOEFF))

    assert np.all(np.isfinite(C_inv))
    assert np.allclose(C @ C_inv, np.eye(NCOEFF), atol=1e-6)
