import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase
from .._splines import build_spline_matrix

__all__ = ['splineLNP']

class splineLNP(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', nonlinearity='softplus',
            add_intercept=False, compute_mle=False, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, add_intercept, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def cost(self, b):

        """
        Negetive Log Likelihood.
        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        R = self.R

        filter_output = self.nonlin(XS @ b, self.nonlinearity).flatten()
        r = R * filter_output
        term0 = - np.log(r) @ y
        term1 = np.nansum(r) * dt

        neglogli = term0 + term1
        
        if self.lambd:
            l1 = np.nansum(np.abs(b))
            l2 = np.sqrt(np.nansum(b**2)) 
            neglogli += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
