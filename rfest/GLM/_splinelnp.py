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
    

    def cost(self, p):

        """
        Negetive Log Likelihood.
        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        R = self.R

        intercept = p['intercept'] if self.add_intercept else 0.
        history_output = self.yS @ p['bh'] if self.response_history else 0.
        filter_output = XS @ p['b']
        
        r = R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity).flatten()
        term0 = - np.log(r) @ y
        term1 = np.nansum(r) * dt

        neglogli = term0 + term1
        
        if self.beta:
            l1 = np.linalg.norm(p['b'], 1) 
            l2 = np.linalg.norm(p['b'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
