import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase, interp1d
from .._splines import build_spline_matrix

__all__ = ['splineLNP']

class splineLNP(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', nonlinearity='softplus',
            compute_mle=False, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity
    

    def cost(self, p):

        """
        Negetive Log Likelihood.
        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        R = self.R
        
        filter_output = XS @ p['b'] if self.fit_linear_filter else XS @ self.b_opt
        intercept = p['intercept'] if self.fit_intercept else 0.
  
        if self.fit_history_filter:
            history_output = self.yS @ p['bh']  
        else:
            if hasattr(self, 'bh_spl'):
                history_output = self.yS @ self.bh_spl
            else:
                history_output = 0.
        
        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])

        r = R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity).flatten()
        term0 = - np.log(r) @ y
        term1 = np.sum(r) * dt

        neglogli = term0 + term1
        
        if self.beta:
            l1 = np.linalg.norm(p['b'], 1) 
            l2 = np.linalg.norm(p['b'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
