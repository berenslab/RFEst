import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase, interp1d
from .._splines import build_spline_matrix

__all__ = ['splineLG']


class splineLG(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False,
            nonlinearity='none'):
        
        super().__init__(X, y, dims, df, smooth, compute_mle) 
        self.nonlinearity = nonlinearity
        
    
    def cost(self, p):

        """

        Mean Squared Error.

        """
        
        XS = self.XS
        y = self.y    

        if self.fit_linear_filter:
            filter_output = XS @ p['b']
        else:
            if hasattr(self, 'b_opt'):
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl

        if self.fit_intercept:
            intercept = p['intercept']
        else:
            if hasattr(self, 'intercept'):
                intercept = self.intercept
            else:
                intercept = 0.

        if self.fit_history_filter:
            history_output = self.yS @ p['bh']  
        else:
            if hasattr(self, 'bh_spl'):
                history_output = self.yS @ self.bh_spl
            else:
                history_output = 0.
                

        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])
        
        mse = np.nanmean((y - self.fnl(filter_output + history_output + intercept, self.nonlinearity))**2)

        if self.beta:
            
            l1 = np.linalg.norm(p['b'], 1)
            l2 = np.linalg.norm(p['b'], 2)
            mse += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1) 
            
        return mse       
