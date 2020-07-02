import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase
from .._splines import build_spline_matrix

__all__ = ['splineLG']


class splineLG(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', add_intercept=False, compute_mle=False,
            nonlinearity='none'):
        
        super().__init__(X, y, dims, df, smooth, add_intercept, compute_mle) 
        self.nonlinearity = nonlinearity
    
    def cost(self, p):

        """

        Mean Squared Error.

        """
        
        XS = self.XS
        y = self.y    

        if self.response_history:
            yS = self.yS
            mse = np.nanmean((y - self.nonlin(XS @ p['b'] + yS @ p['bh'], self.nonlinearity))**2)
        else:
            mse = np.nanmean((y - self.nonlin(XS @ p['b'], self.nonlinearity))**2)
        
        if self.beta:
            
            l1 = np.linalg.norm(p['b'], 1)
            l2 = np.linalg.norm(p['b'], 2)
            mse += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1) 
            
        return mse       
