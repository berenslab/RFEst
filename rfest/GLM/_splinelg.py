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
    
    def cost(self, b):

        """

        Mean Squared Error.

        """
        
        XS = self.XS
        y = self.y    
        
        mse = np.nanmean((y - self.nonlin(XS @ b, self.nonlinearity))**2)

        if self.lambd:
            
            l1 = np.linalg.norm(b, 1)
            l2 = np.linalg.norm(b, 2)
            mse += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1) 
    
        return mse       
