import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import Base, splineBase, interp1d
from .._splines import build_spline_matrix

from copy import deepcopy

__all__ = ['splineLG']


class splineLG(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False,
            nonlinearity='none'):
        
        super().__init__(X, y, dims, df, smooth, compute_mle) 
        self.nonlinearity = nonlinearity
        
    def forward_pass(self, p, extra):

        XS = self.XS if extra is None else extra['XS']

        if hasattr(self, 'h_spl'):
            yS = self.yS if extra is None else extra['yS']

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
            history_output = yS @ p['bh']  
        else:
            if hasattr(self, 'bh_opt'):
                history_output = yS @ self.bh_opt
            elif hasattr(self, 'bh_spl'):
                history_output = yS @ self.bh_spl
            else:
                history_output = 0.
                
        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])

        yhat = self.fnl(filter_output + history_output + intercept, self.nonlinearity)

        return yhat

    def cost(self, p, extra=None, precomputed=None):

        """

        Mean Squared Error.

        """
        
        y = self.y if extra is None else extra['y']
        yhat = self.forward_pass(p, extra) if precomputed is None else precomputed

        mse = np.nanmean((y - yhat)**2)

        if self.beta and extra is None:
            
            l1 = np.linalg.norm(p['b'], 1)
            l2 = np.linalg.norm(p['b'], 2)
            mse += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1) 
            
        return mse       