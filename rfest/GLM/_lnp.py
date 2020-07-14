import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import Base

__all__ = ['LNP']

class LNP(Base):

    """

    Linear-Nonliear-Poisson model.

    """

    def __init__(self, X, y, dims, dt, R=1, compute_mle=False,
            nonlinearity='softplus',**kwargs):

        super().__init__(X, y, dims, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity
        

    def cost(self, p):

        X = self.X
        y = self.y
        dt = self.dt
        R = self.R

        filter_output = X @ p['w'] if self.fit_linear_filter else X @ self.w_opt
        intercept = p['intercept'] if self.fit_intercept else 0.

        if self.fit_history_filter:
            history_output = self.yh @ p['h']  
        else:
            if hasattr(self, 'h_mle'):
                history_output = self.yh @ self.h_mle
            else:
                history_output = 0.

        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])

        
        
        r = R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity).flatten() 

        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) * dt # non-spike term

        neglogli = term0 + term1

        if self.beta:
            l1 = np.linalg.norm(p['w'], 1)
            l2 = np.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli

