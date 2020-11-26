import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import Base, interp1d


__all__ = ['LNP']

class LNP(Base):

    """

    Linear-Nonliear-Poisson model.

    """

    def __init__(self, X, y, dims, compute_mle=False,
            nonlinearity='softplus',**kwargs):

        super().__init__(X, y, dims, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity
        
    def forward_pass(self, p, extra=None):

        """
        Model ouput with current estimated parameters.
        """

        dt = self.dt
        X = self.X if extra is None else extra['X']
        X = X.reshape(X.shape[0], -1)

        if hasattr(self, 'h_mle'):
            if extra is not None:
                yh = extra['yh']
            else:
                yh = self.yh

        if self.fit_intercept:
            intercept = p['intercept'] 
        else:
            if hasattr(self, 'intercept'):
                intercept = self.intercept
            else:
                intercept = 0.
        
        if self.fit_R: # maximum firing rate / scale factor
            R = p['R']
        else:
            if hasattr(self, 'R'):
                R = self.R
            else:
                R = 1.

        if self.fit_linear_filter:
            filter_output = X @ p['w'].flatten()
        else:
            if hasattr(self, 'b_opt'): 
                filter_output = X @ self.w_opt.flatten()
            else:
                filter_output = X @ self.w_spl.flatten()    

        if self.fit_history_filter:
            history_output = yh @ p['h'] 
        else:
            if hasattr(self, 'h_opt'):
                history_output = yh @ self.h_opt
            elif hasattr(self, 'h_mle'):
                history_output = yh @ self.h_mle      
            else:
                history_output = 0.
        
        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])
            
        r = dt * R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity).flatten()

        return r

    def cost(self, p, extra=None, precomputed=None):

        """
        Negetive Log Likelihood.
        """
        y = self.y if extra is None else extra['y']
        r = self.forward_pass(p, extra) if precomputed is None else precomputed
        r = np.maximum(r, 1e-20) # remove zero to avoid nan in log.
        dt = self.dt

        term0 = - np.log(r / dt) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1

        if self.beta and extra is None:
            l1 = np.linalg.norm(p['w'], 1)
            l2 = np.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli

