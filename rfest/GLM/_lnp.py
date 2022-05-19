import jax.numpy as jnp
from jax.config import config

from rfest.GLM._base import Base

config.update("jax_enable_x64", True)

__all__ = ['LNP']


class LNP(Base):
    """
    Linear-Nonlinear-Poisson model.
    """

    def __init__(self, X, y, dims, compute_mle=False, nonlinearity='softplus', **kwargs):

        super().__init__(X, y, dims, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def forwardpass(self, p, extra=None):
        """
        Model output with current estimated parameters.
        """

        X = self.X if extra is None else extra['X']
        X = X.reshape(X.shape[0], -1)

        if self.h_mle is not None:
            if extra is not None:
                yh = extra['yh']
            else:
                yh = self.yh
        else:
            yh = None

        if self.fit_intercept:
            intercept = p['intercept']
        else:
            if self.intercept is not None:
                intercept = self.intercept
            else:
                intercept = 0.

        if self.fit_R:  # maximum firing rate / scale factor
            R = p['R']
        else:
            if self.R is not None:
                R = self.R
            else:
                R = 1.

        if self.fit_linear_filter:
            filter_output = X @ p['w'].flatten()
        else:
            if self.w_opt is not None:
                filter_output = X @ self.w_opt.flatten()
            elif self.w_mle is not None:
                filter_output = X @ self.w_mle.flatten()
            else:
                filter_output = X @ self.w_sta.flatten()

        if self.fit_history_filter:
            history_output = yh @ p['h']
        else:
            if self.h_opt is not None:
                history_output = yh @ self.h_opt
            elif self.h_mle is not None:
                history_output = yh @ self.h_mle
            else:
                history_output = 0.

        if self.fit_nonlinearity:
            nl_params = p['nl_params']
        else:
            if self.nl_params is not None:
                nl_params = self.nl_params
            else:
                nl_params = None

        r = self.dt * R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity,
                                   params=nl_params).flatten()

        return r

    def cost(self, p, extra=None, precomputed=None):
        return self.compute_cost(p, p['w'], 'poisson', extra, precomputed)
