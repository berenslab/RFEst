import jax.numpy as jnp
from jax.config import config
from rfest.GLM._base import Base

config.update("jax_enable_x64", True)

__all__ = ['LNP']


class LNP(Base):
    """

    Linear-Nonliear-Poisson model.

    """

    def __init__(self, X, y, dims, compute_mle=False, nonlinearity='softplus', **kwargs):

        super().__init__(X, y, dims, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def forward_pass(self, p, extra=None):

        """
        Model ouput with current estimated parameters.
        """

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

        if self.fit_R:  # maximum firing rate / scale factor
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
            nl_params = p['nl_params']
        else:
            if hasattr(self, 'nl_params'):
                nl_params = self.nl_params
            else:
                nl_params = None

        r = self.dt * R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity,
                                   params=nl_params).flatten()

        return r

    def cost(self, p, extra=None, precomputed=None):

        """
        Negetive Log Likelihood.
        """
        y = self.y if extra is None else extra['y']
        r = self.forward_pass(p, extra) if precomputed is None else precomputed
        r = jnp.maximum(r, 1e-20)  # remove zero to avoid nan in log.
        dt = self.dt

        term0 = - jnp.log(r / dt) @ y  # spike term from poisson log-likelihood
        term1 = jnp.sum(r)  # non-spike term

        neglogli = term0 + term1

        if self.beta and extra is None:
            l1 = jnp.linalg.norm(p['w'], 1)
            l2 = jnp.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
