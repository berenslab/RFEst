import jax.numpy as jnp
from jax.config import config
from rfest.GLM._base import splineBase

config.update("jax_enable_x64", True)

__all__ = ['splineLNP']


class splineLNP(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', nonlinearity='softplus', compute_mle=False, **kwargs):
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def forwardpass(self, p, extra=None):

        """
        Model ouput with current estimated parameters.
        """

        XS = self.XS if extra is None else extra['XS']

        if hasattr(self, 'bh_spl'):
            if extra is not None and 'yS' in extra:
                yS = extra['yS']
            else:
                yS = self.yS

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

        if self.fit_nonlinearity:
            nl_params = p['nl_params']
        else:
            if hasattr(self, 'nl_params'):
                nl_params = self.nl_params
            else:
                nl_params = None

        if self.fit_linear_filter:
            filter_output = XS @ p['b']
        else:
            if hasattr(self, 'b_opt'):
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl

        if self.fit_history_filter:
            history_output = yS @ p['bh']
        else:
            if hasattr(self, 'bh_opt'):

                history_output = yS @ self.bh_opt
            elif hasattr(self, 'bh_spl'):
                history_output = yS @ self.bh_spl
            else:
                history_output = jnp.array([0.])

        r = self.dt * R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity,
                                   params=nl_params).flatten()

        return r

    def cost(self, p, extra=None, precomputed=None):

        """
        Negetive Log Likelihood.
        """

        y = self.y if extra is None else extra['y']
        r = self.forwardpass(p, extra) if precomputed is None else precomputed
        r = jnp.maximum(r, 1e-20)  # remove zero to avoid nan in log.
        dt = self.dt

        term0 = - jnp.log(r / dt) @ y
        term1 = jnp.sum(r)

        neglogli = term0 + term1

        if self.beta and extra is None:  # for w
            l1 = jnp.linalg.norm(p['b'], 1)
            l2 = jnp.linalg.norm(p['b'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        if hasattr(self, 'Cinv'):
            neglogli += 0.5 * p['b'] @ self.Cinv @ p['b']

        return neglogli
