import jax.numpy as jnp
from jax.config import config

from rfest.GLM._base import splineBase

config.update("jax_enable_x64", True)

__all__ = ['splineLNP']


# noinspection PyUnboundLocalVariable
class splineLNP(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', nonlinearity='softplus', compute_mle=False, **kwargs):
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def forwardpass(self, p, extra=None):

        """
        Model output with current estimated parameters.
        """

        XS = self.XS if extra is None else extra['XS']

        if self.bh_spl is not None:
            if extra is not None and 'yS' in extra:
                yS = extra['yS']
            else:
                yS = self.yS

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

        if self.fit_nonlinearity:
            nl_params = p['nl_params']
        else:
            if self.nl_params is not None:
                nl_params = self.nl_params
            else:
                nl_params = None

        if self.fit_linear_filter:
            filter_output = XS @ p['b']
        else:
            if self.b_opt is not None:
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl

        if self.fit_history_filter:
            history_output = yS @ p['bh']
        else:
            if self.bh_opt is not None:
                history_output = yS @ self.bh_opt
            elif self.bh_spl is not None:
                history_output = yS @ self.bh_spl
            else:
                history_output = jnp.array([0.])

        r = self.dt * R * self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity,
                                   params=nl_params).flatten()

        return r

    def cost(self, p, extra=None, precomputed=None):
        return self.compute_cost(p, p['b'], 'poisson', extra, precomputed)

