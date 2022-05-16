import jax.numpy as jnp
from jax.config import config

from rfest.GLM._base import splineBase

config.update("jax_enable_x64", True)

__all__ = ['splineLG']


# noinspection PyUnboundLocalVariable
class splineLG(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False, nonlinearity='none', **kwargs):

        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.nonlinearity = nonlinearity

    def forwardpass(self, p, extra=None):

        XS = self.XS if extra is None else extra['XS']

        if self.h_spl is not None:
            yS = self.yS if extra is None else extra['yS']

        if self.fit_linear_filter:
            filter_output = XS @ p['b']
        else:
            if self.b_opt is not None:
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl

        if self.fit_intercept:
            intercept = p['intercept']
        else:
            if self.intercept is not None:
                intercept = self.intercept
            else:
                intercept = 0.

        if self.fit_history_filter:
            history_output = yS @ p['bh']
        else:
            if self.bh_opt is not None:
                history_output = yS @ self.bh_opt
            elif self.bh_spl is not None:
                history_output = yS @ self.bh_spl
            else:
                history_output = 0.

        if self.fit_nonlinearity:
            nl_params = p['nl_params']
        else:
            if self.nl_params is not None:
                nl_params = self.nl_params
            else:
                nl_params = None

        yhat = self.fnl(filter_output + history_output + intercept, nl=self.nonlinearity, params=nl_params).flatten()

        return yhat

    def cost(self, p, extra=None, precomputed=None):

        """

        Mean Squared Error.

        """

        y = self.y if extra is None else extra['y']
        yhat = self.forwardpass(p, extra) if precomputed is None else precomputed

        mse = jnp.nanmean((y - yhat) ** 2)

        if self.beta and extra is None:
            l1 = jnp.linalg.norm(p['b'], 1)
            l2 = jnp.linalg.norm(p['b'], 2)
            mse += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        if self.Cinv is not None:
            mse += 0.5 * p['b'] @ self.Cinv @ p['b']

        return mse
