import jax.numpy as jnp
from jax import config

from rfest.GLM._base import splineBase

config.update("jax_enable_x64", True)

__all__ = ["splineLG"]


class splineLG(splineBase):

    def __init__(
        self,
        X,
        y,
        dims,
        df,
        smooth="cr",
        compute_mle=False,
        output_nonlinearity="none",
        **kwargs
    ):

        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.output_nonlinearity = output_nonlinearity

    def compute_filter_output(self, XS, p=None):
        if self.fit_linear_filter:
            filter_output = XS @ p["b"]
        else:
            if self.b_opt is not None:
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl
        return filter_output

    def forwardpass(self, p=None, extra=None):

        X = self.XS if extra is None else extra["XS"]

        if self.h_spl is not None:
            y = self.yS if extra is None else extra.get("yS", None)
        else:
            y = None

        intercept = self.get_intercept(p)
        filter_output = self.compute_filter_output(X, p)
        history_output = self.compute_history_output(y, p)

        nl_params = self.get_nl_params(p)
        yhat = self.fnl(
            filter_output + history_output + intercept,
            nl=self.output_nonlinearity,
            params=nl_params,
        ).flatten()

        return yhat

    def cost(self, p, extra=None, precomputed=None):
        return self.compute_cost(p, p["b"], "gaussian", extra, precomputed)
