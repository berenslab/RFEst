import jax.numpy as jnp
from jax import config

from rfest.GLM._base import Base

config.update("jax_enable_x64", True)

__all__ = ["LNP"]


class LNP(Base):
    """
    Linear-Nonlinear-Poisson model.
    """

    def __init__(
        self, X, y, dims, compute_mle=False, output_nonlinearity="softplus", **kwargs
    ):

        super().__init__(X, y, dims, compute_mle, **kwargs)
        self.output_nonlinearity = output_nonlinearity

    def compute_filter_output(self, X, p=None):

        if self.fit_linear_filter:
            filter_output = X @ p["w"].flatten()
        else:
            if self.w_opt is not None:
                filter_output = X @ self.w_opt.flatten()
            elif self.w_mle is not None:
                filter_output = X @ self.w_mle.flatten()
            else:
                filter_output = X @ self.w_sta.flatten()
        return filter_output

    def forwardpass(self, p=None, extra=None):
        """
        Model output with current estimated parameters.
        """

        X = self.X if extra is None else extra["X"]
        X = X.reshape(X.shape[0], -1)

        if self.h_mle is not None:
            y = extra["yh"] if extra is not None else self.yh
        else:
            y = None

        intercept = self.get_intercept(p)
        R = self.get_R(p)
        history_output = self.compute_history_output(y, p)
        filter_output = self.compute_filter_output(X, p)

        nl_params = self.get_nl_params(p)
        r = (
            self.dt
            * R
            * self.fnl(
                filter_output + history_output + intercept,
                nl=self.output_nonlinearity,
                params=nl_params,
            ).flatten()
        )

        return r

    def cost(self, p, extra=None, precomputed=None):
        return self.compute_cost(p, p["w"], "poisson", extra, precomputed)
