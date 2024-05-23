import jax.numpy as jnp
import jax.random as random
from jax import config

from rfest.GLM._base import Base
from rfest.utils import build_design_matrix

config.update("jax_enable_x64", True)

__all__ = ["LNLN"]


class LNLN(Base):
    """

    Multi-filters Linear-Nonlinear-Poisson model.

    """

    def __init__(
        self,
        X,
        y,
        dims,
        compute_mle=False,
        output_nonlinearity="softplus",
        filter_nonlinearity="softplus",
        **kwargs
    ):

        super().__init__(X, y, dims, compute_mle, **kwargs)

        # Optimization
        self.n_s = None

        # Parameters
        self.output_nonlinearity = output_nonlinearity
        self.filter_nonlinearity = filter_nonlinearity
        self.fit_subunits_weight = (
            kwargs["fit_subunits_weight"]
            if "fit_subunits_weight" in kwargs.keys()
            else False
        )

    def compute_filter_output(self, X, p=None):

        nl_params = self.get_nl_params(p, n_s=self.n_s)

        if self.fit_linear_filter:
            linear_output = X @ p["w"].reshape(self.n_features * self.n_c, self.n_s)
            nonlin_output = jnp.array(
                [
                    self.fnl(
                        linear_output[:, i],
                        nl=self.filter_nonlinearity,
                        params=nl_params[i],
                    )
                    for i in range(self.n_s)
                ]
            )
            filter_output = jnp.mean(nonlin_output, 0)
        else:
            linear_output = X @ self.w_opt.reshape(self.n_features * self.n_c, self.n_s)
            nonlin_output = jnp.array(
                [
                    self.fnl(
                        linear_output[:, i],
                        nl=self.filter_nonlinearity,
                        params=nl_params[i],
                    )
                    for i in range(self.n_s)
                ]
            )
            filter_output = jnp.mean(nonlin_output, 0)
        return filter_output

    def forwardpass(self, p=None, extra=None):

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

        r = (
            self.dt
            * R
            * self.fnl(
                filter_output + history_output + intercept, nl=self.output_nonlinearity
            ).flatten()
        )  # conditional intensity (per bin)

        return r

    def cost(self, p, extra=None, precomputed=None):
        return self.compute_cost(p, p["w"], "poisson", extra, precomputed)

    def fit(
        self,
        p0=None,
        extra=None,
        num_subunits=2,
        num_epochs=1,
        num_iters=3000,
        metric=None,
        alpha=1,
        beta=0.05,
        fit_linear_filter=True,
        fit_intercept=True,
        fit_R=True,
        fit_history_filter=False,
        fit_nonlinearity=False,
        step_size=1e-2,
        tolerance=10,
        verbose=100,
        random_seed=2046,
        return_model=None,
    ):

        self.metric = metric

        self.alpha = alpha  # elastic net parameter (1=L1, 0=L2)
        self.beta = (
            beta  # elastic net parameter - global penalty weight for linear filter
        )

        self.n_s = num_subunits
        self.num_iters = num_iters

        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        if extra is not None:

            if self.h_mle is not None:
                yh = jnp.array(
                    build_design_matrix(
                        extra["y"][:, jnp.newaxis], self.yh.shape[1], shift=1
                    )
                )
                extra.update({"yh": yh})

            extra = {key: jnp.array(extra[key]) for key in extra.keys()}

        # initialize parameters
        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()

        if "w" not in dict_keys:
            key = random.PRNGKey(random_seed)
            w0 = (
                0.01
                * random.normal(
                    key, shape=(self.n_features * self.n_c * self.n_s,)
                ).flatten()
            )
            p0.update({"w": w0})

        if "intercept" not in dict_keys:
            p0.update({"intercept": jnp.array([0.0])})

        if "R" not in dict_keys and self.fit_R:
            p0.update({"R": jnp.array([1.0])})

        if "h" not in dict_keys:
            try:
                p0.update({"h": self.h_mle})
            except:
                p0.update({"h": None})

        if "nl_params" not in dict_keys:
            if self.nl_params is not None:
                p0.update({"nl_params": [self.nl_params for _ in range(self.n_s + 1)]})
            else:
                p0.update({"nl_params": [None for _ in range(self.n_s + 1)]})

        self.p0 = p0
        self.p_opt = self.optimize_params(
            p0,
            extra,
            num_epochs,
            num_iters,
            metric,
            step_size,
            tolerance,
            verbose,
            return_model,
        )
        self.R = self.p_opt["R"]

        if fit_linear_filter:
            if self.n_c > 1:
                self.w_opt = self.p_opt["w"].reshape(
                    self.n_features, self.n_c, self.n_s
                )
            else:
                self.w_opt = self.p_opt["w"].reshape(self.n_features, self.n_s)

        if fit_history_filter:
            self.h_opt = self.p_opt["h"]

        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt["nl_params"]

        if fit_intercept:
            self.intercept = self.p_opt["intercept"]
