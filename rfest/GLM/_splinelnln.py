import jax.numpy as jnp
import jax.random as random
from jax.config import config

from rfest.GLM._base import splineBase
from rfest.utils import build_design_matrix

config.update("jax_enable_x64", True)

__all__ = ['splineLNLN']


class splineLNLN(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', filter_nonlinearity='softplus', output_nonlinearity='softplus',
                 compute_mle=False, **kwargs):

        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.filter_nonlinearity = filter_nonlinearity
        self.output_nonlinearity = output_nonlinearity

    def forwardpass(self, p, extra=None):

        dt = self.dt
        XS = self.XS if extra is None else extra['XS']

        if hasattr(self, 'h_spl'):
            yS = self.yS if extra is None else extra['yS']

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
                nl_params = [self.nl_params for i in range(self.n_s)]
            else:
                nl_params = [None for i in range(self.n_s)]

        if self.fit_linear_filter:
            linear_output = XS @ p['b'].reshape(self.n_b * self.n_c, self.n_s)
            nonlin_output = jnp.array(
                [self.fnl(linear_output[:, i], nl=self.filter_nonlinearity, params=nl_params[i]) for i in
                 range(self.n_s)])
            filter_output = jnp.mean(nonlin_output, 0)
        else:
            linear_output = XS @ self.b_opt.reshape(self.n_b * self.n_c, self.n_s)
            nonlin_output = jnp.array(
                [self.fnl(linear_output[:, i], nl=self.filter_nonlinearity, params=nl_params[i]) for i in
                 range(self.n_s)])
            filter_output = jnp.mean(nonlin_output, 0)

        if self.fit_history_filter:
            history_output = yS @ p['bh']
        else:
            if hasattr(self, 'bh_opt'):
                history_output = yS @ self.bh_opt
            elif hasattr(self, 'bh_spl'):
                history_output = yS @ self.bh_spl
            else:
                history_output = jnp.array([0.])

        r = dt * R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity,
                              params=nl_params[-1]).flatten()

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

        if self.beta and extra is None:
            l1 = jnp.linalg.norm(p['b'], 1)
            l2 = jnp.linalg.norm(p['b'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli

    def fit(self, p0=None, extra=None, num_subunits=2,
            num_epochs=1, num_iters=3000,
            initialize='random', metric=None,
            alpha=1, beta=0.05,
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False,
            step_size=1e-2, tolerance=10, verbose=100, random_seed=2046, return_model='best_dev_cost'):

        self.metric = metric

        self.alpha = alpha  # elastic net parameter (1=L1, 0=L2)
        self.beta = beta  # elastic net parameter - global penalty weight

        self.n_s = num_subunits
        self.num_iters = num_iters

        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept
        self.fit_R = fit_R

        # initialize parameters
        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()
        if 'b' not in dict_keys:
            if initialize == 'random':  # not necessary, but for consistency with others.
                key = random.PRNGKey(random_seed)
                b0 = 0.01 * random.normal(key, shape=(self.n_b * self.n_c * self.n_s,)).flatten()
                p0.update({'b': b0})

        if 'intercept' not in dict_keys:
            p0.update({'intercept': jnp.zeros(1)})

        if 'R' not in dict_keys:
            p0.update({'R': jnp.array([1.])})

        if 'bh' not in dict_keys:
            try:
                p0.update({'bh': self.bh_spl})
            except:
                p0.update({'bh': None})

        if 'nl_params' not in dict_keys:
            if hasattr(self, 'nl_params'):
                p0.update({'nl_params': [self.nl_params for i in range(self.n_s + 1)]})
            else:
                p0.update({'nl_params': [None for i in range(self.n_s + 1)]})

        if extra is not None:

            if self.n_c > 1:
                XS_ext = jnp.dstack([extra['X'][:, :, i] @ self.S for i in range(self.n_c)]).reshape(extra['X'].shape[0],
                                                                                                    -1)
                extra.update({'XS': XS_ext})
            else:
                extra.update({'XS': extra['X'] @ self.S})

            if hasattr(self, 'h_spl'):
                yh = jnp.array(build_design_matrix(extra['y'][:, jnp.newaxis], self.Sh.shape[0], shift=1))
                yS = yh @ self.Sh
                extra.update({'yS': yS})

            extra = {key: jnp.array(extra[key]) for key in extra.keys()}

        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_epochs, num_iters, metric, step_size, tolerance, verbose,
                                          return_model)

        self.R = self.p_opt['R'] if fit_R else jnp.array([1.])

        if fit_linear_filter:
            self.b_opt = self.p_opt['b']

            if self.n_c > 1:
                self.w_opt = jnp.stack(
                    [(self.S @ self.b_opt.reshape(self.n_b, self.n_c, self.n_s)[:, :, i]) for i in range(self.n_s)],
                    axis=-1)
            else:
                self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_s)

        if fit_history_filter:
            self.bh_opt = self.p_opt['bh']
            self.h_opt = self.Sh @ self.bh_opt

        if fit_intercept:
            self.intercept = self.p_opt['intercept']

        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt['nl_params']
