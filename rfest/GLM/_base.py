import time

import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import config, grad, jit

try:
    from jax.example_libraries import optimizers, stax
    from jax.example_libraries.stax import BatchNorm, Dense, Relu
except ImportError:
    from jax.experimental import optimizers
    from jax.experimental import stax
    from jax.experimental.stax import Dense, BatchNorm, Relu

from rfest.loss import loss_mse, loss_neglogli, loss_penalty
from rfest.metrics import corrcoef, mse, r2
from rfest.nonlinearities import *
from rfest.priors import smoothness_kernel
from rfest.splines import bs, build_spline_matrix, cc, cr
from rfest.utils import build_design_matrix, uvec

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

__all__ = ["Base", "splineBase"]


class Base:
    """

    Base class for all GLMs.

    """

    def __init__(self, X, y, dims, compute_mle=False, dt=1.0):
        """

        Parameters
        ==========
        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : array_like, shape (n_samples, )
            Recorded response

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sy, sx]

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """
        # Optimization
        self.intercept = None
        self.R = None

        self.h_opt = None
        self.w_spl = None
        self.w_opt = None
        self.p0 = None
        self.p_opt = None

        self.metric = None
        self.Cinv = None
        self.alpha = None
        self.beta = None

        self.num_iters = None
        self.fit_R = None
        self.fit_linear_filter = None
        self.fit_history_filter = None
        self.fit_nonlinearity = None
        self.fit_intercept = None

        # store meta
        self.best_iteration = None
        self.return_model = None
        self.train_stop = None

        self.cost_dev = None
        self.cost_train = None
        self.metric_train = None
        self.metric_dev = None
        self.metric_train = None
        self.metric_dev_opt = None
        self.total_time_elapsed = None

        # Non-linearity
        self.filter_nonlinearity = None
        self.output_nonlinearity = None
        self.nl_params_opt = None
        self.nl_params = None
        self.nl_xrange = None
        self.nl_basis = None
        self.nl_bins = None
        self.fnl_fitted = None
        self.fnl_nonparametric = None

        # History filter
        self.h_mle = None
        self.yh = None
        self.shift_h = None

        self.w_stc = None

        self.ndim = len(dims)
        if self.ndim == 4:  # [t, x, y, c]
            self.n_samples, self.n_features, self.n_c = X.shape
            self.dims = dims[:-1]
        else:
            self.n_samples, self.n_features = X.shape
            self.n_c = 1
            self.dims = dims  # assumed order [t, y, x]

        self.dt = dt  # time bin size (for LNP and LNLN)
        self.compute_mle = compute_mle

        # compute sufficient statistics

        self.XtY = X.T @ y
        if jnp.all(y == y.astype(int)):  # if y is spikes
            self.w_sta = self.XtY / sum(y)
        else:  # if y is not spike
            self.w_sta = self.XtY / len(y)

        if self.n_c > 1:
            self.w_sta = self.w_sta.reshape(self.n_features, self.n_c)

        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = jnp.linalg.lstsq(self.XtX, self.XtY, rcond=None)[0]
            if self.n_c > 1:
                self.w_mle = self.w_mle.reshape(self.n_features, self.n_c)

        self.X = jnp.array(X)  # stimulus design matrix
        self.y = jnp.array(y)  # response

    def fit_STC(
        self,
        prewhiten=False,
        n_repeats=10,
        percentile=100.0,
        random_seed=2046,
        verbose=5,
    ):
        """

        Spike-triggered Covariance Analysis.

        Parameters
        ==========

        prewhiten: bool

        n_repeats: int
            Number of repeats for STC significance test.

        percentile: float
            Valid range of STC significance test.

        verbose: int
        random_seed: int
        """

        def get_stc(_X, _y, _w):

            n = len(_X)
            ste = _X[_y != 0]
            proj = ste - ste * _w * _w.T
            stc = proj.T @ proj / (n - 1)

            _eigvec, _eigval, _ = jnp.linalg.svd(stc)

            return _eigvec, _eigval

        key = random.PRNGKey(random_seed)

        y = self.y

        if prewhiten:

            if self.compute_mle is False:
                self.XtX = self.X.T @ self.X
                self.w_mle = jnp.linalg.solve(self.XtX, self.XtY)

            X = jnp.linalg.solve(self.XtX, self.X.T).T
            w = uvec(self.w_mle)

        else:
            X = self.X
            w = uvec(self.w_sta)

        eigvec, eigval = get_stc(X, y, w)

        self.w_stc = dict()
        if n_repeats:
            print("STC significance test: ")
            eigval_null = []
            for counter in range(n_repeats):
                if verbose:
                    if counter % int(verbose) == 0:
                        print(f"  {counter + 1:}/{n_repeats}")

                y_randomize = random.permutation(key, y)
                _, eigval_randomize = get_stc(X, y_randomize, w)
                eigval_null.append(eigval_randomize)
            else:
                if verbose:
                    print(f"Done.")
            eigval_null = jnp.vstack(eigval_null)
            max_null, min_null = jnp.percentile(
                eigval_null, percentile
            ), jnp.percentile(eigval_null, 100 - percentile)
            mask_sig_pos = eigval > max_null
            mask_sig_neg = eigval < min_null
            mask_sig = jnp.logical_or(mask_sig_pos, mask_sig_neg)

            self.w_stc["eigvec"] = eigvec
            self.w_stc["pos"] = eigvec[:, mask_sig_pos]
            self.w_stc["neg"] = eigvec[:, mask_sig_neg]

            self.w_stc["eigval"] = eigval
            self.w_stc["eigval_mask"] = mask_sig
            self.w_stc["eigval_pos_mask"] = mask_sig_pos
            self.w_stc["eigval_neg_mask"] = mask_sig_neg

            self.w_stc["max_null"] = max_null
            self.w_stc["min_null"] = min_null

        else:
            self.w_stc["eigvec"] = eigvec
            self.w_stc["eigval"] = eigval
            self.w_stc["eigval_mask"] = jnp.ones_like(eigval).astype(bool)

    def initialize_history_filter(self, dims, shift=1):
        """
        Parameters
        ==========

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]

        shift : int
            Should be 1 or larger.

        """
        y = self.y
        yh = jnp.array(build_design_matrix(y[:, jnp.newaxis], dims, shift=shift))
        self.shift_h = shift
        self.yh = jnp.array(yh)
        self.h_mle = jnp.linalg.solve(yh.T @ yh, yh.T @ y)

    def fit_nonparametric_nonlinearity(self, nbins=50, w=None):

        if w is None:
            if self.w_spl is not None:
                w = self.w_spl.flatten()
            elif self.w_mle is not None:
                w = self.w_mle.flatten()
            elif self.w_sta is not None:
                w = self.w_sta.flatten()
        else:
            w = jnp.array(w)

        X = self.X
        X = X.reshape(X.shape[0], -1)
        y = self.y

        output_raw = X @ uvec(w)
        output_spk = X[y != 0] @ uvec(w)

        hist_raw, bins = jnp.histogram(output_raw, bins=nbins, density=True)
        hist_spk, _ = jnp.histogram(output_spk, bins=bins, density=True)

        mask = ~(hist_raw == 0)

        yy0 = hist_spk[mask] / hist_raw[mask]

        self.nl_bins = bins[1:]
        self.fnl_nonparametric = interp1d(bins[1:][mask], yy0)

    def initialize_parametric_nonlinearity(
        self, init_to="exponential", method=None, params_dict=None
    ):

        if method is None:  # if no methods specified, use defaults.
            method = self.output_nonlinearity or self.filter_nonlinearity
        else:  # otherwise, overwrite the default nonlinearity.
            self.output_nonlinearity = method
            if self.filter_nonlinearity is not None:
                self.filter_nonlinearity = method

        assert method is not None

        # prepare data
        if params_dict is None:
            params_dict = {}
        xrange = params_dict["xrange"] if "xrange" in params_dict else 5
        nx = params_dict["nx"] if "nx" in params_dict else 1000
        x0 = jnp.linspace(-xrange, xrange, nx)

        if init_to == "exponential":
            y0 = jnp.exp(x0)
        elif init_to == "softplus":
            y0 = softplus(x0)
        elif init_to == "relu":
            y0 = relu(x0)
        elif init_to == "nonparametric":
            y0 = self.fnl_nonparametric(x0)
        elif init_to == "gaussian":
            import scipy.signal

            # noinspection PyUnresolvedReferences
            y0 = scipy.signal.gaussian(nx, nx / 10)
        else:
            raise NotImplementedError(init_to)

        # fit nonlin
        if method == "spline":
            smooth = params_dict["smooth"] if "smooth" in params_dict else "cr"
            df = params_dict["df"] if "df" in params_dict else 7
            if smooth == "cr":
                X = cr(x0, df)
            elif smooth == "cc":
                X = cc(x0, df)
            elif smooth == "bs":
                deg = params_dict["degree"] if "degree" in params_dict else 3
                X = bs(x0, df, deg)
            else:
                raise NotImplementedError(smooth)

            opt_params = jnp.linalg.pinv(X.T @ X) @ X.T @ y0

            self.nl_basis = X

            def _nl(_opt_params, x_new):
                return jnp.maximum(interp1d(x0, X @ _opt_params)(x_new), 0)

        elif method == "nn":

            def loss(_params, _data):
                x = _data["x"]
                y = _data["y"]
                yhat = _predict(_params, x)
                return jnp.mean((y - yhat) ** 2)

            @jit
            def step(_i, _opt_state, _data):
                p = get_params(_opt_state)
                g = grad(loss)(p, _data)
                return opt_update(_i, g, _opt_state)

            random_seed = (
                params_dict["random_seed"] if "random_seed" in params_dict else 2046
            )
            key = random.PRNGKey(random_seed)

            step_size = params_dict["step_size"] if "step_size" in params_dict else 0.01
            layer_sizes = (
                params_dict["layer_sizes"]
                if "layer_sizes" in params_dict
                else [10, 10, 1]
            )
            layers = []
            for layer_size in layer_sizes:
                layers.append(Dense(layer_size))
                layers.append(BatchNorm(axis=(0, 1)))
                layers.append(Relu)
            else:
                layers.pop(-1)

            init_random_params, _predict = stax.serial(*layers)

            num_subunits = (
                params_dict["num_subunits"] if "num_subunits" in params_dict else 1
            )
            _, init_params = init_random_params(key, (-1, num_subunits))

            opt_init, opt_update, get_params = optimizers.adam(step_size)
            opt_state = opt_init(init_params)

            num_iters = params_dict["num_iters"] if "num_iters" in params_dict else 1000
            if num_subunits == 1:
                data = {"x": x0.reshape(-1, 1), "y": y0.reshape(-1, 1)}
            else:
                data = {
                    "x": jnp.vstack([x0 for _ in range(num_subunits)]).T,
                    "y": y0.reshape(-1, 1),
                }

            for i in range(num_iters):
                opt_state = step(i, opt_state, data)
            opt_params = get_params(opt_state)

            def _nl(_opt_params, x_new):
                if len(x_new.shape) == 1:
                    x_new = x_new.reshape(-1, 1)
                return jnp.maximum(_predict(_opt_params, x_new), 0)

        else:
            raise NotImplementedError(method)

        self.nl_xrange = x0
        self.nl_params = opt_params
        self.fnl_fitted = _nl

    def fnl(self, x, nl, params=None):
        """
        Choose a fixed nonlinear function or fit a flexible one ('nonparametric').
        """

        if nl == "softplus":
            return softplus(x)

        elif nl == "exponential":
            return jnp.exp(x)

        elif nl == "softmax":
            return softmax(x)

        elif nl == "sigmoid":
            return sigmoid(x)

        elif nl == "tanh":
            return jnp.tanh(x)

        elif nl == "relu":
            return relu(x)

        elif nl == "leaky_relu":
            return leaky_relu(x)

        elif nl == "selu":
            return selu(x)

        elif nl == "swish":
            return swish(x)

        elif nl == "elu":
            return elu(x)

        elif nl == "none":
            return x

        elif nl == "nonparametric":
            return self.fnl_nonparametric(x)

        elif nl == "spline" or nl == "nn":

            return self.fnl_fitted(params, x)

        else:
            raise ValueError(f"Input filter nonlinearity `{nl}` is not supported.")

    def compute_cost(self, p, penalty_w, dist, extra=None, precomputed=None):
        """
        Negative Log Likelihood.
        """
        y = self.y if extra is None else extra["y"]
        r = self.forwardpass(p, extra) if precomputed is None else precomputed

        if dist == "poisson":
            loss = loss_neglogli(y, r, dt=self.dt)
        elif dist == "gaussian":
            loss = loss_mse(y, r)
        else:
            raise NotImplementedError(dist)

        if (self.beta is not None) and (extra is None):
            loss += loss_penalty(penalty_w, self.alpha, self.beta)

        if self.Cinv is not None:
            loss += 0.5 * p["b"] @ self.Cinv @ p["b"]

        return loss

    def forwardpass(self, p=None, extra=None):
        raise NotImplementedError()

    def compute_filter_output(self, X, p=None):
        raise NotImplementedError()

    def get_intercept(self, p=None):
        if self.fit_intercept:
            intercept = p["intercept"]
        else:
            if self.intercept is not None:
                intercept = self.intercept
            else:
                intercept = 0.0
        return intercept

    def get_R(self, p=None):
        if self.fit_R:  # maximum firing rate / scale factor
            R = p["R"]
        else:
            if self.R is not None:
                R = self.R
            else:
                R = 1.0
        return R

    def get_nl_params(self, p=None, n_s=None):
        if self.fit_nonlinearity:
            nl_params = p["nl_params"]
        else:
            if self.nl_params is not None:
                nl_params = self.nl_params
            else:
                nl_params = None

            if n_s is not None:
                nl_params = [nl_params] * n_s

        return nl_params

    def compute_history_output(self, yh=None, p=None):
        if self.fit_history_filter:
            history_output = yh @ p["h"]
        else:
            if self.h_opt is not None:
                history_output = yh @ self.h_opt
            elif self.h_mle is not None:
                history_output = yh @ self.h_mle
            else:
                history_output = 0.0
        return history_output

    @staticmethod
    def print_progress(
        i, time_elapsed, c_train=None, c_dev=None, m_train=None, m_dev=None
    ):
        opt_info = f"{i}".ljust(13) + f"{time_elapsed:>.3f}".ljust(13)
        if c_train is not None:
            opt_info += f"{c_train:.3f}".ljust(16)
        if c_dev is not None:
            opt_info += f"{c_dev:.3f}".ljust(16)
        if m_train is not None:
            opt_info += f"{m_train:.3f}".ljust(16)
        if m_dev is not None:
            opt_info += f"{m_dev:.3f}".ljust(16)
        print(opt_info)

    @staticmethod
    def print_progress_header(c_train=False, c_dev=False, m_train=False, m_dev=False):
        opt_title = "Iters".ljust(13) + "Time (s)".ljust(13)
        if c_train is not None:
            opt_title += "Cost (train)".ljust(16)
        if c_dev is not None:
            opt_title += "Cost (dev)".ljust(16)
        if m_train is not None:
            opt_title += "Metric (train)".ljust(16)
        if m_dev is not None:
            opt_title += "Metric (dev)".ljust(16)
        print(opt_title)

    def optimize_params(
        self,
        p0,
        extra,
        num_epochs,
        num_iters,
        metric,
        step_size,
        tolerance,
        verbose,
        return_model=None,
        atol=1e-5,
        min_iters=300,
    ) -> dict:
        """
        Gradient descent using JAX optimizer, and verbose logging.
        """
        if return_model is None:
            if extra is not None:
                return_model = "best_dev_cost"
            else:
                return_model = "best_train_cost"

        assert (extra is not None) or (
            "dev" not in return_model
        ), "Cannot use dev set if dev set is not given."

        if num_epochs != 1:
            raise NotImplementedError()

        @jit
        def step(_i, _opt_state):
            p = get_params(_opt_state)
            g = grad(self.cost)(p)
            return opt_update(_i, g, _opt_state)

        # preallocation
        cost_train = np.zeros(num_iters)
        cost_dev = np.zeros(num_iters)
        metric_train = np.zeros(num_iters)
        metric_dev = np.zeros(num_iters)
        params_list = []

        if verbose:
            self.print_progress_header(
                c_train=True,
                c_dev=extra,
                m_train=metric is not None,
                m_dev=metric is not None and extra,
            )

        time_start = time.time()

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)

        i = 0
        c_dev = None
        m_train = None
        m_dev = None
        y_pred_dev = None

        for i in range(num_iters):

            opt_state = step(i, opt_state)
            params = get_params(opt_state)
            params_list.append(params)

            y_pred_train = self.forwardpass(p=params, extra=None)
            c_train = self.cost(p=params, precomputed=y_pred_train)
            cost_train[i] = c_train

            if extra is not None:
                y_pred_dev = self.forwardpass(p=params, extra=extra)
                c_dev = self.cost(p=params, extra=extra, precomputed=y_pred_dev)
                cost_dev[i] = c_dev

            if metric is not None:

                m_train = self.compute_score(self.y, y_pred_train, metric)
                metric_train[i] = m_train

                if extra is not None:
                    m_dev = self.compute_score(extra["y"], y_pred_dev, metric)
                    metric_dev[i] = m_dev

            time_elapsed = time.time() - time_start
            if verbose and (i % int(verbose) == 0):
                self.print_progress(
                    i,
                    time_elapsed,
                    c_train=c_train,
                    c_dev=c_dev,
                    m_train=m_train,
                    m_dev=m_dev,
                )

            if tolerance and i > min_iters:  # tolerance = 0: no early stop.

                total_time_elapsed = time.time() - time_start
                cost_train_slice = cost_train[i - tolerance : i]
                cost_dev_slice = cost_dev[i - tolerance : i]

                if jnp.all(cost_dev_slice[1:] - cost_dev_slice[:-1] > 0):
                    stop = "dev_stop"
                    if verbose:
                        print(
                            f"Stop at {i} steps: "
                            + f"cost (dev) has been monotonically increasing for {tolerance} steps.\n"
                        )
                    break

                if jnp.all(cost_train_slice[:-1] - cost_train_slice[1:] < atol):
                    stop = "train_stop"
                    if verbose:
                        print(
                            f"Stop at {i} steps: "
                            + f"cost (train) has been changing less than {atol} for {tolerance} steps.\n"
                        )
                    break

        else:
            total_time_elapsed = time.time() - time_start
            stop = "maxiter_stop"

            if verbose:
                print("Stop: reached {0} steps.\n".format(num_iters))

        if return_model == "best_dev_cost":
            best = np.argmin(cost_dev[: i + 1])

        elif return_model == "best_train_cost":
            best = np.argmin(cost_train[: i + 1])

        elif return_model == "best_dev_metric":
            if metric in ["mse", "gcv"]:
                best = np.argmin(metric_dev[: i + 1])
            else:
                best = np.argmax(metric_dev[: i + 1])

        elif return_model == "best_train_metric":
            if metric in ["mse", "gcv"]:
                best = np.argmin(metric_train[: i + 1])
            else:
                best = np.argmax(metric_train[: i + 1])

        else:
            if return_model != "last":
                print("Provided `return_model` is not supported. Fallback to `last`")
            if stop == "dev_stop":
                best = i - tolerance
            else:
                best = i

        if verbose:
            print(
                f"Returning model: {return_model} at iteration {best} of {i} (Max: {num_iters}).\n"
            )

        params = params_list[best]
        metric_dev_opt = metric_dev[best]

        self.best_iteration = best
        self.return_model = return_model
        self.train_stop = stop

        self.cost_train = cost_train[: i + 1]
        self.cost_dev = cost_dev[: i + 1]
        self.metric_train = metric_train[: i + 1]
        self.metric_dev = metric_dev[: i + 1]
        self.metric_dev_opt = metric_dev_opt
        self.total_time_elapsed = total_time_elapsed

        return params

    def fit(
        self,
        p0=None,
        extra=None,
        initialize="random",
        num_epochs=1,
        num_iters=5,
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
        verbose=1,
        random_seed=2046,
        return_model=None,
    ):
        """

        Parameters
        ==========

        p0 : dict
            * 'b': Initial spline coefficients.
            * 'bh': Initial response history filter coefficients

        extra : None or dict {'X': X_dev, 'y': y_dev}
            Development set.

        initialize : None or str
            Parametric initialization.
            * if `initialize=None`, `w` will be initialized by STA.
            * if `initialize='random'`, `w` will be randomly initialized.

        num_iters : int
            Max number of optimization iterations.

        metric : None or str
            Extra cross-validation metric. Default is `None`. Or
            * 'mse': mean squared error
            * 'r2': R2 score
            * 'corrcoef': Correlation coefficient

        alpha : float, from 0 to 1.
            Elastic net parameter, balance between L1 and L2 regularization.
            * 0.0 -> only L2
            * 1.0 -> only L1

        beta : float
            Elastic net parameter, overall weight of regularization.

        step_size : float
            Initial step size for JAX optimizer (ADAM).

        tolerance : int
            Set early stop tolerance. Optimization stops when cost (dev) monotonically
            increases or cost (train) stop increases for tolerance=n steps.
            If `tolerance=0`, then early stop is not used.

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.metric = metric  # metric for cross-validation and prediction

        self.alpha = alpha
        self.beta = (
            beta  # elastic net parameter - global penalty weight for linear filter
        )
        self.num_iters = num_iters

        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        # initialize parameters
        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()
        if "w" not in dict_keys:
            if initialize is None:
                p0.update({"w": self.w_sta})
            elif initialize == "random":
                key = random.PRNGKey(random_seed)
                w0 = 0.01 * random.normal(key, shape=(self.w_sta.shape[0],)).flatten()
                p0.update({"w": w0})

        if "intercept" not in dict_keys:
            p0.update({"intercept": jnp.array([0.0])})

        if "R" not in dict_keys and self.fit_R:
            p0.update({"R": jnp.array([1.0])})

        if "h" not in dict_keys:
            if initialize is None and self.h_mle is not None:
                p0.update({"h": self.h_mle})

            elif initialize == "random" and self.h_mle is not None:
                key = random.PRNGKey(random_seed)
                h0 = 0.01 * random.normal(key, shape=(self.h_mle.shape[0],)).flatten()
                p0.update({"h": h0})
            else:
                p0.update({"h": None})

        if "nl_params" not in dict_keys:
            if self.nl_params is not None:
                p0.update({"nl_params": self.nl_params})
            else:
                p0.update({"nl_params": None})

        if extra is not None:

            if self.h_mle is not None:
                yh = jnp.array(
                    build_design_matrix(
                        extra["y"][:, jnp.newaxis], self.yh.shape[1], shift=1
                    )
                )
                extra.update({"yh": yh})

            extra = {key: jnp.array(extra[key]) for key in extra.keys()}

        # store optimized parameters
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
        self.R = self.p_opt["R"] if fit_R else jnp.array([1.0])

        if fit_linear_filter:
            self.w_opt = self.p_opt["w"]

        if fit_history_filter:
            self.h_opt = self.p_opt["h"]

        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt["nl_params"]

        if fit_intercept:
            self.intercept = self.p_opt["intercept"]

    def predict(self, X, y=None, p=None):
        """

        Parameters
        ==========

        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : None or array_like, shape (n_samples, )
            Recorded response. Needed when post-spike filter is fitted.

        p : None or dict
            Model parameters. Only needed if model performance is monitored
            during training.

        """

        extra = {"X": X, "y": y}
        if self.h_mle is not None:

            if y is None:
                raise ValueError("`y` is needed for calculating response history.")

            yh = jnp.array(
                build_design_matrix(
                    extra["y"][:, jnp.newaxis], self.yh.shape[1], shift=self.shift_h
                )
            )
            extra.update({"yh": yh})

        params = self.p_opt if p is None else p
        y_pred = self.forwardpass(params, extra=extra)

        return y_pred

    @staticmethod
    def compute_score(y, y_pred, metric):
        if metric == "r2":
            return r2(y, y_pred)
        elif metric == "mse":
            return mse(y, y_pred)
        elif metric == "corrcoef":
            return corrcoef(y, y_pred)
        else:
            print(f"Metric `{metric}` is not supported.")

    def score(self, X, y, p=None, metric="corrcoef"):
        """Performance measure."""
        y_pred = self.predict(X, y, p)

        if metric == "nll":
            return self.cost(p=self.p_opt, extra={"X": X, "y": y}, precomputed=y_pred)
        else:
            return self.compute_score(y, y_pred, metric)

    def cost(self, p, extra, precomputed):
        raise NotImplementedError()


class splineBase(Base):
    """
    Base class for spline-based GLMs.
    """

    def __init__(self, X, y, dims, df, smooth="cr", compute_mle=False, **kwargs):
        """

        Parameters
        ==========
        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : array_like, shape (n_samples, )
            Recorded response.

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sx, sy].

        df : list or array_like, shape (ndims, )
            Degree of freedom, or the number of basis used for each RF dimension.

        smooth : str
            Type of basis.
            * cr: natural cubic spline (default)
            * cc: cyclic cubic spline
            * bs: B-spline
            * tp: thin plate spine

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """

        super().__init__(X, y, dims, compute_mle, **kwargs)

        # Optimization
        self.bh_opt = None
        self.b_opt = None
        self.extra = None
        self.h_spl = None
        self.bh_spl = None
        self.yS = None
        self.Sh = None

        # Parameters
        self.df = df  # number basis / degree of freedom
        self.smooth = smooth  # type of basis

        S = jnp.array(build_spline_matrix(self.dims, df, smooth))  # for w

        if self.n_c > 1:
            XS = jnp.dstack([self.X[:, :, i] @ S for i in range(self.n_c)]).reshape(
                self.n_samples, -1
            )
        else:
            XS = self.X @ S

        self.S = S  # spline matrix
        self.XS = XS

        self.n_b = S.shape[1]  # num:ber of spline coefficients

        # compute spline-based maximum likelihood
        self.b_spl = jnp.linalg.lstsq(XS.T @ XS, XS.T @ y, rcond=None)[0]

        if self.n_c > 1:
            self.w_spl = S @ self.b_spl.reshape(self.n_b, self.n_c)
        else:
            self.w_spl = S @ self.b_spl

    def initialize_Cinv(self, params):

        df = self.df
        Cinvs = [
            smoothness_kernel(
                [
                    params[i],
                ],
                df[i],
            )[1]
            for i in range(len(df))
        ]

        if len(Cinvs) == 1:
            self.Cinv = Cinvs[0]
        elif len(Cinvs) == 2:
            self.Cinv = jnp.kron(*Cinvs)
        else:
            self.Cinv = jnp.kron(Cinvs[0], jnp.kron(Cinvs[1], Cinvs[2]))

    def cost(self, b, extra):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def initialize_history_filter(self, dims, df, smooth="cr", shift=1):
        """

        Parameters
        ==========

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]

        df : list or array_list
            Number of basis.

        smooth : str
            Type of basis.

        shift : int
            Should be 1 or larger.

        """

        y = self.y
        Sh = jnp.array(
            build_spline_matrix(
                [
                    dims,
                ],
                [
                    df,
                ],
                smooth,
            )
        )  # for h
        yh = jnp.array(
            build_design_matrix(self.y[:, jnp.newaxis], Sh.shape[0], shift=shift)
        )
        yS = yh @ Sh

        self.shift_h = shift
        self.yh = jnp.array(yh)
        self.Sh = Sh  # spline basis for spike-history
        self.yS = yS
        self.bh_spl = jnp.linalg.solve(yS.T @ yS, yS.T @ y)
        self.h_spl = Sh @ self.bh_spl

    # TODO: fix Docstring
    # noinspection PyIncorrectDocstring
    def fit(
        self,
        p0=None,
        extra=None,
        initialize="random",
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
        """

        Parameters
        ==========

        p0 : dict
            * 'b': Initial spline coefficients.
            * 'bh': Initial response history filter coefficients

        initialize : None or str
            Parametric initialization.
            * if `initialize=None`, `b` will be initialized by b_spl.
            * if `initialize='random'`, `b` will be randomly initialized.

        num_iters : int
            Max number of optimization iterations.

        metric : None or str
            Extra cross-validation metric. Default is `None`. Or
            * 'mse': mean squared error
            * 'r2': R2 score
            * 'corrcoef': Correlation coefficient

        alpha : float, from 0 to 1.
            Elastic net parameter, balance between L1 and L2 regularization.
            * 0.0 -> only L2
            * 1.0 -> only L1

        beta : float
            Elastic net parameter, overall weight of regularization for receptive field.

        step_size : float
            Initial step size for JAX optimizer.

        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically
            increases or stop increases for tolerance=n steps.

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.metric = metric

        self.alpha = alpha
        self.beta = (
            beta  # elastic net parameter - global penalty weight for linear filter
        )
        self.num_iters = num_iters

        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        # initial parameters

        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()
        if "b" not in dict_keys:
            if initialize is None:
                p0.update({"b": self.b_spl})
            else:
                if initialize == "random":
                    key = random.PRNGKey(random_seed)
                    b0 = (
                        0.01
                        * random.normal(key, shape=(self.n_b * self.n_c,)).flatten()
                    )
                    p0.update({"b": b0})

        if "intercept" not in dict_keys:
            p0.update({"intercept": jnp.array([0.0])})

        if "R" not in dict_keys:
            p0.update({"R": jnp.array([1.0])})

        if "bh" not in dict_keys:
            if initialize is None and self.bh_spl is not None:
                p0.update({"bh": self.bh_spl})
            elif initialize == "random" and self.bh_spl is not None:
                key = random.PRNGKey(random_seed)
                bh0 = 0.01 * random.normal(key, shape=(len(self.bh_spl),)).flatten()
                p0.update({"bh": bh0})
            else:
                p0.update({"bh": None})

        if "nl_params" not in dict_keys:
            if self.nl_params is not None:
                p0.update({"nl_params": self.nl_params})
            else:
                p0.update({"nl_params": None})

        if extra is not None:

            if self.n_c > 1:
                XS_ext = jnp.dstack(
                    [extra["X"][:, :, i] @ self.S for i in range(self.n_c)]
                ).reshape(extra["X"].shape[0], -1)
                extra.update({"XS": XS_ext})
            else:
                extra.update({"XS": extra["X"] @ self.S})

            if self.h_spl is not None:
                yh_ext = jnp.array(
                    build_design_matrix(
                        extra["y"][:, jnp.newaxis], self.Sh.shape[0], shift=1
                    )
                )
                yS_ext = yh_ext @ self.Sh
                extra.update({"yS": yS_ext})

            extra = {key: jnp.array(extra[key]) for key in extra.keys()}

            self.extra = extra  # store for cross-validation

        # store optimized parameters
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
        self.R = self.p_opt["R"] if fit_R else jnp.array([1.0])

        if fit_linear_filter:
            self.b_opt = self.p_opt["b"]  # optimized RF basis coefficients
            if self.n_c > 1:
                self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_c)
            else:
                self.w_opt = self.S @ self.b_opt  # optimized RF

        if fit_history_filter:
            self.bh_opt = self.p_opt["bh"]
            self.h_opt = self.Sh @ self.bh_opt

        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt["nl_params"]

        if fit_intercept:
            self.intercept = self.p_opt["intercept"]

    def predict(self, X, y=None, p=None):
        """

        Parameters
        ==========

        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : None or array_like, shape (n_samples, )
            Recorded response. Needed when post-spike filter is fitted.

        p : None or dict
            Model parameters. Only needed if model performance is monitored
            during training.

        """

        if self.n_c > 1:
            XS = jnp.dstack([X[:, :, i] @ self.S for i in range(self.n_c)]).reshape(
                X.shape[0], -1
            )
        else:
            XS = X @ self.S

        extra = {"X": X, "XS": XS, "y": y}

        if self.h_spl is not None:

            if y is None:
                raise ValueError("`y` is needed for calculating response history.")

            yh = jnp.array(
                build_design_matrix(
                    extra["y"][:, jnp.newaxis], self.Sh.shape[0], shift=self.shift_h
                )
            )
            yS = yh @ self.Sh
            extra.update({"yS": yS})

        params = self.p_opt if p is None else p
        y_pred = self.forwardpass(params, extra=extra)

        return y_pred

    def compute_history_output(self, yS=None, p=None):
        if self.fit_history_filter:
            history_output = yS @ p["bh"]
        else:
            if self.bh_opt is not None:
                history_output = yS @ self.bh_opt
            elif self.bh_spl is not None:
                history_output = yS @ self.bh_spl
            else:
                history_output = 0.0

        return history_output


class interp1d:
    """
    1D linear interpolation.
    usage:
        x = jnp.linspace(-5, 5, 10)
        y = jnp.cos(x)
        f = interp1d(x, y)
        new_x = jnp.linspace(-5, 5, 100)
        new_y = f(new_x)
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.slopes = jnp.diff(y) / jnp.diff(x)

    def __call__(self, x_new):
        i = jnp.searchsorted(self.x, x_new) - 1
        i = jnp.where(i == -1, 0, i)
        i = jnp.where(i == len(self.x) - 1, -1, i)

        return self.y[i] + self.slopes[i] * (x_new - self.x[i])


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
