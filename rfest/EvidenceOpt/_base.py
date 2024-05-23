import jax.numpy as jnp
from jax import config, grad, jit

try:
    from jax.example_libraries import optimizers
except ImportError:
    from jax.experimental import optimizers

from rfest.priors import *

config.update("jax_enable_x64", True)

__all__ = ["EmpiricalBayes"]


class EmpiricalBayes:
    """

    Base class for evidence optimization methods.

    """

    def __init__(self, X, y, dims, compute_mle=False, **kwargs):
        """

        Initializing the `EmpiricalBayes` class, sufficient statistics are calculated.

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

        # Wrapped with JAX DeviceArray
        self.w_opt = None
        self.optimized_C_post = None
        self.optimized_C_prior = None
        self.optimized_params = None
        self.num_iters = None
        self.p0 = None

        self.X = jnp.array(X)  # stimulus design matrix
        self.y = jnp.array(y)  # response

        self.dims = dims  # assumed order [t, y, x]
        self.n_samples, self.n_features = X.shape

        self.XtX = X.T @ X
        self.XtY = X.T @ y
        self.YtY = y.T @ y

        if jnp.array_equal(y, y.astype(int)):  # if y is spikes
            self.w_sta = self.XtY / sum(y)
        else:  # if y is not spike
            self.w_sta = self.XtY / len(y)

        if compute_mle:  # maximum likelihood estimation
            self.w_mle = jnp.linalg.solve(self.XtX, self.XtY)

        # methods
        self.time = kwargs["time"] if "time" in kwargs.keys() else None
        self.space = kwargs["space"] if "space" in kwargs.keys() else None
        self.n_hp_time = kwargs["n_hp_time"] if "n_hp_time" in kwargs.keys() else None
        self.n_hp_space = (
            kwargs["n_hp_space"] if "n_hp_space" in kwargs.keys() else None
        )

    def cov1d_time(self, params, ncoeff):
        """

        Placeholder for class method `cov1d` in time.
        If you design a new prior, just overwrite this method.
        Same for `cov1d` in space.

        Parameters
        ==========
        params : list or array_like, shape (n_hyperparams_1d,)
            Hyperparameters in one dimension.

        ncoeff : int
            Number of coefficient in one dimension.

        """

        if self.time == "asd":
            return smoothness_kernel(params, ncoeff)

        elif self.time == "ald":
            return locality_kernel(params, ncoeff)

        elif self.time == "ard":
            return sparsity_kernel(params, ncoeff)

        elif self.time == "ridge":
            return ridge_kernel(params, ncoeff)

        else:
            raise NotImplementedError(
                f"`{self.time}` is not supported."
                + "You can implement it yourself by overwriting the `self.cov1d_time()` method."
            )

    def cov1d_space(self, params, ncoeff):

        if self.space == "asd":
            return smoothness_kernel(params, ncoeff)

        elif self.space == "ald":
            return locality_kernel(params, ncoeff)

        elif self.space == "ard":
            return sparsity_kernel(params, ncoeff)

        elif self.space == "ridge":
            return ridge_kernel(params, ncoeff)

        else:
            raise NotImplementedError(
                f"`{self.space}` is not supported."
                + "You can implement it yourself by overwriting the `self.cov1d_space()` method."
            )

    def update_C_prior(self, params):
        """

        Using kronecker product to construct high-dimensional prior covariance.

        Given RF dims = [t, y, x], the prior covariance:

            C = kron(Ct, kron(Cy, Cx))
            Cinv = kron(Ctinv, kron(Cyinv, Cxinv))

        """

        n_hp_time = self.n_hp_time
        n_hp_space = self.n_hp_space

        rho = params[1]
        params_time = params[2 : 2 + n_hp_time]

        # Covariance Matrix in Time
        C_t, C_t_inv = self.cov1d_time(params_time, self.dims[0])

        if len(self.dims) == 1:

            C, C_inv = rho * C_t, (1 / rho) * C_t_inv

        elif len(self.dims) == 2:

            # Covariance Matrix in Space
            params_space = params[2 + n_hp_time : 2 + n_hp_time + n_hp_space]
            C_s, C_s_inv = self.cov1d_space(params_space, self.dims[1])

            # Build 2D Covariance Matrix
            C = rho * jnp.kron(C_t, C_s)
            C_inv = (1 / rho) * jnp.kron(C_t_inv, C_s_inv)

        elif len(self.dims) == 3:

            # Covariance Matrix in Space
            params_spacey = params[2 + n_hp_time : 2 + n_hp_time + n_hp_space]
            params_spacex = params[2 + n_hp_time + n_hp_space :]

            C_sy, C_sy_inv = self.cov1d_space(params_spacey, self.dims[1])
            C_sx, C_sx_inv = self.cov1d_space(params_spacex, self.dims[2])

            C_s = jnp.kron(C_sy, C_sx)
            C_s_inv = jnp.kron(C_sy_inv, C_sx_inv)

            # Build 3D Covariance Matrix
            C = rho * jnp.kron(C_t, C_s)
            C_inv = (1 / rho) * jnp.kron(C_t_inv, C_s_inv)

        else:
            raise NotImplementedError(len(self.dims))

        return C, C_inv

    def update_C_posterior(self, params, C_prior_inv):
        """

        See eq(9) in Park & Pillow (2011).

        """

        sigma = params[0]

        C_post_inv = self.XtX / sigma**2 + C_prior_inv
        C_post = jnp.linalg.pinv(C_post_inv)

        m_post = C_post @ self.XtY / (sigma**2)

        return C_post, C_post_inv, m_post

    def negative_log_evidence(self, params):
        """

        See eq(10) in Park & Pillow (2011).

        """

        sigma = params[0]

        (C_prior, C_prior_inv) = self.update_C_prior(params)

        (C_post, C_post_inv, m_post) = self.update_C_posterior(params, C_prior_inv)

        t0 = jnp.log(jnp.abs(2 * jnp.pi * sigma**2)) * self.n_samples
        t1 = jnp.linalg.slogdet(C_prior @ C_post_inv)[1]
        t2 = -m_post.T @ C_post @ m_post
        t3 = self.YtY / sigma**2

        return 0.5 * (t0 + t1 + t2 + t3)

    @staticmethod
    def print_progress_header(params):
        print("Iter\tcost")

    @staticmethod
    def print_progress(i, params, cost):
        print("{0:4d}\t{1:1.3f}".format(i, cost))

    def optimize_params(self, p0, num_iters, step_size, tolerance, verbose, atol=1e-5):
        """

        Perform gradient descent using JAX optimizers.

        """

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)

        @jit
        def step(_i, _opt_state):
            p = get_params(_opt_state)
            g = grad(self.negative_log_evidence)(p)
            return opt_update(_i, g, _opt_state)

        cost_list = []
        params_list = []

        if verbose:
            self.print_progress_header(p0)

        for i in range(num_iters):

            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.negative_log_evidence(params_list[-1]))

            if verbose:
                if i % verbose == 0:
                    self.print_progress(i, params_list[-1], cost_list[-1])

            if len(params_list) > tolerance:

                if jnp.all((jnp.array(cost_list[1:])) - jnp.array(cost_list[:-1]) > 0):
                    params = params_list[0]
                    if verbose:
                        print(
                            "Stop: cost has been monotonically increasing for {} steps.".format(
                                tolerance
                            )
                        )
                    break
                elif jnp.all(
                    jnp.array(cost_list[:-1]) - jnp.array(cost_list[1:]) < atol
                ):
                    params = params_list[-1]
                    if verbose:
                        print(
                            "Stop: cost has been stop changing for {} steps.".format(
                                tolerance
                            )
                        )
                    break
                else:
                    params_list.pop(0)
                    cost_list.pop(0)

        else:

            params = params_list[-1]
            if verbose:
                print(
                    "Stop: reached {0} steps, final cost={1:.5f}.".format(
                        num_iters, cost_list[-1]
                    )
                )

        return params

    def fit(self, p0, num_iters=20, step_size=1e-2, tolerance=10, verbose=True):
        """
        Parameters
        ==========

        p0 : list or array_like, shape (n_hp_time, ) for 1D
                                       (n_hp_time + n_hp_space) for 2D
                                       (n_hp_time + n_hp_space*2) for 3D
            Initial Gaussian prior hyperparameters

        num_iters : int
            Max number of optimization iterations.

        step_size : float
            Initial step size for Jax optimizer.

        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically
            increases or stop increases for tolerance=n steps.

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.p0 = jnp.array(p0)
        self.num_iters = num_iters
        self.optimized_params = self.optimize_params(
            self.p0, num_iters, step_size, tolerance, verbose
        )

        (optimized_C_prior, optimized_C_prior_inv) = self.update_C_prior(
            self.optimized_params
        )

        (optimized_C_post, optimized_C_post_inv, optimized_m_post) = (
            self.update_C_posterior(self.optimized_params, optimized_C_prior_inv)
        )

        self.optimized_C_prior = optimized_C_prior
        self.optimized_C_post = optimized_C_post
        self.w_opt = optimized_m_post
