import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import mean_squared_error
from ._base import EmpiricalBayes
from .._priors import sparsity_kernel
from .._splines import build_spline_matrix

__all__ = ['sARD']

class sARD:

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False, **kwargs):

        self.dims = dims
        self.n_samples, self.n_features = X.shape

        S = np.array(build_spline_matrix(dims, df, smooth=smooth))
        Z = X @ S

        self.XtY = X.T @ y
        if np.array_equal(y, y.astype(bool)): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)

        if compute_mle: #maximum likelihood estimation
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)

        self.X = np.array(X)
        self.y = np.array(y)

        self.n_b = S.shape[1]
        self.S = S
        self.Z = Z
        self.ZtZ = Z.T @ Z
        self.ZtY = Z.T @ y
        self.YtY = y.T @ y

        self.b_spl = np.linalg.solve(Z.T @ Z, Z.T @ y)
        self.w_spl = S @ self.b_spl

    def update_C_prior(self, params):

        rho = params[1]
        theta = params[2:]
        n_b = self.n_b

        C, C_inv = sparsity_kernel(theta, n_b)
        C *= rho
        C_inv /= rho

        return C, C_inv

    def update_C_posterior(self, params, C_prior_inv):

        """
        See eq(9) in Park & Pillow (2011).
        """

        sigma = params[0]

        C_post_inv = self.ZtZ / sigma**2 + C_prior_inv
        C_post = np.linalg.inv(C_post_inv)

        m_post = C_post @ self.ZtY / (sigma**2)

        return C_post, C_post_inv, m_post

    def negative_log_evidence(self, params):

        """

        See eq(10) in Park & Pillow (2011).
        """

        sigma = params[0]

        (C_prior, C_prior_inv) = self.update_C_prior(params)

        (C_post, C_post_inv, m_post) = self.update_C_posterior(params, C_prior_inv)

        t0 = np.log(np.abs(2 * np.pi * sigma**2)) * self.n_samples
        t1 = np.linalg.slogdet(C_prior @ C_post_inv)[1]
        t2 = -m_post.T @ C_post @ m_post
        t3 = self.YtY / sigma**2

        return 0.5 * (t0 + t1 + t2 + t3)

    def print_progress_header(self, params):
        print('Iter\tcost')

    def print_progress(self, i, params, cost):
        print('{0:4d}\t{1:1.3f}'.format(
                i, cost))

    def optimize_params(self, p0, num_iters, step_size, tolerance, verbal):

        """

        Perform gradient descent using JAX optimizers.
        """

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)

        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            g = grad(self.negative_log_evidence)(p)
            return opt_update(i, g, opt_state)

        cost_list = []
        params_list = []

        if verbal:
            self.print_progress_header(p0)

        for i in range(num_iters):

            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.negative_log_evidence(params_list[-1]))

            if verbal:
                if i % verbal == 0:
                    self.print_progress(i, params_list[-1], cost_list[-1])

            if len(params_list) > tolerance:

                if np.all((np.array(cost_list[1:])) - np.array(cost_list[:-1]) > 0 ):
                    params = params_list[0]
                    if verbal:
                        print('Stop: cost has been monotonically increasing for {} steps.'.format(tolerance))
                    break
                elif np.all(np.array(cost_list[:-1]) - np.array(cost_list[1:]) < 1e-5):
                    params = params_list[-1]
                    if verbal:
                        print('Stop: cost has been stop changing for {} steps.'.format(tolerance))
                    break
                else:
                    params_list.pop(0)
                    cost_list.pop(0)

        else:

            params = params_list[-1]
            if verbal:
                print('Stop: reached {0} steps, final cost={1:.5f}.'.format(num_iters, cost_list[-1]))

        return params

    def fit(self, p0=None, num_iters=20, step_size=1e-2, tolerance=10, verbal=True, random_seed=1990):


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
        verbal: int
            When `verbal=0`, progress is not printed. When `verbal=n`,
            progress will be printed in every n steps.
        """

        if p0 is None:
            key = random.PRNGKey(random_seed)
            p0 = random.normal(key, shape=(self.n_b+2, )).flatten()
        
        self.p0 = np.array(p0)
        self.num_iters = num_iters
        self.optimized_params = self.optimize_params(self.p0, num_iters, step_size, tolerance, verbal)

        (optimized_C_prior,
         optimized_C_prior_inv) = self.update_C_prior(self.optimized_params)

        (optimized_C_post,
         optimized_C_post_inv,
         optimized_m_post) = self.update_C_posterior(self.optimized_params,
                                                   optimized_C_prior_inv)

        self.optimized_C_prior = optimized_C_prior
        self.optimized_C_post = optimized_C_post
        self.b_opt = optimized_m_post
        self.w_opt = self.S @ self.b_opt
