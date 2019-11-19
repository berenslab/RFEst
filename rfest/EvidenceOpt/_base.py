import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import mean_squared_error

__all__ = ['EmpiricalBayes']

class EmpiricalBayes:

    """
    
    Base class for evidence optimization methods.

    """

    def __init__(self, X, y, dims, compute_mle=True):
        
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
        self.X = np.array(X) # stimulus design matrix
        self.y = np.array(y) # response 
        
        self.dims = dims # assumed order [t, y, x]
        self.n_samples, self.n_features = X.shape

        self.XtX = X.T @ X
        self.XtY = X.T @ y
        self.YtY = y.T @ y

        if compute_mle:
            self.w_sta = self.XtY
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)
                         #maximum likelihood estimation

    def _make_1D_covariance(self, params, ncoeff):

        """
        
        Placeholder for class method `_make_1d_covariance`.
        If you design a new prior, fill it in this method. 

        Parameters
        ==========
        params : list or array_like, shape (n_hyperparams_1d,)
            Hyperparameters in one dimension.

        ncoeff : int
            Number of coefficient in one dimension.

        """

        pass
    
    def update_C_prior(self, params):

        """
        
        Using kronecker product to construct high-dimensional prior covariance.

        Given RF dims = [t, y, x], the prior covariance:

            C = kron(Ct, kron(Cy, Cx))
            Cinv = kron(Ctinv, kron(Cyinv, Cxinv))
            
        """

        nh = self.n_hyperparams_1d

        rho = params[1]
        params_time = params[ 2+0*nh : 2+1*nh ]

        # Covariance Matrix in Time
        C_t, C_t_inv = self._make_1D_covariance(params_time, self.dims[0])

        if len(self.dims) == 1:

            C, C_inv = C_t, C_t_inv

        elif len(self.dims) ==2:

            # Covariance Matrix in Space 
            params_space = params[ 2+1*nh : 2+2*nh ]
            C_s, C_s_inv = self._make_1D_covariance(params_space, self.dims[1])

            # Build 2D Covariance Matrix 
            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)  

        elif len(self.dims) ==3:

            # Covariance Matrix in Space 
            params_spacey = params[ 2+1*nh : 2+2*nh ]
            params_spacex = params[ 2+2*nh : 2+3*nh ]
        
            C_sy, C_sy_inv = self._make_1D_covariance(params_spacey, self.dims[1])
            C_sx, C_sx_inv = self._make_1D_covariance(params_spacex, self.dims[2])
            
            C_s = np.kron(C_sy, C_sx)
            C_s_inv = np.kron(C_sy_inv, C_sx_inv)

            # Build 3D Covariance Matrix
            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)  

        return C, C_inv
    
    def update_C_posterior(self, params, C_prior_inv):

        """

        See eq(9) in Park & Pillow (2011).

        """

        sigma = params[0]

        C_post_inv = self.XtX / sigma**2 + C_prior_inv
        C_post = np.linalg.inv(C_post_inv)
        
        m_post = C_post @ self.XtY / (sigma**2)
        
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
        pass

    def print_progress(self, i, params, cost):
        pass 
    
    def optimize_params(self, initial_params, num_iters, step_size, tolerance, verbal):

        """
        
        Perform gradient descent using JAX optimizers. 

        """
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(initial_params)
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            g = grad(self.negative_log_evidence)(p)
            return opt_update(i, g, opt_state)

        cost_list = []
        params_list = []

        if verbal:
            self.print_progress_header(initial_params)

        for i in range(num_iters):
            
            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.negative_log_evidence(params_list[-1]))

            if verbal:
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
    
    def fit(self, initial_params, num_iters=20, step_size=1e-2, tolerance=10, verbal=True):


        """
        Parameters
        ==========

        initial_params : list or array_like, shape (n_hypyerparams_1d * ndim, )
            Initial (hyper)parameters

        num_iters : int
            Number of iterations. 

        step_size : float
            
        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically 
            increases or stop increases for tolerance=n steps.

        verbal: int
            When `verbal=0`, progress is not printed. When `verbal=n`,
            progress will be printed in every n steps.

        """

        self.num_iters = num_iters       
        self.optimized_params = self.optimize_params(initial_params, num_iters, step_size, tolerance, verbal)

        (optimized_C_prior, 
         optimized_C_prior_inv) = self.update_C_prior(self.optimized_params)
        
        (optimized_C_post, 
         optimized_C_post_inv, 
         optimized_m_post) = self.update_C_posterior(self.optimized_params,
                                                   optimized_C_prior_inv)
        
        self.optimized_C_prior = optimized_C_prior
        self.optimized_C_post = optimized_C_post
        self.w_opt = optimized_m_post
        
    def _rcv(self, w, wSTA_test, X_test, y_test):

        """Relative Mean Squared Error"""

        a = mean_squared_error(y_test, X_test @ w)  
        b = mean_squared_error(y_test, X_test @ wSTA_test)

        return a - b

    def measure_prediction_performance(self, X_test, y_test):

        wSTA_test = np.linalg.solve(X_test.T @ X_test, X_test.T @ y_test)

        w = self.w_opt.ravel()

        return self._rcv(w, wSTA_test, X_test, y_test)
