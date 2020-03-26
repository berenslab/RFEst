import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import mean_squared_error

from .._splines import build_spline_matrix

__all__ = ['splineBase']

class splineBase:

    """
    
    Base class for spline-based GLMs. 
    
    """

    def __init__(self, X, y, dims, df, smooth='cr', add_intercept=False, compute_mle=False, **kwargs):

        """
        
        Parameters
        ==========
        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : array_like, shape (n_samples, )
            Recorded response

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sy, sx]

        df : int
            Degree of freedom for spline /smooth basis. 

        smooth : str
            Spline or smooth to be used. Current supported methods include:
            * `bs`: B-spline
            * `cr`: Cubic Regression spline
            * `tp`: (Simplified) Thin Plate regression spline 

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """ 

        # store meta data

        self.dims = dims # assumed order [t, y, x]
        self.ndim = len(dims)
        self.n_samples, self.n_features = X.shape
        
        # compute sufficient statistics

        S = np.array(build_spline_matrix(dims, df, smooth)) # spline matrix
        
        if add_intercept:
            X = np.hstack([np.ones(self.n_samples)[:, np.newaxis], X])
            S = np.vstack([np.ones(S.shape[1]), S])

        XS = X @ S
        self.XtY = X.T @ y
        if np.array_equal(y, y.astype(bool)): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)
        
        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)

        self.X = np.array(X) # stimulus design matrix
        self.y = np.array(y) # response

        self.S = S # spline matrix
        self.XS = XS 
        self.n_b = S.shape[1] # num:ber of spline coefficients
        
        # compute spline-based maximum likelihood 
        self.b_spl = np.linalg.solve(XS.T @ XS, XS.T @ y)
        self.w_spl = S @ self.b_spl

        # store more meta data
        
        self.df = df 
        self.smooth = smooth   

        self.dt = kwargs['dt'] if 'dt' in kwargs.keys() else 1 # time bin size (for spike data)


    def cost(self, b):
        pass


    def optimize_params(self, p0, num_iters, step_size, tolerance, verbal):

        """

        Gradient descent using JAX optimizer.
        
        """
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            g = grad(self.cost)(p)
            return opt_update(i, g, opt_state)

        cost_list = []
        params_list = []    

        if verbal:
            print('{0}\t{1}\t'.format('Iter', 'Cost'))

        for i in range(num_iters):
            
            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.cost(params_list[-1]))
            
            if verbal:
                if i % int(verbal) == 0:
                    print('{0}\t{1:.3f}\t'.format(i,  cost_list[-1]))
            
            if len(params_list) > tolerance:
                
                if np.all((np.array(cost_list[1:])) - np.array(cost_list[:-1]) > 0 ):
                    params = params_list[0]
                    if verbal:
                        print('Stop at {} steps: cost has been monotonically increasing for {} steps.'.format(i, tolerance))
                    break
                elif np.all(np.array(cost_list[:-1]) - np.array(cost_list[1:]) < 1e-5):
                    params = params_list[-1]
                    if verbal:
                        print('Stop at {} steps: cost has been changing less than 1e-5 for {} steps.'.format(i, tolerance))
                    break                    
                else:
                    params_list.pop(0)
                    cost_list.pop(0)     
        else:
            params = params_list[-1]
            if verbal:
                print('Stop: reached {0} steps, final cost={1:.5f}.'.format(num_iters, cost_list[-1]))
            
        return params


    def fit(self, p0=None, num_iters=5, alpha=0.5, lambd=0.5,
            step_size=1e-2, tolerance=10, verbal=1):
            
        """

        Parameters
        ==========

        p0 : array_like, shape (n_b, ) or (n_b, n_subunits)
            Initial spline coefficients.

        num_iters : int
            Max number of optimization iterations.
        
        alpha : float, from 0 to 1.
            Elastic net parameter, balance between L1 and L2 regulization.
            * 0.0 -> only L2
            * 1.0 -> only L1

        lambd : float
            Elastic net parameter, overall weight of regulization.

        step_size : float
            Initial step size for JAX optimizer.

        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically 
            increases or stop increases for tolerance=n steps.

        verbal: int
            When `verbal=0`, progress is not printed. When `verbal=n`,
            progress will be printed in every n steps.

        """

        self.lambd = lambd
        self.alpha = alpha 
        self.num_iters = num_iters   
        
        if p0 is None: # if p0 is not provided, initialize it with spline MLE.
            p0 = self.b_spl
        
        self.b_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        self.w_opt = self.S @ self.b_opt


    def _rcv(self, w, wSTA_test, X_test, y_test):

        """Relative Mean Squared Error"""

        a = mean_squared_error(y_test, X_test @ w)
        b = mean_squared_error(y_test, X_test @ wSTA_test)

        return a - b
        

    def measure_prediction_performance(self, X_test, y_test):

        wSTA_test = np.linalg.solve(X_test.T @ X_test, X_test.T @ y_test)

        w = self.w_opt.ravel()

        return self._rcv(w, wSTA_test, X_test, y_test)
