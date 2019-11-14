import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

import patsy

__all__ = ['splineLG']

class splineLG:

    """

    Linear Model with sufficiently flexible spline basis under Gaussian Noise. 

    """
    
    def __init__(self, X, y, dims, df, degree=3, smooth='cr', compute_mle=True):

        """
        
        Initializing the `splineLG` class, some sufficient statistics are calculated optionally.

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
            
        degree: int
            B-spline order, only used when `smooth=bs`

        smooth : str
            Spline or smooth to be used. Current supported methods include:
            * `bs`: B-spline
            * `cr`: Cubic Regression spline
            * `tp`: (Simplified) Thin Plate regression spline 

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """
        
        self.X = X # stimulus design matrix
        self.y = y # response 
        
        self.dims = dims # assumed order [t, y, x]
        self.ndim = len(dims)
        self.n_samples, self.n_features = X.shape

        if compute_mle:
            self.XtX = X.T @ X
            self.w_sta = X.T @ y
            self.w_mle = np.linalg.solve(self.XtX, self.w_sta)
        
        S = self._build_spline_matrix(df, smooth)
        self.S = S
        self.XS = X @ S        
        
        # compute spline-based maximum likelihood 
        self.b_spl = np.linalg.solve(self.XS.T @ self.XS, S.T @ X.T @ y)
        self.w_spl = S @ self.b_spl

        # store meta data
        self.df = df 
        self.smooth = smooth   

    def _build_spline_matrix(self, df, smooth):
        
        # initialize list of degree of freedom for each dimension
        if np.ndim(df) != 0 and len(df) != self.ndim:
            raise ValueError("`df` must be an integer or an array the same length as `dims`")
        elif np.ndim(df) == 0:
            df = np.ones(self.ndim) * df

        # choose smooth basis
        if smooth =='bs':
            basis = bs
        elif smooth == 'cr':
            basis = patsy.cr
        elif smooth == 'tp':
            basis = tp
        else:
            raise ValueError("Input method `{}` is not supported.".format(smooth))
        
        # build spline matrix
        if self.ndim == 1:
        
            S = basis(np.arange(self.dims[0]), df[0])
            
        elif self.ndim == 2:
        
            g0, g1 = np.meshgrid(np.arange(self.dims[0]), np.arange(self.dims[1]), indexing='ij')
            S = te(basis(g0.ravel(), df[0]), 
                   basis(g1.ravel(), df[1]))
            
        elif self.ndim == 3:
            
            g0, g1, g2 = np.meshgrid(np.arange(self.dims[0]), 
                                     np.arange(self.dims[1]), 
                                     np.arange(self.dims[2]), indexing='ij')
            S = te(basis(g0.ravel(), df[0]), 
                   basis(g1.ravel(), df[1]), 
                   basis(g2.ravel(), df[2]))
            
        return S
    
    def cost(self, B):
        
        XS = self.XS
        y = self.y    
        
        mse = np.sum((y - XS @ B)**2) / len(y)

        if self.lambd:
            mse += self.lambd * ((1 - self.alpha) * np.linalg.norm(B, 2) + self.alpha * np.linalg.norm(B, 1)) 
    
        return mse
    
    def optimize_params(self, initial_params, num_iters, step_size, tolerance, verbal):
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(initial_params)
        
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
                print('Stop: reached {} steps, final cost={}.'.format(num_iters, cost_list[-1]))
            
        return params      
    
    def fit(self, initial_params=None, num_iters=5, alpha=0.5, lambd=0.5,
            step_size=1e-2, tolerance=10, verbal=True, random_seed=2046):

        self.lambd = lambd # elastic net parameter
        self.alpha = alpha # elastic net parameter
        self.num_iters = num_iters   
        
        if initial_params is None:
        
            B = self.b_spl
        
        self.b_opt = self.optimize_params(B, num_iters, step_size, tolerance, verbal)
        self.w_opt = self.S @ self.b_opt 
        
def tp(x, df):

    """
    
    Simplified implementation of the truncated Thin Plate (TP) regression spline.

    """
    
    def eta(r):
        return r**2 * np.log(r + 1e-10)

    E = eta(np.abs(x.ravel() - x.ravel().reshape(-1,1)))
    U, _, _ = np.linalg.svd(E)
    S = U[:, :int(df)]
    
    return S / np.linalg.norm(S)

def bs(x, df, degree):
    
    from scipy.interpolate import BSpline
    
    def _sort_all_knots(x, df, degree=3):

        order = degree + 1

        n_inner_knots = df - order

        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1] * 100
        inner_knots = np.percentile(x, knot_quantiles)

        all_knots = np.concatenate(([np.min(x), np.max(x)] * order,
                                    inner_knots))
        all_knots.sort()

        return all_knots

    knots = _sort_all_knots(x, df, degree)
    n_bases = len(knots) - (degree + 1) 
    coeff = np.eye(n_bases)
    S = np.vstack([BSpline(knots, coeff[i], degree)(x) for i in range(n_bases)]).T
    
    return S

def te(*args):

    """

    Tensor Product smooth. Numericially the same as `patsy.te`.

    """
    
    As = list(args)
    
    def columnwise_product(A2, A1):
        return np.hstack([A2 * A1[:, i][:, np.newaxis] for i in range(A1.shape[1])])    

    if len(As)==1:
        return As[0]
    
    return columnwise_product(te(*As[:-1]), As[-1])
