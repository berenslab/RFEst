import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

__all__ = ['LNLN']

class splineLNLN:

    """
    Spline-based multi-filters Linear-Nonliear-Poisson model with fixed (softplus) nonlinearity. 
    """
    
    def __init__(self, X, y, dt, dims, compute_mle=True):
        
        self.X = X # stimulus design matrix
        self.y = y # response 
        self.dt = dt # time bin size 
        
        self.dims = dims # assumed order [t, y, x]
        self.n_samples, self.n_features = X.shape

        if compute_mle:
            self.w_mle = np.linalg.solve(X.T @ X, X.T @ y)
        else:
            self.w_mle = None
        
        S = self._make_splines_matrix(df_splines)
        self.S = S
        self.n_spline_coeff = self.S.shape[1]
        self.w_spl = S @ np.linalg.inv(S.T @ X.T @ X @ S) @ S.T @ X.T @ y
        
        
    def _make_splines_matrix(self, df):
        
        if np.ndim(df) != 0 and len(df) != self.ndim:
            raise ValueError("`df` must be an integer or an array the same length as `dims`")
        elif np.ndim(df) == 0:
            df = np.ones(self.ndim) * df
        
        if self.ndim == 1:
        
            self.S = patsy.cr(np.arange(self.dims[0]), df[0])
            
        elif self.ndim == 2:
        
            g0, g1 = np.meshgrid(np.arange(self.dims[1]), np.arange(self.dims[0]))
            S = patsy.te(patsy.cr(g0.ravel(), df[0]), patsy.cr(g1.ravel(), df[1]))
            
        elif self.ndim == 3:
            
            g0, g1, g2 = np.meshgrid(np.arange(self.dims[1]), 
                                     np.arange(self.dims[0]), 
                                     np.arange(self.dims[2]))
            S = patsy.te(patsy.cr(g0.ravel(), df[0]), 
                         patsy.cr(g1.ravel(), df[1]), 
                         patsy.cr(g2.ravel(), df[2]))
            
        return S
        
        
        
    def neglogposterior(self, B):
        
        XS = self.X @ self.S
        y = self.y
        dt = self.dt
        
        def nonlin(x):
            return np.log(1 + np.exp(x)) + 1e-17

        filter_output = np.sum(nonlin(XS @ B.reshape(self.n_spline_coeff, self.n_subunits)), 1)
        
        r = dt * nonlin(filter_output).flatten() # conditional intensity (per bin)
        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        return neglogli
        
    def optimize_params(self, initial_params, num_iters, step_size, tolerance, verbal):
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(initial_params)
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            g = grad(self.neglogposterior)(p)
            return opt_update(i, g, opt_state)

        cost_list = []
        params_list = []    

        if verbal:
            print('{0}\t{1}\t'.format('Iter', 'Cost'))

        for i in range(num_iters):
            
            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.neglogposterior(params_list[-1]))
            
            if verbal:
                print('{0}\t{1:.3f}\t'.format(i,  cost_list[-1]))
            
            if len(params_list) > tolerance:
                
                if np.all((np.array(cost_list[1:])) - np.array(cost_list[:-1]) > 0 ):
                    params = params_list[0]
                    print('Stop: cost has been monotonically increasing for {} steps.'.format(tolerance))
                    break
                elif np.all(np.array(cost_list[:-1]) - np.array(cost_list[1:]) < 1e-5):
                    params = params_list[-1]
                    print('Stop: cost has been stop changing for {} steps.'.format(tolerance))
                    break                    
                else:
                    params_list.pop(0)
                    cost_list.pop(0)     
        else:
            print('Stop: reached maxiter = {}.'.format(num_iters))
            params = params_list[-1]
            
        return params      
    
    def fit(self, initial_params=None, num_subunits=1, num_iters=5, step_size=1e-2, tolerance=10, verbal=True):

        self.n_subunits = num_subunits
        self.num_iters = num_iters   
        
        if initial_params is None:
        
            key = random.PRNGKey(1)
            initial_params = 0.01 * random.normal(key, shape=(self.n_spline_coeff, self.n_subunits)).flatten()
        
        self.B_opt = self.optimize_params(initial_params, num_iters, step_size, tolerance, verbal)   
        self.w_opt = self.S @ self.B_opt
