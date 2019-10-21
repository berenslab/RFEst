import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

__all__ = ['LNLN']

class LNLN:
    
    """
    Multi-filters Linear-Nonliear-Poisson model with fixed (softplus) nonlinearity. 
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
        
        self.Cinv = None
    
    def add_C_posterior(self, Cinv):
        self.Cinv = Cinv
        
    def neglogposterior(self, K):
        
        X = self.X
        y = self.y
        dt = self.dt
        
        def nonlin(x):
            return np.log(1 + np.exp(x)) + 1e-17

        filter_output = np.sum(nonlin(X @ K.reshape(self.n_features, self.n_subunits)), 1)
        
        r = dt * nonlin(filter_output).flatten() # conditional intensity (per bin)
        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.Cinv is not None:
            
            Cinv = self.Cinv
            K = K.reshape(self.n_features, self.n_subunits)
            p = .5 * K.T @ Cinv @ K
            neglogli += p

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
    
    def fit(self, initial_params=None, num_subunits=1, num_iters=5, step_size=1e-2, tolerance=10, verbal=True, random_seed=1):

        self.n_subunits = num_subunits
        self.num_iters = num_iters   
        
        if initial_params is None:
        
            key = random.PRNGKey(random_seed)
            initial_params = 0.01 * random.normal(key, shape=(self.n_features, self.n_subunits)).flatten()
        
        self.w_opt = self.optimize_params(initial_params, num_iters, step_size, tolerance, verbal)
