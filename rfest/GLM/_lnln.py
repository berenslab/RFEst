import jax.numpy as np
import jax.random as random
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
    
    def __init__(self, X, y, dims, dt, R=1, compute_mle=False,
            output_nonlinearity='softplus', filter_nonlinearity='softplus',**kwargs):
        
        self.X = X # stimulus design matrix
        self.y = y # response 
        self.dt = dt # time bin size
        self.R = R # maximum firing rate
        
        self.dims = dims # assumed order [t, y, x]
        self.n_samples, self.n_features = X.shape

        self.XtY = X.T @ y
        if np.array_equal(y, y.astype(bool)): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)

        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)

        self.output_nonlinearity = output_nonlinearity
        self.filter_nonlinearity = filter_nonlinearity

    def nonlin(self, x, nl):
        if  nl == 'softplus':
            return np.log(1 + np.exp(x)) + 1e-7
        elif nl == 'exponential':
            return np.exp(x)
        elif nl == 'relu':
            return np.maximum(1e-7, x)
        elif nl == 'none':
            return x
        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def cost(self, K):
        
        X = self.X
        y = self.y
        dt = self.dt
        
        def nonlin(x):
            return np.log(1 + np.exp(x)) + 1e-17

        filter_output = np.sum(self.nonlin(X @ K.reshape(self.n_features, self.n_subunits), nl=self.filter_nonlinearity), 1)
        
        r = dt * self.nonlin(filter_output, nl=self.output_nonlinearity).flatten() # conditional intensity (per bin)
        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.lambd:
            l1 = np.linalg.norm(K, 1)
            l2 = np.linalg.norm(K, 2)
            neglogli += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1)
        # nuc = np.linalg.norm(B.reshape(self.n_spline_coeff, self.n_subunits), 'nuc') # wait for JAX implementation
        if self.gamma:
            nuc = np.sum(np.linalg.svd(K.reshape(self.n_features, self.n_subunits), full_matrices=False, compute_uv=False), axis=-1)
            neglogli += self.gamma * nuc
        
        return neglogli
    
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
    
    def fit(self, p0=None, num_subunits=1, num_iters=5,  alpha=0.5, lambd=0.05, gamma=0.0,
            step_size=1e-2, tolerance=10, verbal=True, random_seed=2046):

        self.lambd = lambd # elastic net parameter - global weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        self.gamma = gamma # nuclear norm parameter
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   
        if p0 is None:
        
            key = random.PRNGKey(random_seed)
            p0 = 0.01 * random.normal(key, shape=(self.n_features, self.n_subunits)).flatten()
        
        self.w_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal).reshape(self.n_features, self.n_subunits)
