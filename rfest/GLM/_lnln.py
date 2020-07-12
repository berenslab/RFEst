import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import Base

__all__ = ['LNLN']

class LNLN(Base):
    
    """
    
    Multi-filters Linear-Nonliear-Poisson model. 
    
    """
    
    def __init__(self, X, y, dims, dt, R=1, compute_mle=False, add_intercept=False,
            output_nonlinearity='softplus', filter_nonlinearity='softplus',**kwargs):
        
        super().__init__(X, y, dims, add_intercept, compute_mle, **kwargs)
        self.output_nonlinearity = output_nonlinearity
        self.filter_nonlinearity = filter_nonlinearity
        self.fit_subunits_weight = kwargs['fit_subunits_weight'] if 'fit_subunits_weight' in kwargs.keys() else False


    def cost(self, p):
        
        X = self.X
        y = self.y
        dt = self.dt
        R = self.R

        if self.fit_subunits_weight:
            subunits_weight = np.maximum(p['subunits_weight'], 1e-7)
            subunits_weight /= np.sum(subunits_weight)
        else:
            subunits_weight = np.ones(self.n_subunits) / self.n_subunits # equal weight

        intercept = p['intercept'] if self.add_intercept else 0
        history_output = self.yh @ p['h'] if self.response_history else 0
        filter_output = np.sum(self.fnl(X @ p['w'].reshape(self.n_features, self.n_subunits), nl=self.filter_nonlinearity), 1)
        
        r = R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity).flatten() # conditional intensity (per bin)
        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) * dt # non-spike term

        neglogli = term0 + term1
        
        if self.beta:
            l1 = np.linalg.norm(p['w'], 1)
            l2 = np.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
        
    
    def fit(self, p0=None, num_subunits=1, num_iters=5,  alpha=0.5, beta=0.05,
            step_size=1e-2, tolerance=10, verbal=True, random_seed=2046):

        self.beta = beta # elastic net parameter - global weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   

        if p0 is None:
        
            key = random.PRNGKey(random_seed)
            subunits_weight = np.ones(num_subunits)/num_subunits

            w0 = 0.01 * random.normal(key, shape=(self.n_features, self.n_subunits)).flatten()
            p0 = {'w': w0, 
                  'subunits_weight': subunits_weight}
            p0.update({'h': self.h_mle}) if self.response_history else p0.update({'h': None})
            p0.update({'intercept': 0.}) if self.add_intercept else p0.update({'intercept': None})

        self.p0 = p0
        self.p_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        self.w_opt = self.p_opt['w']
        self.h_opt = self.p_opt['h'] if self.response_history else None
        self.intercept = self.p_opt['intercept'] if self.add_intercept else 0

