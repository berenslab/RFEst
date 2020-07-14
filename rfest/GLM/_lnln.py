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

        intercept = p['intercept'] if self.fit_intercept else 0.

        if self.fit_subunits_weight:
            subunits_weight = np.maximum(p['subunits_weight'], 1e-7)
            subunits_weight /= np.sum(subunits_weight)
        else:
            subunits_weight = np.ones(self.n_subunits) / self.n_subunits # equal weight

        if self.fit_linear_filter:
            filter_output = np.sum(self.fnl(X @ p['w'].reshape(self.n_features, self.n_subunits), nl=self.filter_nonlinearity), 1)
        else:
            filter_output = np.sum(self.fnl(X @ self.w_opt.reshape(self.n_features, self.n_subunits), nl=self.filter_nonlinearity), 1)
        
        if self.fit_history_filter:
            history_output = self.yh @ p['h']  
        else:
            if hasattr(self, 'h_mle'):
                history_output = self.yh @ self.h_mle
            else:
                history_output = 0.
                
        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])

        r = R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity).flatten() # conditional intensity (per bin)
        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) * dt # non-spike term

        neglogli = term0 + term1
        
        if self.beta:
            l1 = np.linalg.norm(p['w'], 1)
            l2 = np.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
        
    
    def fit(self, p0=None, num_subunits=2, num_iters=5, num_iters_init=100, 
            alpha=1, beta=0.05, 
            fit_linear_filter=True, fit_history_filter=False, 
            fit_nonlinearity=False, fit_intercept=True, 
            fit_subunits_weight=False, 
            step_size=1e-2, tolerance=10, verbal=1, random_seed=2046):

        self.beta = beta # elastic net parameter - global penalty weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   

        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept
        self.fit_subunits_weight = fit_subunits_weight

        subunits_weight = np.ones(num_subunits)/num_subunits

        if p0 is None:
        
            key = random.PRNGKey(random_seed)

            w0 = 0.01 * random.normal(key, shape=(self.n_features, self.n_subunits)).flatten() 
            p0 = {'w': w0}
            p0.update({'h': self.h_mle}) if self.fit_history_filter else p0.update({'h': None})
            p0.update({'bnl': self.bnl}) if self.fit_nonlinearity else p0.update({'bnl': None})
            p0.update({'intercept': 0.}) if self.fit_intercept else p0.update({'intercept': None})
            p0.update({'subunits_weight': subunits_weight})

        self.p0 = p0
        self.p_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        self.w_opt = self.p_opt['w']  if fit_linear_filter else self.w_opt
        self.h_opt = self.p_opt['h'] if self.fit_history_filter else None
        self.intercept = self.p_opt['intercept'] if self.fit_intercept else 0.
        self.subunits_weight = self.p_opt['subunits_weight'] if fit_subunits_weight else subunits_weight
