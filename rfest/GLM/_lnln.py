import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers
from jax.config import config
config.update("jax_enable_x64", True)

import time
import itertools
from ._base import Base, interp1d
from ..utils import build_design_matrix

__all__ = ['LNLN']

class LNLN(Base):
    
    """
    
    Multi-filters Linear-Nonliear-Poisson model. 
    
    """
    
    def __init__(self, X, y, dims, compute_mle=False,
            output_nonlinearity='softplus', filter_nonlinearity='softplus',**kwargs):
        
        super().__init__(X, y, dims, compute_mle, **kwargs)
        self.output_nonlinearity = output_nonlinearity
        self.filter_nonlinearity = filter_nonlinearity
        self.fit_subunits_weight = kwargs['fit_subunits_weight'] if 'fit_subunits_weight' in kwargs.keys() else False


    def forward_pass(self, p, extra):

        dt = self.dt
        X = self.X if extra is None else extra['X']
        X = X.reshape(X.shape[0], -1)
    
        if hasattr(self, 'h_mle'):
            if extra is not None:
                yh = extra['yh']
            else:
                yh = self.yh

        if self.fit_intercept:
            intercept = p['intercept'] 
        else:
            if hasattr(self, 'intercept'):
                intercept = self.intercept
            else:
                intercept = 0.
        
        if self.fit_R: # maximum firing rate / scale factor
            R = p['R']
        else:
            if hasattr(self, 'R'):
                R = self.R
            else:
                R = 1.

        if self.fit_nonlinearity:
            nl_params = p['nl_params']
        else:
            if hasattr(self, 'nl_params'):
                nl_params = [self.nl_params for i in range(self.n_s)]
            else:
                nl_params = [None for i in range(self.n_s)]
        
        if self.fit_linear_filter:
            linear_output = X @ p['w'].reshape(self.n_features * self.n_c, self.n_s)
            nonlin_output = np.array([self.fnl(linear_output[:, i], nl=self.filter_nonlinearity, params=nl_params[i]) for i in range(self.n_s)])
            filter_output = np.mean(nonlin_output, 0) 
        else:
            linear_output = X @ self.w_opt.reshape(self.n_features * self.n_c, self.n_s)
            nonlin_output = np.array([self.fnl(linear_output[:, i], nl=self.filter_nonlinearity, params=nl_params[i]) for i in range(self.n_s)])
            filter_output = np.mean(nonlin_output, 0)

        if self.fit_history_filter:
            history_output = yh @ p['h']  
        else:
            if hasattr(self, 'h_opt'):
                history_output = yh @ self.h_mle
            elif hasattr(self, 'h_mle'):
                history_output = yh @ self.h_mle
            else:
                history_output = 0.
        
        r = dt * R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity).flatten() # conditional intensity (per bin)

        return r


    def cost(self, p, extra=None, precomputed=None):
        
        y = self.y if extra is None else extra['y']
        r = self.forward_pass(p, extra) if precomputed is None else precomputed 
        r = np.maximum(r, 1e-20) # remove zero to avoid nan in log.
        dt = self.dt

        term0 = - np.log(r / dt) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.beta and extra is None:
            l1 = np.linalg.norm(p['w'], 1)
            l2 = np.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
        
        
    def fit(self, p0=None, extra=None, num_subunits=2, 
            num_epochs=1, num_iters=3000, metric=None,
            alpha=1, beta=0.05, 
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False, 
            step_size=1e-2, tolerance=10, verbose=100, random_seed=2046):

        self.metric = metric

        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        self.beta = beta # elastic net parameter - global penalty weight for linear filter
        
        self.n_s = num_subunits
        self.num_iters = num_iters   
        
        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        if extra is not None:
            
            if hasattr(self, 'h_mle'):
                yh = np.array(build_design_matrix(extra['y'][:, np.newaxis], self.yh.shape[1], shift=1))
                extra.update({'yh': yh}) 

            extra = {key: np.array(extra[key]) for key in extra.keys()}

        # initialize parameters
        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()

        if 'w' not in dict_keys:
            key = random.PRNGKey(random_seed)
            w0 = 0.01 * random.normal(key, shape=(self.n_features * self.n_c * self.n_s, )).flatten() 
            p0.update({'w': w0})

        if 'intercept' not in dict_keys:
            p0.update({'intercept': np.array([0.])})

        if 'R' not in dict_keys and self.fit_R:
            p0.update({'R': np.array([1.])})

        if 'h' not in dict_keys:
            try:
                p0.update({'h': self.h_mle})            
            except:
                p0.update({'h': None})  
        
        if 'nl_params' not in dict_keys:
            if hasattr(self, 'nl_params'):
                p0.update({'nl_params': [self.nl_params for i in range(self.n_s+1)]})
            else:
                p0.update({'nl_params': [None for i in range(self.n_s + 1)]})
        
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_epochs, num_iters, metric, step_size, tolerance, verbose)   
        self.R = self.p_opt['R']
        
        if fit_linear_filter: 
            if self.n_c > 1:
                self.w_opt = self.p_opt['w'].reshape(self.n_features, self.n_c, self.n_s)
            else:
                self.w_opt = self.p_opt['w'].reshape(self.n_features, self.n_s)
        
        if fit_history_filter:
            self.h_opt = self.p_opt['h']
        
        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt['nl_params']
       
        if fit_intercept:
            self.intercept = self.p_opt['intercept']