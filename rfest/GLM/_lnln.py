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

        if self.fit_R:
            R = p['R']
        else:
            R = np.array([1.])

        if self.fit_nonlinearity:
            if np.ndim(p['bnl']) != 1:
                self.fitted_nonlinearity = [interp1d(self.bins, self.Snl @ p['bnl'][i]) for i in range(self.n_s)]
            else:
                self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl']) 

        if self.fit_linear_filter:
            filter_output = np.mean(self.fnl(X @ p['w'].reshape(self.n_features * self.n_c, self.n_s), nl=self.filter_nonlinearity), 1)
        else:
            filter_output = np.mean(self.fnl(X @ self.w_opt.reshape(self.n_features * self.n_c, self.n_s), nl=self.filter_nonlinearity), 1)
        
        if self.fit_history_filter:
            history_output = yh @ p['h']  
        else:
            if hasattr(self, 'h_opt'):
                history_output = yh @ self.h_mle
            elif hasattr(self, 'h_mle'):
                history_output = yh @ self.h_mle
            else:
                history_output = 0.
        
        if self.fit_intercept:
            intercept = p['intercept']
        else:
            if hasattr(self, 'intercept'):
                intercept = self.intercept
            else:
                intercept = 0.
                
        r = dt * R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity).flatten() # conditional intensity (per bin)

        return r


    def cost(self, p, extra=None, precomputed=None):
        
        y = self.y if extra is None else extra['y']
        r = self.forward_pass(p, extra) if precomputed is None else precomputed 
        dt = self.dt

        term0 = - np.log(r / dt) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.beta and extra is None:
            l1 = np.linalg.norm(p['w'], 1)
            l2 = np.linalg.norm(p['w'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
        
        
    def fit(self, p0=None, extra=None, num_subunits=2, num_iters=5, metric=None,
            alpha=1, beta=0.05, 
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False, 
            step_size=1e-2, tolerance=10, verbose=1, random_seed=2046):

        self.beta = beta # elastic net parameter - global penalty weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        
        self.n_s = num_subunits
        self.num_iters = num_iters   
        
        if fit_nonlinearity:  
            self.fitted_nonlinearity = [interp1d(self.bins, self.Snl @ self.bnl) for i in range(num_subunits)]
            self.bnl = np.repeat(self.bnl[:, np.newaxis], num_subunits, axis=1).T
            
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

        if 'bnl' not in dict_keys:
            try:
                p0.update({'bnl': self.bnl})
            except:
                p0.update({'bnl': None})
        else:
            if np.ndim(p0['bnl']) != 1:
                self.fitted_nonlinearity = [interp1d(self.bins, self.Snl @ p0['bnl'][i]) for i in range(num_subunits)]
            else:
                self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p0['bnl'])

        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_iters, metric, step_size, tolerance, verbose)   
        self.R = self.p_opt['R']
        
        if fit_linear_filter: 
            if self.n_c > 1:
                self.w_opt = self.p_opt['w'].reshape(self.n_features, self.n_c, self.n_s)
            else:
                self.w_opt = self.p_opt['w'].reshape(self.n_features, self.n_s)
        
        if fit_history_filter:
            self.h_opt = self.p_opt['h']
        
        if fit_intercept:
            self.intercept = self.p_opt['intercept']

        if fit_nonlinearity:
            self.bnl_opt = self.p_opt['bnl']