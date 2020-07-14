import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase, interp1d
from .._splines import build_spline_matrix

from ..MF import KMeans, semiNMF

__all__ = ['splineLNLN']

class splineLNLN(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', filter_nonlinearity='softplus', output_nonlinearity='softplus', 
                 compute_mle=False, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, compute_mle, **kwargs)
        self.filter_nonlinearity = filter_nonlinearity
        self.output_nonlinearity = output_nonlinearity

    def cost(self, p):

        """

        Negetive Log Likelihood.

        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        R = self.R

        
        if self.fit_subunits_weight:
            subunits_weight = np.maximum(p['subunits_weight'], 1e-7)
            subunits_weight /= np.sum(subunits_weight)
        else:
            subunits_weight = self.p0['subunits_weight'] # equal weight

        
        if self.fit_linear_filter:
            filter_output = np.nansum(self.fnl(XS @ p['b'].reshape(self.n_b, self.n_subunits), nl=self.filter_nonlinearity) * subunits_weight, 1) 
        else:
            filter_output = np.nansum(self.fnl(XS @ self.b_opt.reshape(self.n_b, self.n_subunits), nl=self.filter_nonlinearity) * subunits_weight, 1) 
            

        
        if self.fit_linear_filter:
            filter_output = XS @ p['b']
        else:
            if hasattr(self, 'b_opt'):
                filter_output = XS @ self.b_opt
            else:
                filter_output = XS @ self.b_spl

        
        if self.fit_intercept:
            intercept = p['intercept']
        else:
            if hasattr(self, 'intercept'):
                intercept = self.intercept
            else:
                intercept = 0.

        
        if self.fit_history_filter:
            history_output = self.yS @ p['bh']  
        else:
            if hasattr(self, 'bh_spl'):
                history_output = self.yS @ self.bh_spl
            else:
                history_output = 0.

        
        if self.fit_nonlinearity:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p['bnl'])

        r = R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity) 
        
        term0 = - np.log(r) @ y
        term1 = np.nansum(r) * dt

        neglogli = term0 + term1

        if self.beta:
            l1 = np.linalg.norm(p['b'], 1)
            l2 = np.linalg.norm(p['b'], 2)
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
            b0 = 0.01 * random.normal(key, shape=(self.n_b, self.n_subunits)).flatten()
            
            p0 = {'b': b0}
            p0.update({'bh': self.bh_spl}) if self.fit_history_filter else p0.update({'bh': None})
            p0.update({'bnl': self.bnl}) if self.fit_nonlinearity else p0.update({'bnl': None})
            p0.update({'intercept': 0.}) if self.fit_intercept else p0.update({'intercept': None})
            p0.update({'subunits_weight': subunits_weight})

        self.p0 = p0
        self.p_opt = self.optimize_params(self.p0, num_iters, step_size, tolerance, verbal)   
        
        if fit_linear_filter:
            self.b_opt = self.p_opt['b'].reshape(self.n_b, self.n_subunits)
            self.w_opt = self.S @ self.b_opt
        
        if fit_history_filter:
            self.h_opt = self.Sh @ self.p_opt['bh']
        
        if fit_intercept:
            self.intercept = self.p_opt['intercept']
        
        if fit_subunits_weight:
            self.subunits_weight = self.p_opt['subunits_weight']

        if fit_nonlinearity:
            self.bnl_opt = self.p_opt['bnl']

