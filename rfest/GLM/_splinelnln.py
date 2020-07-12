import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase
from .._splines import build_spline_matrix

from ..MF import KMeans, semiNMF

__all__ = ['splineLNLN']

class splineLNLN(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', output_nonlinearity='softplus', filter_nonlinearity='softplus',
                 add_intercept=False, compute_mle=False, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, add_intercept, compute_mle, **kwargs)
        self.output_nonlinearity = output_nonlinearity
        self.filter_nonlinearity = filter_nonlinearity
        self.fit_subunits_weight = kwargs['fit_subunits_weight'] if 'fit_subunits_weight' in kwargs.keys() else False


    def cost(self, p):

        """

        Negetive Log Likelihood.

        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
        R = self.R

        if self.add_intercept:
            intercept = p['intercept']
        else:
            intercept = 0
        
        if self.fit_subunits_weight:
            subunits_weight = np.maximum(p['subunits_weight'], 1e-7)
            subunits_weight /= np.sum(subunits_weight)
        else:
            subunits_weight = np.ones(self.n_subunits) / self.n_subunits # equal weight

        if self.response_history:
            yS = self.yS
            history_output =  yS @ p['bh']
        else:
            history_output = 0
        

        filter_output = np.nansum(self.fnl(XS @ p['b'].reshape(self.n_b, self.n_subunits), nl=self.filter_nonlinearity) * subunits_weight, 1)
        r = R * self.fnl(filter_output + history_output + intercept, nl=self.output_nonlinearity) 
        
        term0 = - np.log(r) @ y
        term1 = np.nansum(r) * dt

        neglogli = term0 + term1

        if self.beta:
            l1 = np.linalg.norm(p['b'], 1)
            l2 = np.linalg.norm(p['b'], 2)
            neglogli += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)
        
        return neglogli


    def fit(self, p0=None, num_subunits=2, num_iters=5, num_iters_init=100, alpha=1, beta=0.05,
            step_size=1e-2, tolerance=10, verbal=1, random_seed=2046):

        self.beta = beta # elastic net parameter - global penalty weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   
            
        if p0 is None:

            key = random.PRNGKey(random_seed)
            b0 = 0.01 * random.normal(key, shape=(self.n_b, self.n_subunits)).flatten()
            subunits_weight = np.ones(num_subunits)/num_subunits
            
            p0 = {'b': b0, 
                  'subunits_weight': subunits_weight}
            p0.update({'bh': self.bh_spl}) if self.response_history else p0.update({'bh': None})
            p0.update({'intercept': 0.}) if self.add_intercept else p0.update({'intercept': None})

        self.p0 = p0
        self.p_opt = self.optimize_params(self.p0, num_iters, step_size, tolerance, verbal)   
        self.b_opt = self.p_opt['b'].reshape(self.n_b, self.n_subunits)
        self.w_opt = self.S @ self.b_opt
        self.h_opt = self.Sh @ self.p_opt['bh'] if self.response_history else None
        self.intercept = self.p_opt['intercept'] if self.add_intercept else 0
