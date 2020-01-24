import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase
from .._splines import build_spline_matrix

__all__ = ['splineLNLN']

class splineLNLN(splineBase):

    def __init__(self, X, y, dims, df, smooth='cr', output_nonlinearity='softplus', filter_nonlinearity='softplus',
                 add_intercept=False, compute_mle=True, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, add_intercept, compute_mle, **kwargs)
        self.output_nonlinearity = output_nonlinearity
        self.filter_nonlinearity = filter_nonlinearity

    def cost(self, b):

        """

        Negetive Log Likelihood.

        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
                
        def filter_nonlin(x):
            nl = self.filter_nonlinearity
            if  nl == 'softplus':
                return np.log(1 + np.exp(x)) + 1e-17
            elif nl == 'exponential':
                return np.exp(x)
            elif nl == 'square':
                return np.power(x, 2)
            elif nl == 'relu':
                return np.maximum(0, x)
            elif nl == 'none':
                return x
            else:
                raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')
            
        def output_nonlin(x):
            nl = self.output_nonlinearity
            if  nl == 'softplus':
                return np.log(1 + np.exp(x)) + 1e-17
            elif nl == 'exponential':
                return np.exp(x)
            elif nl == 'square':
                return np.power(x, 2)
            elif nl == 'relu':
                return np.maximum(0, x)
            elif nl == 'none':
                return x
            else:
                raise ValueError(f'Input output nonlinearity `{nl}` is not supported.')

        filter_output = np.sum(filter_nonlin(XS @ b.reshape(self.n_b, self.n_subunits)), 1)
        r = dt * output_nonlin(filter_output).flatten() # conditional intensity (per bin)
        
        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.lambd:
            l1 = np.sum(np.abs(b))
            l2 = np.sqrt(np.sum(b**2)) 
            neglogli += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1)
        # nuc = np.linalg.norm(b.reshape(self.n_b, self.n_subunits), 'nuc') # wait for JAX update
        if self.gamma:
            nuc = np.sum(np.linalg.svd(b.reshape(self.n_b, self.n_subunits), full_matrices=False, compute_uv=False), axis=-1)
            neglogli += self.gamma * nuc
        
        return neglogli

    def fit(self, p0=None, num_subunits=1, num_iters=5, alpha=0.5, lambd=0.05, gamma=0.0,
            step_size=1e-2, tolerance=10, verbal=1, random_seed=2046):

        self.lambd = lambd # elastic net parameter - global weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        self.gamma = gamma # nuclear norm parameter
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   
        
        if p0 is None:
        
            key = random.PRNGKey(random_seed)
            p0 = 0.01 * random.normal(key, shape=(self.n_b, self.n_subunits)).flatten()
        
        self.b_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)   
        self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_subunits)
