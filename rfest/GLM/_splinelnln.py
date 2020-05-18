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

    def cost(self, b):

        """

        Negetive Log Likelihood.

        """
        
        XS = self.XS
        y = self.y
        dt = self.dt
                
        filter_output = np.sum(self.nonlin(XS @ b.reshape(self.n_b, self.n_subunits), nl=self.filter_nonlinearity), 1)
        r = dt * self.nonlin(filter_output, nl=self.output_nonlinearity).flatten() # conditional intensity (per bin)
        
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

    def fit(self, p0='random',num_subunits=2, num_iters=5, num_iters_init=100, alpha=1, lambd=0.05, gamma=0.0,
            step_size=1e-2, tolerance=10, verbal=1, random_seed=2046):

        self.lambd = lambd # elastic net parameter - global weight
        self.alpha = alpha # elastic net parameter (1=L1, 0=L2)
        self.gamma = gamma # nuclear norm parameter
        
        self.n_subunits = num_subunits
        self.num_iters = num_iters   
        
        if type(p0) == str:

            if p0 == 'random':
                if verbal:
                    print('Randomly initializing subunits...')
                key = random.PRNGKey(random_seed)
                p0 = 0.01 * random.normal(key, shape=(self.n_b, self.n_subunits)).flatten()
        
            elif p0 == 'kmeans':
                if verbal:
                    print('Initializing subunits with K-means clustering...')
                kms = KMeans(self.X[self.y!=0].T, k=self.n_subunits, build_S=True, dims=self.dims, df=self.df)
                kms.fit(num_iters=num_iters_init, verbal=verbal, tolerance=tolerance)
                
                self.b_kms = kms.B
                self.w_kms = kms.W
                
                p0 = self.b_kms

            elif p0 == 'seminmf':
                if verbal:
                    print('Initializing subunits with semi-NMF...')
                nmf = semiNMF(self.X[self.y!=0].T, k=self.n_subunits, build_L=True, dims_L=self.dims, df_L=self.df)
                nmf.fit(num_iters=num_iters_init, verbal=verbal, tolerance=tolerance)
                
                self.b_nmf = nmf.B
                self.w_nmf = nmf.W

                p0 = self.b_nmf

            else:
                raise ValueError(f'Initialization `{p0}` is not supported.')
            
            if verbal:
                print('Finished Initialization. \n')

        else:
            p0 = p0

        self.b_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)   
        self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_subunits)
