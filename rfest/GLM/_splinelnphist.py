import jax.numpy as np
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from ._base import splineBase
from .._utils import build_design_matrix
from .._splines import build_spline_matrix

__all__ = ['splineLNPHist']


class splineLNPHist(splineBase):
    
    """
    
    Spline-based Linear-Nonlinear-Poisson model with spike history filter.
    
    """

    def __init__(self, X, y, dims, df, smooth='cr', nonlinearity='softplus',
                 add_intercept=False, compute_mle=True, **kwargs):
        
        super().__init__(X, y, dims, df, smooth, add_intercept, compute_mle, **kwargs)
        
        # spline is more flexible than raised cosine. but is it the best choice?
        Sh = np.array(build_spline_matrix([dims[0], ], df, smooth)) 
#         _, Sh = build_raised_cosine_matrix(nh=df, endpoints=np.array([0.1, 10]), b=1.5, dt=0.1)
        
        yh = build_design_matrix(y[:, np.newaxis], Sh.shape[0], shift=1)
        
        self.Sh = Sh # spline basis for spike history filter
        self.yh = yh # spike history design matrix
        self.yhSh = yh @ Sh 
        
        self.n_bw = self.S.shape[1] # number of basis coefficients for RF
        self.n_bh = self.Sh.shape[1] # number of basis coefficients for spike history filter
        self.n_b = self.n_bw + self.n_bh # total number of basis coefficients
        
        self.nonlinearity = nonlinearity
        
    def cost(self, b):

        """
        Negetive Log Likelihood.
        """
        
        XS = self.XS
        yhSh = self.yhSh
        y = self.y
        dt = self.dt
    
        def nonlin(x):
            nl = self.nonlinearity
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

        filter_output = nonlin(XS @ b[:self.S.shape[1]] + yhSh @ b[self.S.shape[1]:]).flatten()
        r = dt * filter_output

        term0 = - np.log(r) @ y # spike term from poisson log-likelihood
        term1 = np.sum(r) # non-spike term

        neglogli = term0 + term1
        
        if self.lambd:
            l1 = np.sum(np.abs(b))
            l2 = np.sqrt(np.sum(b**2)) 
            neglogli += self.lambd * ((1 - self.alpha) * l2 + self.alpha * l1)

        return neglogli
    
    def fit(self, p0=None, num_iters=5, alpha=0.5, lambd=0.5,
            step_size=1e-2, tolerance=10, verbal=1):


        """
        Parameters
        ==========
        p0 : array_like, shape (n_b, ) or (n_b, n_subunits)
            Initial spline coefficients.
        num_iters : int
            Max number of optimization iterations.
        
        alpha : float, from 0 to 1.
            Elastic net parameter, balance between L1 and L2 regulization.
            * 0.0 -> only L2
            * 1.0 -> only L1
        lambd : float
            Elastic net parameter, overall weight of regulization.
        step_size : float
            Initial step size for JAX optimizer.
        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically 
            increases or stop increases for tolerance=n steps.
        verbal: int
            When `verbal=0`, progress is not printed. When `verbal=n`,
            progress will be printed in every n steps.
        """

        self.lambd = lambd
        self.alpha = alpha 
        self.num_iters = num_iters   
        
        if p0 is None: # if p0 is not provided, initialize it with spline MLE.
            p0 = np.hstack([self.b_spl.flatten(), np.ones(self.Sh.shape[1])])
        
        self.b_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        self.w_opt = self.S @ self.b_opt[:self.n_bw] # fitted RF
        self.h_opt = self.Sh @ self.b_opt[self.n_bw:] # fitted spike history filter
        
# not used. keep it here for now. may move to a seperate file in the future.
def build_raised_cosine_matrix(nh, endpoints, b, dt):
    
    """
    Make basis of raised cosines with logarithmically stretched time axis.
    
    Ported from [matlab code](https://github.com/pillowlab/raisedCosineBasis)
    
    Parameters
    ==========
    nh : int
        number of basis vectors
    
    endpoints : array like, shape=(2, )
        absoute temporal position of center of 1st and last cosine basis vector
        
    b : float
        offset for nonlinear stretching of x axis: y=log(x+b)
    
    dt : float
        time bin size of bins representing basis
        
    Return
    ======
    
    ttgrid : shape=(nt, )
        time lattice on which basis is defined
    
    basis : shape=(nt, nh)
        original cosine basis vectors
        
    """
    
    def nl(x):
        return np.log(x + 1e-20)
    
    def invnl(x):
        return np.exp(x) - 1e-20
    
    def raised_cosine_basis(x, c, dc):
        return 0.5 * (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(x-c)*np.pi/dc/2)))+1)
    
    yendpoints = nl(endpoints + b)
    dctr = np.diff(yendpoints) / (nh - 1)
    ctrs = np.linspace(yendpoints[0], yendpoints[1], nh)
    maxt = invnl(yendpoints[1]+2*dctr) - b
    ttgrid = np.arange(0, maxt+dt, dt)
    nt = len(ttgrid)
    
    xx = np.tile(nl(ttgrid+b)[:, np.newaxis], (1, nh))
    cc = np.tile(ctrs, (nt, 1))
    
    basis = raised_cosine_basis(xx, cc, dctr)
    
    return ttgrid, basis

