import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import mean_squared_error

from .._utils import build_design_matrix
from .._splines import build_spline_matrix
from scipy.optimize import minimize

__all__ = ['splineBase']

class splineBase:

    """
    
    Base class for spline-based GLMs. 
    
    """

    def __init__(self, X, y, dims, df, smooth='cr', add_intercept=False, compute_mle=False, **kwargs):

        """
        
        Parameters
        ==========
        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : array_like, shape (n_samples, )
            Recorded response

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sy, sx]

        df : int
            Degree of freedom for spline /smooth basis. 

        smooth : str
            Spline or smooth to be used. Current supported methods include:
            * `bs`: B-spline
            * `cr`: Cubic Regression spline
            * `tp`: (Simplified) Thin Plate regression spline 

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """ 

        # store meta data

        self.dims = dims # assumed order [t, y, x]
        self.ndim = len(dims)
        self.n_samples, self.n_features = X.shape
        
        # compute sufficient statistics

        S = np.array(build_spline_matrix(dims, df, smooth)) # for w

        if add_intercept:
            X = np.hstack([np.ones(self.n_samples)[:, np.newaxis], X])
            S = np.vstack([np.ones(S.shape[1]), S])

        XS = X @ S
        self.XtY = X.T @ y
        if np.array_equal(y, y.astype(bool)): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)
        
        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)

        self.X = np.array(X) # stimulus design matrix
        self.y = np.array(y) # response

        self.S = S # spline matrix
        self.XS = XS 
        
        self.n_b = S.shape[1] # num:ber of spline coefficients
        
        # compute spline-based maximum likelihood 
        self.b_spl = np.linalg.solve(XS.T @ XS, XS.T @ y)
        self.w_spl = S @ self.b_spl

        # store more meta data
        
        self.df = df 
        self.smooth = smooth   

        self.dt = kwargs['dt'] if 'dt' in kwargs.keys() else 1 # time bin size (for spike data)
        self.R = kwargs['R'] if 'R' in kwargs.keys() else 1 # maximum firing rate

        self.response_history = False # by default response history filter
                                      # is not computed, call `add_response_history_fitler` if needed.

    def STC(self, transform=None, n_repeats=10, percentile=100., random_seed=1990, verbal=5):
        
        """
        
        Spike-triggered Covariance Analysis. 
        
        Parameters
        ==========
        
        transform: None or Str
            * None - Original X is used
            * 'whiten' - pre-whiten X
            * 'spline' - pre-whiten and smooth X by spline
            
        n_repeats: int
            Number of repeats for STC significance test.
            
        percentile: float
            Valid range of STC significance test.
        
        """        
        
        def get_stc(X, y, sta):

            n = len(X)
            ste = X[y!=0]
            proj = ste - ste * sta * sta.T
            stc = proj.T @ proj / (n - 1)

            eigvec, eigval, _ = np.linalg.svd(stc)

            return eigvec, eigval

        key = random.PRNGKey(random_seed)
        
        y = self.y
        
        if transform is not None:
            
            if transform == 'whiten':
                X = np.linalg.solve(self.X.T @ self.X, self.X.T).T
                sta = norm(self.w_mle)
                
            if transform == 'spline':
                X = (self.S @ np.linalg.solve(self.XS.T @ self.XS, self.XS.T)).T
                sta = norm(self.w_spl)
        
        else:
            X = self.X
            sta = norm(self.w_sta)

        eigvec, eigval = get_stc(X, y, sta)

        if n_repeats:
            eigval_null = []
            for counter in range(n_repeats):
                if counter % int(verbal) == 0:
                    print(f'{counter+1:}/{n_repeats}')
                
                y_randomize = random.permutation(key, y)
                _, eigval_randomize = get_stc(X, y_randomize, sta)
                eigval_null.append(eigval_randomize)
            else:
                print(f'{counter+1}/{n_repeats}. Done.')
            eigval_null = np.vstack(eigval_null)
            max_null, min_null = np.percentile(eigval_null, percentile), np.percentile(eigval_null, 100-percentile)
            mask_sig_pos = eigval > max_null
            mask_sig_neg = eigval < min_null
            mask_sig = np.logical_or(mask_sig_pos, mask_sig_neg)

            self.w_stc_pos = eigvec[:, mask_sig_pos]
            self.w_stc_neg = eigvec[:, mask_sig_neg]
            self.w_stc_eigval = eigval
            self.w_stc_eigval_mask = mask_sig 
            self.w_stc_max_null = max_null
            self.w_stc_min_null = min_null
            
        else:
            self.w_stc = eigvec
            self.w_stc_eigval = eigval


    def add_response_history_filter(self, dims, df, smooth='cr'):
        
        """
        Parameters
        ==========
        
        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]
        
        df : list or array_list
            number of basis.
        """
        

        y = self.y
        Sh = np.array(build_spline_matrix(dims[:1], df[:1], smooth)) # for h
        yh = np.array(build_design_matrix(self.y[:, np.newaxis], Sh.shape[0], shift=1)) 
        yS = yh @ Sh

        self.yh = np.array(yh)    
        self.Sh = Sh # spline basis for spike-history
        self.yS = yS        
        self.bh_spl = np.linalg.solve(yS.T @ yS, yS.T @ y)
        self.h_spl = Sh @ self.bh_spl

        self.response_history = True

    def fit_nonlinearity(self, nbin=50, df=7, w='w_spl'):
        
        """
        
        Estimate nonlinearity with nonparametric method, then
        interpolate with spline.
        
        Parameters
        ==========
        
        nbin: int
            Number of bins for histogram.
        
        df : int
            Number of basis for spline.
        
        w : str or array_lik
            RF used for nonlinearity estimation. 
            * 'w_sta': spike-triggered average
            * 'w_mle': maximum-likelihood
            * 'w_spl': spline-interpolated RF.        
            * A RF computed in other ways (e.g. STC) can be feed 
                in as a numpy array. 
        
        """

        if type(w) is str:
            if w == 'w_sta':
                w = self.w_sta
            elif w == 'w_mle':
                w = self.w_mle
            elif w == 'w_spl':
                w = self.w_spl
        else:
            w = np.array(w)

        B = np.array(build_spline_matrix(dims=[nbin,], df=[df,], smooth='cr'))

        output_raw = self.X @ norm(w)
        output_spk = self.X[self.y!=0] @ norm(w)

        hist_raw, bins = np.histogram(output_raw, bins=nbin, density=True)
        hist_spk, _ = np.histogram(output_spk, bins=bins, density=True)

        mask = ~(hist_raw ==0)
        
        yy0 = hist_spk[mask] / hist_raw[mask]
        yy = interp1d(bins[1:][mask], yy0)(bins[1:])
        
        b0 = np.ones(B.shape[1])
        func = lambda b: np.mean((yy - B @ b)**2)

        bnl = minimize(func, b0).x

        self.fitted_nonlin = interp1d(bins[1:], B @ bnl)
        self.nonparam_nonlin = yy
        self.bins = bins[1:]


    def nonlin(self, x, nl):
        
        '''
        Choose a fixed nonlinearity or fit a flexible one ('nonparametric').
        '''

        if  nl == 'softplus':
            return np.log(1 + np.exp(x)) + 1e-7
        
        elif nl == 'exponential':
            return np.exp(x)
        
        elif nl == 'softmax':
            z = np.exp(x)
            return z / z.sum()
        
        elif nl == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        
        elif nl == 'tanh':
            return 2 / (1 + np.exp(-2*x)) - 1
        
        elif nl == 'relu':
            return np.where(x > 0, x, 1e-15) 

        elif nl == 'leaky_relu':
            return np.where(x > 0, x, x * 0.01)

        elif nl == 'selu':
            return 1.0507 * np.where(x > 0, x, 1.6733 * np.exp(x) - 1.6733)

        elif nl == 'swish':
            return x / ( 1 + np.exp(-x))
        
        elif nl == 'elu':
            return np.where(x > 0, x, np.exp(x)-1)
       
        elif nl == 'none':
            return x
       
        elif nl == 'nonparametric':
            return np.maximum(self.fitted_nonlin(x), 1e-7)

        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def cost(self, b):
        pass


    def optimize_params(self, p0, num_iters, step_size, tolerance, verbal):

        """

        Gradient descent using JAX optimizer.
        
        """
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            g = grad(self.cost)(p)
            return opt_update(i, g, opt_state)

        cost_list = []
        cost_history = []
        params_list = []    

        if verbal:
            print('{0}\t{1}\t'.format('Iter', 'Cost'))

        for i in range(num_iters):
            
            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.cost(params_list[-1]))
            cost_history.append(self.cost(params_list[-1]))

            if verbal:
                if i % int(verbal) == 0:
                    print('{0}\t{1:.3f}\t'.format(i,  cost_list[-1]))
            
            if len(params_list) > tolerance:
                
                if len(cost_history) > 300 and np.all((np.array(cost_list[1:])) - np.array(cost_list[:-1]) > 0 ):
                    params = params_list[0]
                    if verbal:
                        print('Stop at {} steps: cost has been monotonically increasing for {} steps.'.format(i, tolerance))
                    break
                elif len(cost_history) > 300 and np.all(np.array(cost_list[:-1]) - np.array(cost_list[1:]) < 1e-5):
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
                print('Stop: reached {0} steps, final cost={1:.5f}.'.format(num_iters, cost_list[-1]))
        self.cost_history = cost_history

        return params


    def fit(self, p0=None, num_iters=5, alpha=1, lambd=0.5,
            step_size=1e-2, tolerance=10, verbal=1):
            
        """

        Parameters
        ==========

        p0 : dict 
            * 'b': Initial spline coefficients.
            * 'bh': Initial response history filter coefficients

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
            if self.response_history:
                p0 = {'b': self.b_spl, 'bh': self.bh_spl}
            else:
                p0 = {'b': self.b_spl, 'bh': None}
        
        self.p_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        self.b_opt = self.p_opt['b']
        self.w_opt = self.S @ self.b_opt
        
        if self.response_history:
            self.h_opt = self.Sh @ self.p_opt['bh']

    def _rcv(self, w, wSTA_test, X_test, y_test):

        """Relative Mean Squared Error"""

        a = mean_squared_error(y_test, X_test @ w)
        b = mean_squared_error(y_test, X_test @ wSTA_test)

        return a - b
        

    def measure_prediction_performance(self, X_test, y_test):

        wSTA_test = np.linalg.solve(X_test.T @ X_test, X_test.T @ y_test)

        w = self.w_opt.ravel()

        return self._rcv(w, wSTA_test, X_test, y_test)

class interp1d:

    """
    1D linear intepolation.

    usage:
        x = np.linspace(-5, 5, 10)
        y = np.cos(x)
        f = interp1d(x, y)
        new_x = np.linspace(-5, 5, 100)
        new_y = f(new_x)
    """

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.slopes = np.diff(y) / np.diff(x)

    def __call__(self, x_new):

        i = np.searchsorted(self.x, x_new) - 1
        i = np.where(i == -1, 0, i)
        i = np.where(i == len(self.x) - 1, -1, i)

        return self.y[i] + self.slopes[i] * (x_new - self.x[i])

def norm(x):
    return x / np.linalg.norm(x)
