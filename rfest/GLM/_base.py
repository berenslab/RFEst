import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import mean_squared_error

from .._utils import build_design_matrix, norm
from .._splines import build_spline_matrix
from scipy.optimize import minimize

__all__ = ['Base', 'splineBase']

class Base:

    """

    Base class for spline-based GLMs.

    """

    def __init__(self, X, y, dims, compute_mle=False, **kwargs):

        """

        Parameters
        ==========
        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : array_like, shape (n_samples, )
            Recorded response

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sy, sx]

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """

        # store meta data

        self.dims = dims # assumed order [t, y, x]
        self.ndim = len(dims)
        self.n_samples, self.n_features = X.shape

        self.dt = kwargs['dt'] if 'dt' in kwargs.keys() else 1 # time bin size (for LNP and LNLN)
        self.R = kwargs['R'] if 'R' in kwargs.keys() else 1 # a constant for scaling firing rate.

        self.compute_mle = compute_mle

        # compute sufficient statistics

        self.XtY = X.T @ y
        if (y == y.astype(int)).all(): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)

        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)

        self.X = np.array(X) # stimulus design matrix
        self.y = np.array(y) # response


    def STC(self, prewhiten=False, n_repeats=10, percentile=100., random_seed=2046, verbal=5):

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

        def get_stc(X, y, w):

            n = len(X)
            ste = X[y!=0]
            proj = ste - ste * w * w.T
            stc = proj.T @ proj / (n - 1)

            eigvec, eigval, _ = np.linalg.svd(stc)

            return eigvec, eigval

        key = random.PRNGKey(random_seed)

        y = self.y

        if prewhiten:

            if self.compute_mle is False:
                self.XtX = self.X.T @ self.X
                self.w_mle = np.linalg.solve(self.XtX, self.XtY)
            
            X = np.linalg.solve(self.XtX, self.X.T).T
            w = norm(self.w_mle)

        else:
            X = self.X
            w = norm(self.w_sta)

        eigvec, eigval = get_stc(X, y, w)
        if n_repeats:
            print('STC significance test: ')
            eigval_null = []
            for counter in range(n_repeats):
                if verbal:
                    if counter % int(verbal) == 0:
                        print(f'  {counter+1:}/{n_repeats}')

                y_randomize = random.permutation(key, y)
                _, eigval_randomize = get_stc(X, y_randomize, w)
                eigval_null.append(eigval_randomize)
            else:
                if verbal:
                    print(f' Done.')
            eigval_null = np.vstack(eigval_null)
            max_null, min_null = np.percentile(eigval_null, percentile), np.percentile(eigval_null, 100-percentile)
            mask_sig_pos = eigval > max_null
            mask_sig_neg = eigval < min_null
            mask_sig = np.logical_or(mask_sig_pos, mask_sig_neg)

            self.w_stc = eigvec
            self.w_stc_pos = eigvec[:, mask_sig_pos]
            self.w_stc_neg = eigvec[:, mask_sig_neg]

            self.w_stc_eigval = eigval
            self.w_stc_eigval_mask = mask_sig
            self.w_stc_eigval_pos_mask = mask_sig_pos
            self.w_stc_eigval_neg_mask = mask_sig_neg

            self.w_stc_max_null = max_null
            self.w_stc_min_null = min_null

        else:
            self.w_stc = eigvec
            self.w_stc_eigval = eigval
            self.w_stc_eigval_mask = np.ones_like(eigval).astype(bool)


    def initialize_response_history_filter(self, dims):

        """
        Parameters
        ==========

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]

        df : list or array_list
            number of basis.
        """
        y = self.y
        yh = np.array(build_design_matrix(y[:, np.newaxis], dims, shift=1))
        self.yh = np.array(yh)
        self.h_mle = np.linalg.solve(yh.T @ yh, yh.T @ y)
        
    
    def initialize_nonlinearity(self, nbin=50, df=7, w='w_sta'):

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

        Snl = np.array(build_spline_matrix(dims=[nbin,], df=[df,], smooth='cr'))

        output_raw = self.X @ norm(w)
        output_spk = self.X[self.y!=0] @ norm(w)

        hist_raw, bins = np.histogram(output_raw, bins=nbin, density=True)
        hist_spk, _ = np.histogram(output_spk, bins=bins, density=True)

        mask = ~(hist_raw ==0)

        yy0 = hist_spk[mask] / hist_raw[mask]
        yy = interp1d(bins[1:][mask], yy0)(bins[1:])

        b0 = np.ones(Snl.shape[1])
        func = lambda b: np.mean((yy - Snl @ b)**2)

        bnl = minimize(func, b0).x

        self.Snl = Snl
        self.bnl = bnl
        self.fitted_nonlinearity = interp1d(bins[1:], Snl @ bnl)
        self.nonparametric_nonlinearity = yy
        self.bins = bins[1:]

    def fnl(self, x, nl):

        '''
        Choose a fixed nonlinear function or fit a flexible one ('nonparametric').
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
            return np.maximum(self.fitted_nonlinearity(x), 1e-7)

        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def cost(self, w):
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

    def fit(self, p0=None, num_iters=5, alpha=1, beta=0.5,
            fit_linear_filter=True, fit_history_filter=False, 
            fit_nonlinearity=False, fit_intercept=True,
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

        beta : float
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

        self.beta = beta
        self.alpha = alpha
        self.num_iters = num_iters

        self.fit_linear_filter = fit_linear_filter, 
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        if type(p0) is str:
            if p0 == 'opt':
                p0 = {'b': self.w_opt}     
                p0.update({'h': self.h_opt}) if self.fit_history_filter else p0.update({'h': None})
                p0.update({'intercept': 0.}) if self.fit_intercept else p0.update({'intercept': None})
        else:
            if p0 is None: # if p0 is not provided, initialize it with spline MLE.
                p0 = {'w': self.w_sta}
                p0.update({'h': self.h_mle}) if self.fit_history_filter else p0.update({'h': None})
                p0.update({'intercept': 0.}) if self.fit_intercept else p0.update({'intercept': None})

        # store optimized parameters
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        self.w_opt = self.p_opt['w'] if self.fit_linear_filter else w_opt
        self.h_opt = self.p_opt['h'] if self.fit_history_filter else None
        self.bnl_opt = self.p_opt['bnl'] if self.fit_nonlinearity else None
        self.intercept = self.p_opt['intercept'] if self.fit_intercept else 0


class splineBase(Base):


    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False, **kwargs):

        super().__init__(X, y, dims, compute_mle, **kwargs) 
        
        self.df = df # number basis / degree of freedom
        self.smooth = smooth # type of basis

        S = np.array(build_spline_matrix(dims, df, smooth)) # for w
        XS = self.X @ S
        self.S = S # spline matrix
        self.XS = XS

        self.n_b = S.shape[1] # num:ber of spline coefficients

        # compute spline-based maximum likelihood
        self.b_spl = np.linalg.solve(XS.T @ XS, XS.T @ y)
        self.w_spl = S @ self.b_spl
        

    def cost(self, b):
        pass


    def initialize_response_history_filter(self, dims, df, smooth='cr'):

        """
        Parameters
        ==========

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]

        df : list or array_list
            number of basis.
        """


        y = self.y
        Sh = np.array(build_spline_matrix([dims, ], [df, ], smooth)) # for h
        yh = np.array(build_design_matrix(self.y[:, np.newaxis], Sh.shape[0], shift=1))
        yS = yh @ Sh

        self.yh = np.array(yh)
        self.Sh = Sh # spline basis for spike-history
        self.yS = yS
        self.bh_spl = np.linalg.solve(yS.T @ yS, yS.T @ y)
        self.h_spl = Sh @ self.bh_spl


    def fit(self, p0=None, num_iters=5, alpha=1, beta=0.5,
            fit_linear_filter=True, fit_history_filter=False, 
            fit_nonlinearity=False, fit_intercept=True,
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

        beta : float
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

        self.beta = beta
        self.alpha = alpha
        self.num_iters = num_iters

        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        # initial parameters
        if type(p0) is str:
            if p0 == 'opt':
                p0 = {'b': self.b_opt}
                p0.update({'bh': self.bh_opt}) if self.fit_history_filter else p0.update({'bh': None})
                p0.update({'bnl': self.bnl_opt}) if self.fit_nonlinearity else p0.update({'bnl': None})
                p0.update({'intercept': self.intercept}) if self.fit_intercept else p0.update({'intercept': None})                
        else:
            if p0 is None: # if p0 is not provided, initialize it with spline MLE.
                p0 = {'b': self.b_spl}
                p0.update({'bh': self.bh_spl}) if self.fit_history_filter else p0.update({'bh': None})
                p0.update({'bnl': self.bnl}) if self.fit_nonlinearity else p0.update({'bnl': None})
                p0.update({'intercept': 0.}) if self.fit_intercept else p0.update({'intercept': None})
        
        # store optimized parameters
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, num_iters, step_size, tolerance, verbal)
        
        self.b_opt = self.p_opt['b'] if fit_linear_filter else self.b_opt # optimized RF basis coefficients
        self.w_opt = self.S @ self.b_opt # optimized RF
        
        self.bh_opt = self.p_opt['bh'] if self.fit_history_filter else None
        self.h_opt = self.Sh @ self.bh_opt if self.fit_history_filter else None
        
        self.bnl_opt = self.p_opt['bnl'] if self.fit_nonlinearity else None
        self.intercept = self.p_opt['intercept'] if self.fit_intercept else 0.



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

if __name__ == "__main__": 

    import doctest
    doctest.testmod(verbose=True)
