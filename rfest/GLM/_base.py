import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers

from jax.config import config
config.update("jax_enable_x64", True)

import time
import itertools

from .._utils import build_design_matrix, uvec
from .._splines import build_spline_matrix
from .._metrics import accuracy, r2, mse, corrcoef

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

        # store meta

        self.ndim = len(dims)
        if self.ndim == 4: # [t, x, y, c]
            self.n_samples, self.n_features, self.n_c = X.shape 
            self.dims = dims[:-1] 
        else:
            self.n_samples, self.n_features = X.shape
            self.dims = dims # assumed order [t, y, x]
        
        self.dt = kwargs['dt'] if 'dt' in kwargs.keys() else 1 # time bin size (for LNP and LNLN)
        self.R = kwargs['R'] if 'R' in kwargs.keys() else 1 # scale factor for firing rate (for LNP and LNLN)

        self.compute_mle = compute_mle

        # compute sufficient statistics

        self.XtY = X.T @ y
        if (y == y.astype(int)).all(): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)

        
        if hasattr(self, 'n_c') : 
            self.w_sta = self.w_sta.reshape(self.n_features, self.n_c)

        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)
            if hasattr(self, 'n_c'): 
                self.w_mle = self.w_mle.reshape(self.n_features, self.n_c)       

        self.X = np.array(X) # stimulus design matrix
        self.y = np.array(y) # response


    def STC(self, prewhiten=False, n_repeats=10, percentile=100., random_seed=2046, verbose=5):

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
            w = uvec(self.w_mle)

        else:
            X = self.X
            w = uvec(self.w_sta)

        eigvec, eigval = get_stc(X, y, w)
        if n_repeats:
            print('STC significance test: ')
            eigval_null = []
            for counter in range(n_repeats):
                if verbose:
                    if counter % int(verbose) == 0:
                        print(f'  {counter+1:}/{n_repeats}')

                y_randomize = random.permutation(key, y)
                _, eigval_randomize = get_stc(X, y_randomize, w)
                eigval_null.append(eigval_randomize)
            else:
                if verbose:
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


    def initialize_history_filter(self, dims, shift=1):

        """
        Parameters
        ==========

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the response-history filter. It should be 1D [nt, ]

        """
        y = self.y
        yh = np.array(build_design_matrix(y[:, np.newaxis], dims, shift=shift))
        self.yh = np.array(yh)
        self.h_mle = np.linalg.solve(yh.T @ yh, yh.T @ y)
        
    
    def initialize_nonlinearity(self, nbin=50, df=7, w=None):

        """

        Estimate nonlinearity with nonparametric method, then
        interpolate with spline.

        Parameters
        ==========

        nbin: int
            Number of bins for histogram.

        df : int
            Number of basis for spline.

        w : None or array_lik
            RF used for nonlinearity estimation. If w=None, 
            w will search for a pre-calculated RF, in order of
            `w_spl`, `w_mle`, `w_sta`. You can also feed in a
            RF calculated in another way as a numpy array. 

        """

        if w is None:
            if hasattr(self, 'w_spl'):
                w = self.w_spl
            elif hasattr(self, 'w_mle'):
                w = self.w_mle
            elif hasattr(self, 'w_sta'):
                w = self.w_sta
        else:
            w = np.array(w)

        Snl = np.array(build_spline_matrix(dims=[nbin,], df=[df,], smooth='cr'))

        output_raw = self.X @ uvec(w)
        output_spk = self.X[self.y!=0] @ uvec(w)

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
        self.bins = bins[1:]
        self.fitted_nonlinearity = interp1d(self.bins, Snl @ bnl)
        self.nl0 = self.fitted_nonlinearity(self.bins)
        self.nonparametric_nl = yy

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
            if np.ndim(self.p0['bnl']) > 1:
                return np.maximum(np.vstack([self.fitted_nonlinearity[i](x[:, i]) for i in range(self.n_subunits)]), 1e-7).T
            else:
                return np.maximum(self.fitted_nonlinearity(x), 1e-7)

        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def cost(self, w, extra):
        pass


    def optimize_params(self, p0, extra, num_iters, metric, step_size, tolerance, verbose):

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

        cost_train = []
        cost_dev = []
        metric_train = []
        metric_dev = []
        params_list = []

        if verbose:
            if extra is None:
                if metric is not None:
                    print('{0}\t{1:>10}\t{2:>10}'.format('Iters', 'Time (s)', 'Cost (train)'))
                else:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Metric (train)')) 
            else:
                if metric is None:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Cost (dev)'))
                else:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Cost (dev)', 'Metric (train)', 'Metric (dev)')) 

        time_start = time.time()
        for i in range(num_iters):

            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))

            y_pred_train = self.forward_pass(p=params_list[-1], extra=None)
            cost_train.append(self.cost(p=params_list[-1], precomputed=y_pred_train))
            if extra is not None:
                y_pred_dev = self.forward_pass(p=params_list[-1], extra=extra)
                cost_dev.append(self.cost(p=params_list[-1], extra=extra, precomputed=y_pred_dev))

            if metric is not None:
                
                metric_train.append(self._score(self.y, y_pred_train, metric))
                if extra is not None:
                    metric_dev.append(self._score(extra['y'], y_pred_dev, metric))                

            time_elapsed = time.time() - time_start
            if verbose:
                if i % int(verbose) == 0:
                    if extra is None:
                        if metric is None:
                           print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}'.format(i, time_elapsed, cost_train[-1]))
                        else:
                           print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}'.format(i, time_elapsed, cost_train[-1], metric_train[-1])) 

                    else:
                        if metric is None:
                            print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}'.format(i, time_elapsed, cost_train[-1], cost_dev[-1]))
                        else:
                            print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}\t{4:>10.3f}\t{5:>10.3f}'.format(i, time_elapsed, cost_train[-1], cost_dev[-1], metric_train[-1], metric_dev[-1]))

            if tolerance and len(params_list) > tolerance: # tolerance = 0: no early stop.

                total_time_elapsed = time.time() - time_start

                if i > 300 and np.all((np.array(cost_dev[-tolerance+1:]) - np.array(cost_dev[-tolerance:-1])) > 0) and extra is not None:
                    params = params_list[0]
                    if verbose:
                        print('Stop at {0} steps: dev cost has been monotonically increasing for {1} steps, total time elapsed = {2:.03f} s'.format(i, tolerance, total_time_elapsed))
                    break
                
                if i > 300 and np.all(np.array(cost_train[-tolerance:-1]) - np.array(cost_train[-tolerance+1:]) < 1e-5):
                    params = params_list[-1]
                    if verbose:
                        print('Stop at {0} steps: training cost has been changing less than 1e-5 for {1} steps, total time elapsed = {2:.03f} s'.format(i, tolerance, total_time_elapsed))
                    break
                
                params_list.pop(0)
        else:
            params = params_list[-1]
            total_time_elapsed = time.time() - time_start

            if verbose:
                print('Stop: reached {0} steps, total time elapsed= {1:.3f} s.'.format(num_iters, total_time_elapsed))
        
        self.cost_train = cost_train
        self.cost_dev = cost_dev
        self.metric_train = metric_train
        self.metric_dev = metric_dev

        return params

    def fit(self, p0=None, extra=None, initialize=None,
            num_iters=5, metric=None, alpha=1, beta=0.5,
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False, 
            step_size=1e-2, tolerance=10, verbose=1, random_seed=1990):

        """

        Parameters
        ==========

        p0 : dict
            * 'b': Initial spline coefficients.
            * 'bh': Initial response history filter coefficients

        extra : None or dict {'X': X_dev, 'y': y_dev}
            Development set.

        initialize : None or str
            Paramteric initalization. 
            * if `initialize=None`, `w` will be initialized by STA.
            * if `initialize='random'`, `w` will be randomly initialized.

        num_iters : int
            Max number of optimization iterations.

        metric : None or str
            Extra cross-validation metric. Default is `None`. Or
            * 'mse': mean squared error
            * 'r2': R2 score
            * 'corrcoef': Correlation coefficient

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

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.metric = metric # metric for cross-validation and prediction

        self.beta = beta
        self.alpha = alpha
        self.num_iters = num_iters

        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter, 
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        # initialize parameters 
        if p0 is None:
            p0 = {}

        dict_keys = p0.keys()
        if 'w' not in dict_keys:
            if initialize is None:
                p0.update({'w': self.w_sta})
            else:
                if initialize == 'random':
                    key = random.PRNGKey(random_seed)
                    w0 = 0.01 * random.normal(key, shape=(self.w_sta.shape[0], )).flatten()
                    p0.update({'w': w0})


        if 'intercept' not in dict_keys:
            p0.update({'intercept': np.array([0.])})

        if 'R' not in dict_keys and self.fit_R:
            p0.update({'R': np.array([1.])})

        if 'h' not in dict_keys:
            if hasattr(self, 'h_mle'):
                p0.update({'h': self.h_mle})            
            else:
                p0.update({'h': None})  

        if 'bnl' not in dict_keys:
            if hasattr(self, 'bnl'):
                p0.update({'bnl': self.bnl})
            else:
                p0.update({'bnl': None})
        else:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p0['bnl'])

        if extra is not None:
            
            if hasattr(self, 'h_mle'):
                yh = np.array(build_design_matrix(extra['y'][:, np.newaxis], self.yh.shape[1], shift=1))
                extra.update({'yh': yh}) 

            extra = {key: np.array(extra[key]) for key in extra.keys()}

        # store optimized parameters
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_iters, metric, step_size, tolerance, verbose)
        self.R = self.p_opt['R'] if fit_R else np.array([1.])

        if fit_linear_filter:
            self.w_opt = self.p_opt['w']
        
        if fit_history_filter:
            self.h_opt = self.p_opt['h']
        
        if fit_nonlinearity:
            self.bnl_opt = self.p_opt['bnl']
        
        if fit_intercept:
            self.intercept = self.p_opt['intercept']

    def predict(self, X, y=None, p=None):
        
        extra = {'X': X, 'y': y}
        if hasattr(self, 'h_mle'):

            if y is None:
                raise ValueError('`y` is needed for calculating response history.')
            
            yh = np.array(build_design_matrix(extra['y'][:, np.newaxis], self.yh.shape[1], shift=1))
            extra.update({'yh': yh}) 

        params = self.p_opt if p is None else p
        y_pred = self.forward_pass(params, extra=extra)

        return y_pred

    def _score(self, y, y_pred, metric):

        if metric == 'r2':
            return r2(y, y_pred)
        elif metric == 'mse':
            return mse(y, y_pred)
        elif metric == 'corrcoef':
            return corrcoef(y, y_pred)
        else:
            print(f'Metric `{metric}` is not supported.')

    def score(self, X, y, p=None, metric='corrcoef'):

        y_pred = self.predict(X, y, p)
        return self._score(y, y_pred, metric)
        


class splineBase(Base):

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False, **kwargs):

        super().__init__(X, y, dims, compute_mle, **kwargs) 
        
        self.df = df # number basis / degree of freedom
        self.smooth = smooth # type of basis

        S = np.array(build_spline_matrix(self.dims, df, smooth)) # for w
        
        if hasattr(self, 'n_c'):
            XS = np.dstack([self.X[:, :, i] @ S for i in range(self.n_c)]).reshape(self.n_samples, -1)
        else:
            XS = self.X @ S

        self.S = S # spline matrix
        self.XS = XS

        self.n_b = S.shape[1] # num:ber of spline coefficients

        # compute spline-based maximum likelihood
        self.b_spl = np.linalg.solve(XS.T @ XS, XS.T @ y)

        if hasattr(self, 'n_c'): 
            self.w_spl = S @ self.b_spl.reshape(self.n_b, self.n_c)
        else:
            self.w_spl = S @ self.b_spl 
        

    def cost(self, b, extra):
        pass


    def initialize_history_filter(self, dims, df, smooth='cr', shift=1):

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
        yh = np.array(build_design_matrix(self.y[:, np.newaxis], Sh.shape[0], shift=shift))
        yS = yh @ Sh

        self.yh = np.array(yh)
        self.Sh = Sh # spline basis for spike-history
        self.yS = yS
        self.bh_spl = np.linalg.solve(yS.T @ yS, yS.T @ y)
        self.h_spl = Sh @ self.bh_spl


    def fit(self, p0=None, extra=None, initialize=None,
            num_iters=5, metric=None, alpha=1, beta=0.5,
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False, 
            step_size=1e-2, tolerance=10, verbose=1, random_seed=1990):

        """

        Parameters
        ==========

        p0 : dict
            * 'b': Initial spline coefficients.
            * 'bh': Initial response history filter coefficients

        initialize : None or str
            Paramteric initalization. 
            * if `initialize=None`, `b` will be initialized by b_spl.
            * if `initialize='random'`, `b` will be randomly initialized.

        num_iters : int
            Max number of optimization iterations.

        metric : None or str
            Extra cross-validation metric. Default is `None`. Or
            * 'mse': mean squared error
            * 'r2': R2 score
            * 'corrcoef': Correlation coefficient

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

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.metric = metric

        self.beta = beta
        self.alpha = alpha
        self.num_iters = num_iters

        self.fit_R = fit_R
        self.fit_linear_filter = fit_linear_filter
        self.fit_history_filter = fit_history_filter
        self.fit_nonlinearity = fit_nonlinearity
        self.fit_intercept = fit_intercept

        # initial parameters

        if p0 is None:
            p0 = {}    
                       
        dict_keys = p0.keys()
        if 'b' not in dict_keys:
            if initialize is None:
                p0.update({'b': self.b_spl})
            else:
                if initialize == 'random':
                    key = random.PRNGKey(random_seed)
                    b0 = 0.01 * random.normal(key, shape=(self.b_spl.shape[0], )).flatten()
                    p0.update({'b': b0})

        if 'intercept' not in dict_keys:
            p0.update({'intercept': np.array([0.])})

        if 'R' not in dict_keys:
            p0.update({'R': np.array([1.])})

        if 'bh' not in dict_keys:
            if hasattr(self, 'bh_spl'):
                p0.update({'bh': self.bh_spl})  
            else:
                p0.update({'bh': None}) 

        if 'bnl' not in dict_keys:
            if hasattr(self, 'bnl'):
                p0.update({'bnl': self.bnl})
            else:
                p0.update({'bnl': None})
        else:
            self.fitted_nonlinearity = interp1d(self.bins, self.Snl @ p0['bnl'])

        if extra is not None:

            if hasattr(self, 'n_c'):
                XS_ext = np.dstack([extra['X'][:, :, i] @ self.S for i in range(self.n_c)]).reshape(extra['X'].shape[0], -1)
                extra.update({'XS': XS_ext}) 
            else:
                extra.update({'XS': extra['X'] @ self.S})
            
            if hasattr(self, 'h_spl'):
                
                yh_ext = np.array(build_design_matrix(extra['y'][:, np.newaxis], self.Sh.shape[0], shift=1))
                yS_ext = yh_ext @ self.Sh
                extra.update({'yS': yS_ext}) 
            
            extra = {key: np.array(extra[key]) for key in extra.keys()}

            self.extra = extra # store for cross-validation

        # store optimized parameters
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_iters, metric, step_size, tolerance, verbose)
        self.R = self.p_opt['R'] if fit_R else np.array([1.])

        if fit_linear_filter:
            self.b_opt = self.p_opt['b'] # optimized RF basis coefficients
            if hasattr(self, 'n_c'):
                self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_c)  
            else:
                self.w_opt = self.S @ self.b_opt # optimized RF
        
        if fit_history_filter:
            self.bh_opt = self.p_opt['bh']
            self.h_opt = self.Sh @ self.bh_opt

        if fit_nonlinearity:
            self.bnl_opt = self.p_opt['bnl']
       
        if fit_intercept:
            self.intercept = self.p_opt['intercept']

    def predict(self, X, y=None, p=None):
        
        if hasattr(self, 'n_c'):
            XS = np.dstack([X[:, :, i] @ self.S for i in range(self.n_c)]).reshape(X.shape[0], -1)
        else:
            XS = X @ self.S
        
        extra = {'X': X, 'XS': XS, 'y': y}

        if hasattr(self, 'h_spl'):

            if y is None:
                raise ValueError('`y` is needed for calculating response history.')
            
            yh = np.array(build_design_matrix(extra['y'][:, np.newaxis], self.Sh.shape[0], shift=1))
            yS = yh @ self.Sh
            extra.update({'yS': yS}) 

        params = self.p_opt if p is None else p
        y_pred = self.forward_pass(params, extra=extra)

        return y_pred

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
