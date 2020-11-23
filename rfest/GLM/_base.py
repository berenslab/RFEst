import jax.numpy as np
import jax.random as random
from jax import grad
from jax import jit
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, BatchNorm, Relu
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import time
import itertools

from ..utils import build_design_matrix, uvec
from ..splines import build_spline_matrix, cr, cc, bs
from ..metrics import accuracy, r2, mse, corrcoef
from ..nonlinearities import *

from scipy.optimize import minimize

__all__ = ['Base', 'splineBase']

class Base:

    """

    Base class for all GLMs.

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
            self.n_c = 1
            self.dims = dims # assumed order [t, y, x]
        
        self.dt = kwargs['dt'] if 'dt' in kwargs.keys() else 1 # time bin size (for LNP and LNLN)
        self.compute_mle = compute_mle

        # compute sufficient statistics

        self.XtY = X.T @ y
        if (y == y.astype(int)).all(): # if y is spike
            self.w_sta = self.XtY / sum(y)
        else:                                 # if y is not spike
            self.w_sta = self.XtY / len(y)

        
        if self.n_c > 1: 
            self.w_sta = self.w_sta.reshape(self.n_features, self.n_c)

        if compute_mle:
            self.XtX = X.T @ X
            self.w_mle = np.linalg.lstsq(self.XtX, self.XtY, rcond=None)[0]
            if self.n_c > 1: 
                self.w_mle = self.w_mle.reshape(self.n_features, self.n_c)       

        self.X = np.array(X) # stimulus design matrix
        self.y = np.array(y) # response


    def fit_STC(self, prewhiten=False, n_repeats=10, percentile=100., random_seed=2046, verbose=5):

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
                    print(f'Done.')
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

        shift : int
            Should be 1 or larger. 

        """
        y = self.y
        yh = np.array(build_design_matrix(y[:, np.newaxis], dims, shift=shift))
        self.yh = np.array(yh)
        self.h_mle = np.linalg.solve(yh.T @ yh, yh.T @ y)

    def fit_nonparametric_nonlinearity(self, nbins=50, w=None):

        if w is None:
            if hasattr(self, 'w_spl'):
                w = self.w_spl.flatten()
            elif hasattr(self, 'w_mle'):
                w = self.w_mle.flatten()
            elif hasattr(self, 'w_sta'):
                w = self.w_sta.flatten()
        else:
            w = np.array(w)

        X = self.X
        X = X.reshape(X.shape[0], -1)
        y = self.y

        output_raw = X @ uvec(w)
        output_spk = X[y!=0] @ uvec(w)

        hist_raw, bins = np.histogram(output_raw, bins=nbins, density=True)
        hist_spk, _ = np.histogram(output_spk, bins=bins, density=True)

        mask = ~(hist_raw ==0)

        yy0 = hist_spk[mask] / hist_raw[mask]
    
        self.nl_bins = bins[1:]
        self.fnl_nonparametric = interp1d(bins[1:][mask], yy0)        

    def initialize_parametric_nonlinearity(self, init_to='exponential', method=None, params_dict=None):

        if method is None:
            if hasattr(self, 'nonlinearity'):
                method = self.nonlinearity
            else:
                method = self.filter_nonlinearity
        else:
            if hasattr(self, 'nonlinearity'):
                self.nonlinearity = method
            else:
                self.filter_nonlinearity = method   
         
        # prepare data 
        if params_dict is None: 
           params_dict = {}
        xrange = params_dict['xrange'] if 'xrange' in params_dict else 5 
        nx = params_dict['nx'] if 'nx' in params_dict else 1000
        x0 = np.linspace(-xrange, xrange, nx)
        if init_to == 'exponential':
            y0 = np.exp(x0)
            
        elif init_to == 'softplus':
            y0 = softplus(x0)

        elif init_to == 'relu':
            y0 = relu(x0)
            
        elif init_to == 'nonparametric':
            y0 = self.fnl_nonparametric(x0)

        elif init_to == 'gaussian':
            import scipy.signal
            y0 = scipy.signal.gaussian(nx, nx/10)
            
        # fit nonlin
        if method == 'spline':
            smooth = params_dict['smooth'] if 'smooth' in params_dict else 'cr'
            df = params_dict['df'] if 'df' in params_dict else 7
            if smooth == 'cr':
                X = cr(x0, df)
            elif smooth == 'cc':
                X = cc(x0, df)
            elif smooth == 'bs':
                deg = params_dict['degree'] if 'degree' in params_dict else 3
                X = bs(x0, df, deg)
            
            opt_params = np.linalg.pinv(X.T @ X) @ X.T @ y0

            self.nl_basis = X
            
            def _nl(opt_params, x_new):
                return np.maximum(interp1d(x0, X @ opt_params)(x_new), 0)
            
        elif method == 'nn':
                
            def loss(params, data):
                x = data['x']
                y = data['y']
                yhat = _predict(params, x)
                return np.mean((y - yhat)**2)
            
            @jit
            def step(i, opt_state, data):
                p = get_params(opt_state)
                g = grad(loss)(p, data)
                return opt_update(i, g, opt_state)

            random_seed = params_dict['random_seed'] if 'random_seed' in params_dict else 2046 
            key = random.PRNGKey(random_seed)

            step_size = params_dict['step_size'] if 'step_size' in params_dict else 0.01 
            layer_sizes = params_dict['layer_sizes'] if 'layer_sizes' in params_dict else [10, 10, 1]
            layers = []
            for layer_size in layer_sizes:
                layers.append(Dense(layer_size))
                layers.append(BatchNorm(axis=(0, 1)))
                layers.append(Relu)
            else:
                layers.pop(-1)

            init_random_params, _predict = stax.serial(
                *layers)

            num_subunits = params_dict['num_subunits'] if 'num_subunits' in params_dict else 1 
            _, init_params = init_random_params(key, (-1, num_subunits))

            opt_init, opt_update, get_params = optimizers.adam(step_size)
            opt_state = opt_init(init_params)

            num_iters = params_dict['num_iters'] if 'num_iters' in params_dict else 1000
            if num_subunits == 1: 
                data = {'x': x0.reshape(-1,1), 'y': y0.reshape(-1,1)}
            else:
                data = {'x': np.vstack([x0 for i in range(num_subunits)]).T, 'y': y0.reshape(-1,1)}

            for i in range(num_iters):
                opt_state = step(i, opt_state, data)
            opt_params = get_params(opt_state)
            
            def _nl(opt_params, x_new):
                if len(x_new.shape) == 1:
                    x_new = x_new.reshape(-1, 1)
                return np.maximum(_predict(opt_params, x_new), 0)
        
        self.nl_xrange = x0
        self.nl_params = opt_params
        self.fnl_fitted = _nl

    def fnl(self, x, nl, params=None):

        '''
        Choose a fixed nonlinear function or fit a flexible one ('nonparametric').
        '''

        if  nl == 'softplus':
            return softplus(x)

        elif nl == 'exponential':
            return np.exp(x)

        elif nl == 'softmax':
            return softmax(x)

        elif nl == 'sigmoid':
            return sigmoid(x)

        elif nl == 'tanh':
            return np.tanh(x)

        elif nl == 'relu':
            return relu(x)

        elif nl == 'leaky_relu':
            return leaky_relu(x)

        elif nl == 'selu':
            return selu(x)

        elif nl == 'swish':
            return swish(x)

        elif nl == 'elu':
            return elu(x)

        elif nl == 'none':
            return x

        elif nl == 'nonparametric':
            return self.fnl_nonparametric(x)

        elif nl == 'spline' or nl == 'nn':

            return self.fnl_fitted(params, x)

        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def cost(self, w, extra):
        pass


    def optimize_params(self, p0, extra, num_epochs, num_iters, metric, step_size, tolerance, verbose):

        """

        Gradient descent using JAX optimizer, and verbose logging. 

        """

        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            g = grad(self.cost)(p)
            return opt_update(i, g, opt_state)

        # preallocation
        cost_train = [0] * num_iters 
        cost_dev = [0] * num_iters
        metric_train = [0] * num_iters
        metric_dev = [0] * num_iters    
        params_list = [0] * num_iters

        if verbose:
            if extra is None:
                if metric is None:
                    print('{0}\t{1:>10}\t{2:>10}'.format('Iters', 'Time (s)', 'Cost (train)'))
                else:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Metric (train)')) 
            else:
                if metric is None:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Cost (dev)'))
                else:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Cost (dev)', 'Metric (train)', 'Metric (dev)')) 

        time_start = time.time()

        for epoch in range(num_epochs):   

            opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
            if epoch == 0:
                opt_state = opt_init(p0)
            else:
                opt_state = opt_init(params)

            if verbose and num_epochs > 1:
                
                print('\n===Epoch {0}==='.format(epoch))
            
            for i in range(num_iters):

                opt_state = step(i, opt_state)
                params_list[i] = get_params(opt_state)

                y_pred_train = self.forward_pass(p=params_list[i], extra=None)
                c_train = self.cost(p=params_list[i], precomputed=y_pred_train)
                cost_train[i] = c_train 
                
                if extra is not None:
                    y_pred_dev = self.forward_pass(p=params_list[i], extra=extra)
                    c_dev = self.cost(p=params_list[i], extra=extra, precomputed=y_pred_dev)
                    cost_dev[i] = c_dev

                if metric is not None:
                    
                    m_train = self._score(self.y, y_pred_train, metric)
                    metric_train[i] = m_train

                    if extra is not None:
                        m_dev = self._score(extra['y'], y_pred_dev, metric)
                        metric_dev[i] = m_dev

                time_elapsed = time.time() - time_start
                if verbose:
                    if i % int(verbose) == 0:
                        if extra is None:
                            if metric is None:
                                print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}'.format(i, time_elapsed, c_train))
                            else:
                                print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}'.format(i, time_elapsed, c_train, m_train)) 

                        else:
                            if metric is None:
                                print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}'.format(i, time_elapsed, c_train, c_dev))
                            else:
                                print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}\t{4:>10.3f}\t{5:>10.3f}'.format(i, time_elapsed, c_train, c_dev, m_train, m_dev))

                if tolerance and i > 300: # tolerance = 0: no early stop.

                    total_time_elapsed = time.time() - time_start
                    cost_train_slice = np.array(cost_train[i-tolerance:i])
                    cost_dev_slice = np.array(cost_dev[i-tolerance:i])
                    
                    if np.all(cost_dev_slice[1:] - cost_dev_slice[:-1] > 0):
                        params = params_list[i-tolerance]
                        if verbose:
                            print('Stop at {0} steps: cost (dev) has been monotonically increasing for {1} steps.\n'.format(i, tolerance))
                        break
                    
                    if np.all(cost_train_slice[:-1] - cost_train_slice[1:] < 1e-5):
                        params = params_list[i]
                        if verbose:
                            print('Stop at {0} steps: cost (train) has been changing less than 1e-5 for {1} steps.\n'.format(i, tolerance))
                        break
            
            else:
                params = params_list[i]
                total_time_elapsed = time.time() - time_start

                if verbose:
                    print('Stop: reached {0} steps.\n'.format(num_iters))
            
        else:    
            print('Total time elapsed: {0:.3f} s.'.format(total_time_elapsed))
            
        self.cost_train = cost_train[:i+1]
        self.cost_dev = cost_dev[:i+1]
        self.metric_train = metric_train[:i+1]
        self.metric_dev = metric_dev[:i+1]

        return params

    def fit(self, p0=None, extra=None, initialize='random',
            num_epochs=1, num_iters=5, metric=None, alpha=1, beta=0.05, 
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False, 
            step_size=1e-2, tolerance=10, verbose=1, random_seed=2046):

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
            Initial step size for JAX optimizer (ADAM).

        tolerance : int
            Set early stop tolerance. Optimization stops when cost (dev) monotonically
            increases or cost (train) stop increases for tolerance=n steps. 
            If `tolerance=0`, then early stop is not used.

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.

        """

        self.metric = metric # metric for cross-validation and prediction

        self.alpha = alpha
        self.beta = beta # elastic net parameter - global penalty weight for linear filter
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

        if 'nl_params' not in dict_keys:
            if hasattr(self, 'nl_params'):
                p0.update({'nl_params': self.nl_params})
            else:
                p0.update({'nl_params': None})

        if extra is not None:
            
            if hasattr(self, 'h_mle'):
                yh = np.array(build_design_matrix(extra['y'][:, np.newaxis], self.yh.shape[1], shift=1))
                extra.update({'yh': yh}) 

            extra = {key: np.array(extra[key]) for key in extra.keys()}

        # store optimized parameters
        self.p0 = p0
        self.p_opt = self.optimize_params(p0, extra, num_epochs, num_iters, metric, step_size, tolerance, verbose)
        self.R = self.p_opt['R'] if fit_R else np.array([1.])

        if fit_linear_filter:
            self.w_opt = self.p_opt['w']
        
        if fit_history_filter:
            self.h_opt = self.p_opt['h']
        
        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt['nl_params']
        
        if fit_intercept:
            self.intercept = self.p_opt['intercept']

    def predict(self, X, y=None, p=None):

        """

        Parameters
        ==========

        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : None or array_like, shape (n_samples, )
            Recorded response. Needed when post-spike filter is fitted.

        p : None or dict
            Model parameters. Only needed if model performance is monitored
            during training.

        """
        
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

        # Performance measure.

        y_pred = self.predict(X, y, p)

        return self._score(y, y_pred, metric)
        

class splineBase(Base):

    """

    Base class for spline-based GLMs.

    """

    def __init__(self, X, y, dims, df, smooth='cr', compute_mle=False, **kwargs):

        """

        Parameters
        ==========
        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : array_like, shape (n_samples, )
            Recorded response.

        dims : list or array_like, shape (ndims, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sx, sy].

        df : list or array_like, shape (ndims, )
            Degree of freedom, or the number of basis used for each RF dimension. 

        smooth : str
            Type of basis. 
            * cr: natrual cubic spline (default)
            * cc: cyclic cubic spline
            * bs: B-spline
            * tp: thin plate spine

        compute_mle : bool
            Compute sta and maximum likelihood optionally.

        """

        super().__init__(X, y, dims, compute_mle, **kwargs) 
        
        self.df = df # number basis / degree of freedom
        self.smooth = smooth # type of basis

        S = np.array(build_spline_matrix(self.dims, df, smooth)) # for w
        
        if self.n_c > 1:
            XS = np.dstack([self.X[:, :, i] @ S for i in range(self.n_c)]).reshape(self.n_samples, -1)
        else:
            XS = self.X @ S

        self.S = S # spline matrix
        self.XS = XS

        self.n_b = S.shape[1] # num:ber of spline coefficients

        # compute spline-based maximum likelihood
        self.b_spl = np.linalg.lstsq(XS.T @ XS, XS.T @ y, rcond=None)[0]

        if self.n_c > 1: 
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
            Number of basis.

        smooth : str
            Type of basis.

        shift : int
            Should be 1 or larger. 

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


    def fit(self, p0=None, extra=None, initialize='random',
            num_epochs=1, num_iters=3000, metric=None, alpha=1, beta=0.05, 
            fit_linear_filter=True, fit_intercept=True, fit_R=True,
            fit_history_filter=False, fit_nonlinearity=False, 
            step_size=1e-2, tolerance=10, verbose=100, random_seed=2046):

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
            Elastic net parameter, overall weight of regulization for receptive field.

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

        self.alpha = alpha
        self.beta = beta # elastic net parameter - global penalty weight for linear filter
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
                    b0 = 0.01 * random.normal(key, shape=(self.n_b * self.n_c, )).flatten()
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

        if 'nl_params' not in dict_keys:
            if hasattr(self, 'nl_params'):
                p0.update({'nl_params': self.nl_params})
            else:
                p0.update({'nl_params': None})

        if extra is not None:

            if self.n_c > 1:
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
        self.p_opt = self.optimize_params(p0, extra, num_epochs, num_iters, metric, step_size, tolerance, verbose)
        self.R = self.p_opt['R'] if fit_R else np.array([1.])

        if fit_linear_filter:
            self.b_opt = self.p_opt['b'] # optimized RF basis coefficients
            if self.n_c > 1:
                self.w_opt = self.S @ self.b_opt.reshape(self.n_b, self.n_c)  
            else:
                self.w_opt = self.S @ self.b_opt # optimized RF
        
        if fit_history_filter:
            self.bh_opt = self.p_opt['bh']
            self.h_opt = self.Sh @ self.bh_opt

        if fit_nonlinearity:
            self.nl_params_opt = self.p_opt['nl_params']
       
        if fit_intercept:
            self.intercept = self.p_opt['intercept']

    def predict(self, X, y=None, p=None):

        """

        Parameters
        ==========

        X : array_like, shape (n_samples, n_features)
            Stimulus design matrix.

        y : None or array_like, shape (n_samples, )
            Recorded response. Needed when post-spike filter is fitted.

        p : None or dict
            Model parameters. Only needed if model performance is monitored
            during training.

        """
        
        if self.n_c > 1:
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
