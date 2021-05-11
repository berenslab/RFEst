import jax.numpy as np
import jax.random as random
from jax import value_and_grad
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

from scipy.optimize import minimize

__all__ = ['GLM']

class GLM:
    
    def __init__(self, distr='poisson', output_nonlinearity='relu'):
        
        # initilize variables
        self.X = {} # design matrix
        self.S = {} # spline matrix
        self.XS = {} # dot product of X and S
        self.y = {} # response
        
        self.b = {} # spline weights
        self.b_mle = {} # mle spline weights
        
        self.w = {} # filter
        self.w_mle = {} # mle filter 

        self.df = {}
        self.dims = {}
        self.filter_names = {}
        self.n_features = {}
        
        self.distr = distr # noise distribution, either gaussian or poisson
        
        self.shift = {}
        self.filter_nonlinearity = {}
        self.output_nonlinearity = output_nonlinearity
        
    def fnl(self, x, kind, params=None):

        '''
        Choose a fixed nonlinear function or fit a flexible one ('nonparametric').
        '''

        if  kind == 'softplus':
            def softplus(x):
                return np.log(1 + np.exp(x))
            return softplus(x)

        elif kind == 'exponential':
            return np.exp(x)

        elif kind == 'softmax':
            def softmax(x):
                z = np.exp(x)
                return z / z.sum()
            return softmax(x)

        elif kind == 'sigmoid':
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            return sigmoid(x)

        elif kind == 'tanh':
            return np.tanh(x)

        elif kind == 'relu':
            def relu(x):
                return np.where(x > 0., x, 0.)
            return relu(x)

        elif kind == 'leaky_relu':
            def leaky_relu(x):
                return np.where(x > 0., x, x * 0.01)
            return leaky_relu(x)

        elif kind == 'none':
            return x

        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def add_design_matrix(self, X, dims, df=None, smooth=None, filter_nonlinearity='none',
                          kind='train', name='stimulus', shift=0):
        
        # check X shape
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if kind not in self.X:
            self.X.update({kind: {}})
            self.filter_names.update({kind: []})
        
        self.filter_nonlinearity[name] = filter_nonlinearity 
        self.shift[name] = shift
        self.dims[name] = dims if type(dims) is not int else [dims, ]

        self.filter_names[kind].append(name)
        self.X[kind][name] = build_design_matrix(X, self.dims[name][0], shift=shift)
        
        if smooth is None:

            if name in self.S:
                
                if kind not in self.XS:
                    self.XS.update({kind: {}})
                
                S = self.S[name]
                self.XS[kind][name] = self.X[kind][name] @ S 
        else:

            if kind not in self.XS:
                self.XS.update({kind: {}})
            
            self.df[name] = df if type(df) is not int else [df, ]
            S = build_spline_matrix(self.dims[name], self.df[name], smooth)
            self.S[name] = S
            self.XS[kind][name] = self.X[kind][name] @ S
                
    def initialize(self, num_subunits=1, dt=0.033, kind='random', random_seed=2046):
        
        self.key = random.PRNGKey(random_seed)
        self.num_subunits = num_subunits
        self.dt = dt
        self.init_kind = kind
        
        if kind =='random':
            
            for name in self.filter_names['train']:
                if 'train' in self.XS and name in self.XS['train']:
                    if 'stimulus' in name:
                        self.b[name] = random.normal(self.key, shape=(self.XS['train'][name].shape[1], num_subunits))
                        self.n_features[name] = len(self.b[name])
                    else:
                        self.b[name] = random.normal(self.key, shape=(self.XS['train'][name].shape[1], 1))
                        self.n_features[name] = len(self.b[name])                        
                else:
                    if 'stimulus' in name:
                        self.w[name] = random.normal(self.key, shape=(self.X['train'][name].shape[1], num_subunits))
                        self.n_features[name] = len(self.w[name])
                    else:
                        self.w[name] = random.normal(self.key, shape=(self.X['train'][name].shape[1], 1))
                        self.n_features[name] = len(self.w[name])                        
        else:
            raise ValueError(f'`{kind}` is not supported.')
                    
    def compute_mle(self, y):
        
        X = np.hstack([self.XS['train'][name] if name in self.XS else self.X['train'][name] for name in self.filter_names])   
        
        XtX = X.T @ X
        Xty = X.T @ y
        mle = np.linalg.solve(XtX, Xty).T
        
        l = np.cumsum(np.hstack([0, [self.n_features[name] for name in self.n_features]]))
        idx = [np.array((l[i], l[i+1])) for i in range(len(l)-1)]
        self.idx = idx
        for i, name in enumerate(self.filter_names['train']):
            if name in self.XS:
                self.b_mle[name] = mle[idx[i][0]:idx[i][1]]
                self.w_mle[name] = self.S[name] @ self.b_mle[name]
            else:
                self.w_mle[name] = mle[idx[i][0]:idx[i][1]]          
        
    def forwardpass(self, p, kind):
        
        intercept = p['intercept'] if 'intercept' in p else self.intercept
        output = np.array([self.fnl(np.sum(self.XS[kind][name] @ p[name], axis=1), 
                                    kind=self.filter_nonlinearity[name]) 
                            if 
                                'train' in self.XS and name in self.XS[kind] 
                            else 
                                np.sum(self.X[kind][name] @ p[name], axis=1) for name in self.filter_names[kind]]).sum(0)
        
        return self.fnl(output + intercept, kind=self.output_nonlinearity)
                
    def cost(self, p, kind='train', precomputed=None):
        
        distr = self.distr
        dt = self.dt
        y = self.y[kind]
        r = self.forwardpass(p, kind) if precomputed is None else precomputed
        
        if distr == 'gaussian':
            loss = np.nanmean((y - r)**2)
        
        elif distr == 'poisson':
            
            r = np.maximum(r, 1e-20) # remove zero to avoid nan in log.
            term0 = - np.log(r) @ y # spike term from poisson log-likelihood
            term1 = np.sum(r) # non-spike term            
            loss = term0 + term1
            
        if self.beta and kind =='train':

            w = np.array([p[name] for name in self.filter_names['train'] if 'stimulus' in name]).flatten()

            l1 = np.linalg.norm(w, 1)
            l2 = np.linalg.norm(w, 2)
            loss += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)
        
        return loss
    
    def optimize(self, p0, num_iters, metric, step_size, tolerance, verbose):
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            l, g = value_and_grad(self.cost)(p)
            return l, opt_update(i, g, opt_state)
        
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)
        
        # preallocation
        cost_train = [0] * num_iters 
        cost_dev = [0] * num_iters
        metric_train = [0] * num_iters
        metric_dev = [0] * num_iters    
        params_list = [0] * num_iters
        
        if verbose:
            if 'dev' not in self.y:
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
        for i in range(num_iters):
            cost_train[i], opt_state = step(i, opt_state)
            params_list[i] = get_params(opt_state)
            
            y_pred_train = self.forwardpass(p=params_list[i], kind='train')
            metric_train[i] = self._score(self.y['train'], y_pred_train, metric)
                     
            if 'dev' in self.y:
                y_pred_dev = self.forwardpass(p=params_list[i], kind='dev')
                cost_dev[i] = self.cost(p=params_list[i], kind='dev', precomputed=y_pred_dev)
                metric_dev[i] = self._score(self.y['dev'], y_pred_dev, metric)      
                
            time_elapsed = time.time() - time_start
            if verbose:
                if i % int(verbose) == 0:
                    if 'dev' not in self.y:
                        if metric is None:
                            print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}'.format(i, time_elapsed, cost_train[i]))
                        else:
                            print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}'.format(i, time_elapsed, cost_train[i], metric_train[i])) 

                    else:
                        if metric is None:
                            print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}'.format(i, time_elapsed, cost_train[i], cost_dev[i]))
                        else:
                            print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}\t{3:>10.3f}\t{4:>10.3f}\t{5:>10.3f}'.format(i, time_elapsed, 
                                                                                                              cost_train[i], cost_dev[i], 
                                                                                                              metric_train[i], metric_dev[i]))
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
                
        self.cost_train = np.hstack(cost_train[:i+1])
        self.cost_dev = np.hstack(cost_dev[:i+1])
        self.metric_train = np.hstack(metric_train[:i+1])
        self.metric_dev = np.hstack(metric_dev[:i+1])
        
        params = params_list[i]
        
        return params
            
    def _score(self, y, y_pred, metric):

        if metric == 'r2':
            return r2(y, y_pred)
        elif metric == 'mse':
            return mse(y, y_pred)
        elif metric == 'corrcoef':
            return corrcoef(y, y_pred)
        else:
            print(f'Metric `{metric}` is not supported.')
            
    def fit(self, y, metric='corrcoef', num_iters=3, alpha=1, beta=0.01, intercept=1., gain=1.,
            step_size=1e-3, tolerance=10, verbose=True, init='random', var_names=None):
        
        self.alpha = alpha
        self.beta = beta
        
        if type(y) is dict:
            self.y['train'] = y['train']
            self.y['dev'] = y['dev']
        else:
            self.y['train'] = y
        
        p0 = {} # parameters to be optimized
        
        if var_names is None:
            var_names = self.filter_names['train'].copy()
            var_names.append('intercept')
            var_names.append('gain')
        
        for name in var_names:
            if name in self.filter_names['train']:
                if 'train' in self.XS and name in self.XS['train']:
                    p0.update({name: self.b[name]})
                else:
                    p0.update({name: self.w[name]})
                
        if 'intercept' in var_names:
            p0.update({'intercept': intercept})
        else:
            self.intercept = 1
        
        # if 'gain' in var_names:
        #     p0.update({'gain': gain})
        # else:
        #     self.gain = 1

        self.p0 = p0
        self.p_opt = self.optimize(p0, num_iters, metric, step_size, tolerance, verbose)
        
        self.b_opt = {}
        self.w_opt = {}
        for name in self.filter_names['train']:
            if 'train' in self.XS and name in self.XS['train']:
                self.b_opt[name] = self.p_opt[name]
                self.w_opt[name] = self.S[name] @ self.b_opt[name]
            else:
                self.w_opt[name] = self.p_opt[name]
        
        if 'intercept' in self.p_opt:
            self.interecept = self.p_opt['intercept']
        # if 'gain' in self.p_opt:
        #     self.gain = self.p_opt['gain'] 

    def predict(self, X):
        
        '''
        Parameters
        ----------
        
        X: np.array or dict
            Stimulus. 
        '''
        
        ws = self.w_opt

        self.X['test'] = {}
        self.XS['test'] = {}
        self.filter_names['test'] = []
        
        if type(X) is dict:
            for name in X:
                self.add_design_matrix(X[name], dims=self.dims[name], shift=self.shift[name], name=name, kind='test')
        else:
            self.add_design_matrix(X, dims=self.dims['stimulus'], shift=self.shift['stimulus'], name='stimulus', kind='test')
        
        ypred = self.forwardpass(self.p_opt, kind='test')
         
        return ypred