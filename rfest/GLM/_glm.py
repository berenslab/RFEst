from jaxlib.xla_client import PaddingType
import numpy as onp
import jax.numpy as np
import jax.random as random
from jax import value_and_grad
from jax import jit, jacfwd, jacrev
from jax.experimental import optimizers
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import time
import scipy.linalg

from ..utils import build_design_matrix
from ..splines import build_spline_matrix
from ..metrics import r2, r2adj, mse, corrcoef, gcv

__all__ = ['GLM']

class GLM:
    
    def __init__(self, distr='poisson', output_nonlinearity='none'):

        '''
        Initialize the GLM class with empty variables.

        Parameters
        ----------

        distr: str
            Noise distribution. Either `gaussian` or `poisson`.

        output_nonlinearity: str
            Nonlinearity for the output layer. 
        ''' 

        # initilize variables

        ## Data
        self.X = {} # design matrix
        self.S = {} # spline matrix
        self.P = {} # penalty matrix
        self.XS = {} # dot product of X and S
        self.XtX = {} # input covariance
        self.Xty = {} # cross-covariance
        self.y = {} # response
        self.y_pred = {} # predicted response
        self.y_pred_upper = {} # predicted response upper limit
        self.y_pred_lower = {} # predicted response lower limit 
        
        ## Model parameters
        self.p = {} # all model paremeters
        self.b = {} # spline weights        
        self.b_se = {} # spline weights standard error
        self.w = {} # filter weights
        self.w_se = {} # filter weights standard error
        self.V = {} # weights covariance
        self.intercept = {} # intercept


        ## Model hypterparameters
        self.df = {} # number of bases for each filter
        self.edf = {} # effective degree of freedom given lam 
        self.dims = {} # filter shapes
        self.n_features = {} # number of features for each filter
        self.filter_names = [] # filter names
        
        self.shift = {} # time shift of the design matrix
        self.filter_nonlinearity = {}
        self.output_nonlinearity = output_nonlinearity
        
        # Noise distribution, either gaussian or poisson
        self.distr = distr 
        
        # others
        self.mle_computed = False
        self.lam = {} # smoothness regularization weight
        self.scores = {} # prediction error metric scores
        self.r2pseudo = {}
        self.corrcoef = {}
        
    def fnl(self, x, kind, params=None):

        '''
        Choose a fixed nonlinear function or fit a flexible one ('nonparametric').

        Parameters
        ----------

        x: np.array, (n_samples, )
            Sum of filter outputs.

        kind: str
            Choice of nonlinearity.
            
        params: None or np.array.
            For flexible nonlinearity. To be implemented.

        Return 
        ------
            Transformed sum of filter outputs.
        '''

        if  kind == 'softplus':
            def softplus(x):
                return np.log(1 + np.exp(x))
            return softplus(x) + 1e-7

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
                return np.where(x > 0., x, 1e-7)
            return relu(x)

        elif kind == 'leaky_relu':
            def leaky_relu(x):
                return np.where(x > 0., x, x * 0.01)
            return leaky_relu(x)

        elif kind == 'none':
            return x

        else:
            raise ValueError(f'Input filter nonlinearity `{nl}` is not supported.')

    def add_design_matrix(self, X, dims=None, df=None, smooth=None, 
                          num_subunits=1  , lam=0., filter_nonlinearity='none',
                          kind='train', name='stimulus', shift=0, burn_in=None):

        '''
        Add input desgin matrix to the model.

        Parameters
        ----------

        X: np.array, shape=(n_samples, ) or (n_samples, n_pixels)
            Original input. 
        
        dims: int, or list / np.array, shape=dim_t, or (dim_t, dim_x, dim_y)
            Filter shape.

        df: None, int, or list / np.array
            Number of spline bases. Should be the same shape as dims.
        
        smooth: None, or str
            Type of spline bases. If None, no basis is used.

        filter_nonlinearity: str
            Nonlinearity for the stimulus filter.

        kind: str
            Datset type, should be one of `train` (training set), 
            `dev` (validation set) or `test` (testing set).

        name: str
            Name of the corresponding filter. 
            A receptive field (stimulus) filter should have `stimulus` in the name. 
            A response-history filter should have `history` in the name.

        shift: int
            Time offset for the design matrix, positive number will shift the design 
            matrix to the past, negative number will shift it to the future.

        burn_in: int or None
            Number of samples / frames to be ignore for prediction.
            (Because the first few frames in the design matrix are full of zero, which 
            tend to predict poorly.)

        '''
        
        # check X shape
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if kind not in self.X:
            self.X.update({kind: {}})

        if kind == 'train':
            self.filter_nonlinearity[name] = filter_nonlinearity 
            self.filter_names.append(name)

            dims = dims if type(dims) is not int else [dims, ]  
            self.dims[name] = dims 
            self.shift[name] = shift
        else:
            dims = self.dims[name]
            shift = self.shift[name] 
            
        if not hasattr(self, 'burn_in'): # if exists, ignore
            self.burn_in = dims[0]-1 if burn_in is None else burn_in # number of first few frames to ignore 
        
        self.X[kind][name] = build_design_matrix(X, dims[0], shift=shift)[self.burn_in:]
         
        if smooth is None:
            # if train set exists and used spline as basis
            # automatically apply the same basis for dev/test set
            if name in self.S:
                if kind not in self.XS:
                    self.XS.update({kind: {}})
                S = self.S[name]
                self.XS[kind][name] = self.X[kind][name] @ S 

            elif kind == 'test':                                        
                if kind not in self.XS:
                   self.XS.update({kind: {}})
                if hasattr(self, 'num_subunits') and self.num_subunits > 1:
                    S = self.S['stimulus_s0']
                else:
                    S = self.S[name]
                self.XS[kind][name] = self.X[kind][name] @ S

            else:
                if kind == 'train':
                    self.n_features[name] = self.X['train'][name].shape[1]

                self.edf[name] = self.n_features[name]
        
        else: # use spline

            if kind not in self.XS:
                self.XS.update({kind: {}})
            
            self.df[name] = df if type(df) is not int else [df, ]
            self.lam[name] = lam if type(lam) is list else [lam,] * len(self.df[name])
            S, P = build_spline_matrix(dims, self.df[name], smooth, self.lam[name], return_P=True)
            self.S[name] = S
            self.P[name] = P # penalty matrix, which absolved lamda already

            XS = self.X[kind][name] @ S
            edf =(XS.T * (np.linalg.inv(XS.T @ XS + P) @ XS.T)).sum()
            self.edf[name] = edf
            self.XS[kind][name] = XS

            if kind =='train': 
                self.n_features[name] = self.XS['train'][name].shape[1]
    
    def initialize(self, y=None, num_subunits=1, dt=0.033, method='random', random_seed=2046, verbose=0):
    
        self.init_method = method # store meta            
        self.num_subunits = num_subunits

        if method =='random':

            self.b['random'] = {}
            self.w['random'] = {}
            if verbose:
                print('Initializing model paraemters randomly...')

            for i, name in enumerate(self.filter_names):
                key = random.PRNGKey(random_seed + i) # change random seed for each filter
                if name in self.S:
                    self.b['random'][name] = random.normal(key, shape=(self.XS['train'][name].shape[1], 1))
                    self.w['random'][name] = self.S[name] @ self.b['random'][name] 
                else:
                    self.w['random'][name] = random.normal(key, shape=(self.X['train'][name].shape[1], 1))
            self.intercept['random'] = 0.
        
            if verbose:
                print('Finished.')
            
        elif method == 'mle':
            
            if verbose:
                print('Initializing model paraemters with maximum likelihood...')
                
            if not self.mle_computed:
                # self.compute_mle(y, num_subunits, random_seed)
                self.compute_mle(y) 
                
            if verbose:
                print('Finished.')
                
        else:
            raise ValueError(f'`{method}` is not supported.')
        
        # rename and repmat: stimulus filter to subunits filters
        filter_names = self.filter_names.copy()
        if num_subunits != 1:
            filter_names.remove('stimulus')
            filter_names = [f'stimulus_s{i}' for i in range(num_subunits)] + filter_names
            
            for name in filter_names:
                if 'stimulus' in name:
                    self.dims[name] = self.dims['stimulus']
                    self.df[name] = self.dims['stimulus']
                    self.shift[name] = self.shift['stimulus']
                    self.filter_nonlinearity[name] = self.filter_nonlinearity['stimulus']
                    self.w[method][name] = self.w[method]['stimulus']
                    if method in self.w_se: 
                        self.w_se[method][name] = self.w_se[method]['stimulus']
                    self.X['train'][name] = self.X['train']['stimulus']
                    self.edf[name] = self.edf['stimulus']
            
                    if 'dev' in self.X:
                        self.X['dev'][name] = self.X['dev']['stimulus']
                        
                    if 'stimulus' in self.S:
                        self.b[method][name] = self.b[method]['stimulus']
                        if method in self.b_se:
                            self.b_se[method][name] = self.b_se[method]['stimulus']
                        self.XS['train'][name] = self.XS['train']['stimulus']
                        
                        if 'dev' in self.XS:
                            self.XS['dev'][name] = self.XS['dev']['stimulus']
                        
                        self.P[name] = self.P['stimulus']
                        self.S[name] = self.S['stimulus']
                        
            try:
                self.b[method].pop('stimulus')
            except:
                pass
            
            self.w[method].pop('stimulus')
            self.X['train'].pop('stimulus')
            self.X['dev'].pop('stimulus')
            self.XS['train'].pop('stimulus')
            self.XS['dev'].pop('stimulus')
            self.S.pop('stimulus')
            self.P.pop('stimulus')
            self.filter_names = filter_names
            
        p0 = {}
        for i, name in enumerate(self.filter_names):
            if name in self.S:
                b = self.b[method][name]
                key = random.PRNGKey(random_seed + i) 
                noise = 0.1 * random.normal(key, shape=b.shape)
                p0.update({name: b + noise})
            else:
                w = self.w[method][name]
                key = random.PRNGKey(random_seed + i) 
                noise = 0.1 * random.normal(key, shape=w.shape)
                p0.update({name: w})

            p0.update({'intercept': self.intercept[method]}) 
        
        # get random variance
        if method == 'random':
            self.y['train'] = y_train[self.burn_in:]
            self.y_pred['random']['train'] = self.forwardpass(self.p0, kind='train')
            # # get filter confidence interval
            self._get_filter_variance(w_type='random')
            self._get_response_variance(w_type='random', kind='train')
            
            if type(y) is dict and 'dev' in y:
                self.y['dev'] = y_dev[self.burn_in:]
                self.y_pred['random']['dev'] = self.forwardpass(self.p0, kind='dev') 
                self._get_response_variance(w_type='random', kind='dev')

        self.dt = dt
        self.p[method] = p0
        self.p0 = p0

    def compute_mle(self, y):

        '''Compute maximum likelihood estimates.
        
        Parameter
        ---------
        
        y: np.array or dict, (n_samples)
            Response. if dict is 
        '''
        
        if type(y) is dict:
            y_train = y['train']
            if 'dev' in y:
                y_dev = y['dev']
        else:
            y = {'train': y}
            y_train = y['train']

        X = np.hstack([self.XS['train'][name] if name in self.XS['train'] else self.X['train'][name] for name in self.filter_names])   
        X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])

        P = scipy.linalg.block_diag(*([0] + [self.P[name] if name in self.P else np.zeros([self.X['train'][name].shape[1], self.X['train'][name].shape[1]]) 
                                                    for name in self.filter_names]))

        XtX = X.T @ X
        Xty = X.T @ y_train[self.burn_in:]

        mle = np.linalg.solve(XtX + P, Xty)

        self.b['mle'] = {}
        self.w['mle'] = {}
        
        # slicing the mle matrix into each filter
        l = np.cumsum(np.hstack([0, [self.n_features[name] for name in self.n_features]]))
        idx = [np.array((l[i], l[i+1])) for i in range(len(l)-1)]
        self.idx = idx

        for i, name in enumerate(self.filter_names):
            if name in self.S:
                self.b['mle'][name] = mle[idx[i][0]:idx[i][1]][:, np.newaxis]
                self.w['mle'][name] = self.S[name] @ self.b['mle'][name]
            else:
                self.w['mle'][name] = mle[idx[i][0]:idx[i][1]][:, np.newaxis]   
    
        self.intercept['mle'] = mle[0]

        self.p['mle'] = {}
        self.y_pred['mle'] = {}

        for name in self.filter_names:
            if name in self.S:
                self.p['mle'].update({name: self.b['mle'][name]})
            else:
                self.p['mle'].update({name: self.w['mle'][name]})

        self.p['mle']['intercept'] = self.intercept['mle']
        # self.edf_tot = np.array([self.edf[name] if name in self.edf else self.n_features[name] for name in self.filter_names]).sum()
        # self.n_params = XtX.shape[1] # total number of model parameters
 
        self.y['train'] = y_train[self.burn_in:]
        self.y_pred['mle']['train'] = self.forwardpass(self.p['mle'], kind='train')
        # # get filter confidence interval
        self._get_filter_variance(w_type='mle')
        self._get_response_variance(w_type='mle', kind='train')
        
        if type(y) is dict and 'dev' in y:
            self.y['dev'] = y_dev[self.burn_in:]
            self.y_pred['mle']['dev'] = self.forwardpass(self.p['mle'], kind='dev') 
            self._get_response_variance(w_type='mle', kind='dev')


        # # performance stats
        # self.p['null'] = {name: np.zeros_like(self.p['mle'][name]) for name in self.p['mle']}
        
        # for method in ['corrcoef', 'r2', 'mse', 'r2adj', 'r2pseudo', 'gcv']:
        #     self.scores[method] = {}
        #     if method == 'r2pseudo':
        #         self.p['null']['intercept'] = self.y['train'].mean()      
        #         self.scores[method]['train'] = np.asarray(1 - self.cost(self.p['mle'], 'train', penalize=False) / self.cost(self.p['null'], 'train', penalize=False) )
            
        #     elif method == 'gcv':
        #         self.scores[method]['train'] = gcv(self.y['train'], self.y_pred['mle']['train'], self.edf_tot)
        #     else:
        #         self.scores[method]['train'] = self._score(self.y['train'], self.y_pred['mle']['train'], method)
             
        #     if type(y) is dict and 'dev' in y:
        #         if method == 'r2pseudo':
        #             self.p['null']['intercept'] = self.y['train'].mean()      
        #             self.scores[method]['dev'] = np.asarray(1 - self.cost(self.p['mle'], 'dev', penalize=False) / self.cost(self.p['null'], 'dev', penalize=False) )
        #         elif method == 'gcv':
        #             self.scores[method]['dev'] = gcv(self.y['dev'], self.y_pred['mle']['dev'], self.edf_tot)
        #         else:
        #             self.scores[method]['dev'] = self._score(self.y['dev'], self.y_pred['mle']['dev'], method)
                
        # null model with mean firing rate as intercept 
 
        # self.r2pseudo['mle'] = {}
        # self.r2pseudo['mle']['train'] = 1 - self.cost(self.p['mle'], 'train', penalize=False) / self.cost(self.p['null'], 'train', penalize=False)
        # if 'dev' in y:
        #     self.p['null']['intercept'] = self.y['dev'].mean()
        #     self.r2pseudo['mle']['dev'] = 1 - self.cost(self.p['mle'], 'dev', penalize=False) / self.cost(self.p['null'], 'dev', penalize=False)

        self.mle_computed = True

    def forwardpass(self, p, kind):

        '''Forward pass of the model.
        
        Parameters
        ----------

        p: dict
            A dictionary of the model parameters to be optimized.

        kind: str
            Dataset type, can be `train`, `dev` or `test`.
        
        '''
        
        intercept = p['intercept'] if 'intercept' in p else self.intercept
                
        filters_output = []
        for name in self.X[kind]: # 
            if 'train' in self.XS and name in self.XS[kind]:
                input_term = self.XS[kind][name]
            else:
                input_term = self.X[kind][name]

            output = self.fnl(np.sum( input_term @ p[name], axis=1), kind=self.filter_nonlinearity[name]) 
            filters_output.append(output)

        filters_output = np.array(filters_output).sum(0)
        final_output = self.fnl(filters_output + intercept, kind=self.output_nonlinearity)
        
        return final_output
                          
    def cost(self, p, kind='train', precomputed=None, penalize=True):

        '''Cost function.
        
        Parameters
        ----------

        p: dict
            A dictionary of the model parameters to be optimized.

        kind: str
            Dataset type, can be `train`, `dev` or `test`.

        precomputed: None or np.array
            Precomputed forward pass output. For avoding duplicate computation. 
        '''
        
        distr = self.distr
        y = self.y[kind]
        r = self.forwardpass(p, kind) if precomputed is None else precomputed

        # cost functions 
        if distr == 'gaussian':
            loss = 0.5 * np.sum((y - r)**2)
        
        elif distr == 'poisson':
            
            r = np.maximum(r, 1e-20) # remove zero to avoid nan in log.
            term0 = - np.log(r) @ y # spike term from poisson log-likelihood
            term1 = np.sum(r) # non-spike term            
            loss = term0 + term1

        # regularization: elasticnet
        if penalize and (hasattr(self, 'beta') or self.beta != 0) and kind == 'train':

            # regularized all filters parameters
            w = np.hstack([p[name].flatten() for name in self.filter_names])

            l1 = np.linalg.norm(w, 1)
            l2 = np.linalg.norm(w, 2)
            loss += self.beta * ((1 - self.alpha) * l2 + self.alpha * l1)
        
        # regularization: spline wiggliness
        if penalize and kind == 'train':
            
            energy = np.array([np.sum(p[name].T @ self.P[name] @ p[name]) for name in self.P]).sum()
            loss += energy
 
        return np.squeeze(loss)
 
    def optimize(self, p0, num_iters, metric, step_size, tolerance, verbose, return_model):

        '''Workhorse of optimization.

        p0: dict
            A dictionary of the initial model parameters to be optimized.  

        num_iters: int
            Maximum number of iteration.

        metric: str
            Method of model evaluation. Can be
            `mse`, `corrcoeff`, `r2`


        step_size: float or jax scheduler
            Learning rate.
        
        tolerance: int
            Tolerance for early stop. If the training cost doesn't change more than 1e-5
            in the last (tolerance) steps, or the dev cost monotonically increase, stop.

        verbose: int
            Print progress. If verbose=0, no progress will be print.

        return_model: str
            Return the 'best' model on dev set metrics or the 'last' model.
        '''
        
        @jit
        def step(i, opt_state):
            p = get_params(opt_state)
            l, g = value_and_grad(self.cost)(p)
            return l, opt_update(i, g, opt_state)

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)

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
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}'.format('Iters', 'Time (s)', 'Cost (train)', f'{metric} (train)')) 
            else:
                if metric is None:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Cost (dev)'))
                else:
                    print('{0}\t{1:>10}\t{2:>10}\t{3:>10}\t{4:>10}\t{5:>10}'.format('Iters', 'Time (s)', 'Cost (train)', 'Cost (dev)', f'{metric} (train)', f'{metric} (dev)')) 

        
        time_start = time.time()
        for i in range(num_iters):
            cost_train[i], opt_state = step(i, opt_state)
            params_list[i] = get_params(opt_state)
            y_pred_train = self.forwardpass(p=params_list[i], kind='train')
            metric_train[i] = self._score(self.y['train'], y_pred_train, metric)
                     
            if 'dev' in self.y:
                y_pred_dev = self.forwardpass(p=params_list[i], kind='dev')
                cost_dev[i] = self.cost(p=params_list[i], kind='dev', precomputed=y_pred_dev, penalize=False)
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

                if 'dev' in self.y and np.all(cost_dev_slice[1:] - cost_dev_slice[:-1] > 0):
                    # params = params_list[i-tolerance]
                    # metric_dev_opt = metric_dev[i-tolerance]
                    if verbose:
                        print('Stop at {0} steps: cost (dev) has been monotonically increasing for {1} steps.\n'.format(i, tolerance))
                    stop = 'dev_stop'
                    break

                if np.all(cost_train_slice[:-1] - cost_train_slice[1:] < 1e-5):
                    # params = params_list[i]
                    # metric_dev_opt = metric_dev[i]
                    if verbose:
                        print('Stop at {0} steps: cost (train) has been changing less than 1e-5 for {1} steps.\n'.format(i, tolerance))
                    stop = 'train_stop'
                    break
                    
        else:
            # params = params_list[i]
            # metric_dev_opt = metric_dev[i]
            total_time_elapsed = time.time() - time_start
            stop = 'maxiter_stop'
            if verbose:
                print('Stop: reached {0} steps.\n'.format(num_iters))
                
        if return_model == 'best_dev_cost':
            best = np.argmin(np.asarray(cost_dev[:i+1]))     

        elif return_model == 'best_train_cost':
            best = np.argmin(np.asarray(cost_train[:i+1]))  

        elif return_model == 'best_dev_metric':
            if metric in ['mse', 'gcv']:
                best = np.argmin(np.asarray(metric_dev[:i+1]))
            else:
                best = np.argmax(np.asarray(metric_dev[:i+1]))

        elif return_model == 'best_train_metric':
            if metric in ['mse', 'gcv']: 
                best = np.argmin(np.asarray(metric_train[:i+1]))
            else:
                best = np.argmax(np.asarray(metric_train[:i+1])) 

        elif return_model == 'last':
            if stop == 'dev_stop':
                best = i-tolerance
            else:
                best = i
        
        else:
            print('Provided `return_model` is not supported. Fallback to `best_dev_cost`') 
            best = np.argmin(np.asarray(cost_dev[:i+1])) 
        
        params = params_list[best]
        metric_dev_opt = metric_dev[best]                

        self.cost_train = np.hstack(cost_train[:i+1])
        self.cost_dev = np.hstack(cost_dev[:i+1])
        self.metric_train = np.hstack(metric_train[:i+1])
        self.metric_dev = np.hstack(metric_dev[:i+1])
        self.metric_dev_opt = metric_dev_opt
        self.total_time_elapsed = total_time_elapsed 

        self.all_params = params_list # not sure if this will occupy a lot of RAM.
        self.y_pred['opt'].update({'train': y_pred_train, 
                              'dev': y_pred_dev})
                
        return params
                       
    def fit(self, y=None, num_iters=3, alpha=1, beta=0.01, lam=0., metric='corrcoef', step_size=1e-3, 
        tolerance=10, verbose=True, var_names=None, return_model='best_dev_cost'):
        
        '''Fit model.
        
        Parameters
        ----------
        
        y: np.array, (n_samples)
            Response. 
        
        num_iters: int
            Maximum number of iteration.
        
        alpha: float
            Balance weight for L1 and L2 regularization. 
            If alpha=1, only L1 applys. Otherwise, only L2 apply.
        
        beta: float
            Overall weight for L1 and L2 regularization.

        lam: float
            Weight for controlling spline wiggliness.

        droput: float
            Droput probability.

        metric: str
            Method of model evaluation. Can be
            `mse`, `corrcoeff`, `r2`

        step_size: float or jax scheduler
            Learning rate.
        
        tolerance: int
            Tolerance for early stop. If the training cost doesn't change more than 1e-5
            in the last (tolerance) steps, or the dev cost monotonically increase, stop.

        verbose: int
            Print progress. If verbose=0, no progress will be print.

        var_names: list of str
            Name of variables to be fitted.
        
        '''

        
        self.alpha = alpha
        self.beta = beta
        self.metric = metric

        if y is None:
            if not 'train' in self.y:
               raise ValueError(f'No `y` is provided.') 
        else:
            if type(y) is dict:
                self.y['train'] = y['train'][self.burn_in:]
                if 'dev' in y:
                    self.y['dev'] = y['dev'][self.burn_in:]
            else:
                self.y['train'] = y[self.burn_in:]

        self.y_pred['opt'] = {}
            
        self.p['opt'] = self.optimize(self.p0, num_iters, metric, step_size, tolerance, verbose, return_model)
        self._extract_opt_params() 

    def _extract_opt_params(self):
        self.b['opt'] = {}
        self.w['opt'] = {}         
        for name in self.filter_names:
            if name in self.S:
                self.b['opt'][name] = self.p['opt'][name]
                self.w['opt'][name] = self.S[name] @ self.b['opt'][name]
            else:
                self.w['opt'][name] = self.p['opt'][name]
        
        self.intercept['opt'] = self.p['opt']['intercept']
        # get filter confidence interval
        self._get_filter_variance(w_type='opt')
        # get prediction confidence interval
        self._get_response_variance(w_type='opt', kind='train')
        if 'dev' in self.y: 
            self._get_response_variance(w_type='opt', kind='dev')

    def predict(self, X, w_type='opt'):
        
        '''
        Prediction on Test set. 

        Parameters
        ----------
        
        X: np.array or dict
            Stimulus. Only the named filters in the dict will be used for prediction.
            Other filters, even trained, will be ignored if no test set provided.

        dropout: float or None
            Dropout probability

        w_type: str
            either `opt` or `mle`

        Note
        ----
        Use self.forwardpass() for prediction on Training / Dev set.
        '''
        
        p = self.p[w_type]

        self.X['test'] = {}
        self.XS['test'] = {}

        if type(X) is dict:
            for name in X:
                self.add_design_matrix(X[name], dims=self.dims[name], shift=self.shift[name], name=name, kind='test')
        else:
            # if X is np.array, assumed it's the stimulus.
            self.add_design_matrix(X, dims=self.dims['stimulus'], shift=self.shift['stimulus'], name='stimulus', kind='test')
        
        # rename and repmat for test set
        if self.num_subunits != 1:
            for name in self.filter_names:
                if 'stimulus' in name:
                    self.X['test'][name] = self.X['test']['stimulus']
                    self.XS['test'][name] = self.XS['test']['stimulus']
            self.X['test'].pop('stimulus')
            self.XS['test'].pop('stimulus')

        y_pred = self.forwardpass(p, kind='test')

        self.y_pred['test'] = y_pred
        self._get_response_variance(w_type=w_type, kind='test')

        return y_pred

    def _score(self, y, y_pred, metric):

        '''
        Metric score for evaluating model prediction.
        '''

        if metric == 'r2':
            return r2(y, y_pred)
        elif metric == 'r2adj':
            return r2adj(y, y_pred, p=self.edf_tot)

        elif metric == 'mse':
            return mse(y, y_pred)

        elif metric == 'corrcoef':
            return corrcoef(y, y_pred)

        elif metric == 'gcv':
            return gcv(y, y_pred, edf=self.edf_tot)

        else:
            print(f'Metric `{metric}` is not supported.')
 
    def score(self, X_test, y_test, metric='corrcoef', w_type='opt', return_prediction=False):

        '''Metric score for evaluating model prediction.
        
        X_test: np.array or dict
            Stimulus. Only the named filters in the dict will be used for prediction.
            Other filters, even trained, will be ignored if no test set provided. 

        y_test: np.array
            Response.

        metric: str
            Method of model evaluation. Can be
            `mse`, `corrcoeff`, `r2`

        return_prediction: bool
            If true, will also return the predicted response `y_pred`.

        Returns
        -------
        s: float
            Metric score.

        y_pred: np.array.
            The predicted response. Optional. 

        '''

        y_test = y_test[self.burn_in:]

        if type(X_test) is dict:
            y_pred = self.predict(X_test, w_type)
        else:
            y_pred = self.predict({'stimulus': X_test}, w_type)

        s = self._score(y_test, y_pred, metric)

        if return_prediction:
            return s, y_pred 
        else:
            return s 

    def _get_filter_variance(self, w_type='opt'):

        """
        Compute the variance and standard error of the weight of each filters.
        """
        
        P = self.P
        S = self.S
        XS = self.XS['train']
        
        # trA = {name: (XS[name].T * (np.linalg.inv(XS[name].T @ XS[name] + P[name]) @ XS[name].T)).sum(0) for name in self.P}
        edf = self.edf

        if self.distr == 'gaussian':
            
            y = self.y['train']
            y_pred = self.y_pred[w_type]['train']
            rsd = y - y_pred# residuals    
            rss = np.sum(rsd ** 2) # residul sum of squares
            rss_var = {name: rss / (len(y) - edf[name]) for name in self.filter_names}

            V = {}
            b_se = {}
            w_se = {}
            for name in self.filter_names:
                if name in XS:
                    V[name] = np.linalg.inv(XS[name].T @ XS[name] + P[name]) * rss_var[name]
                    b_se[name] = np.sqrt(np.diag(V[name]))
                    w_se[name] = S[name] @ b_se[name]
                else:
                    V[name] = np.linalg.inv(X[name].T @ X[name]) * rss_var[name]
                    w_se[name] = np.sqrt(np.diag(V[name])) 

        else:
            
            b = {}
            w = {}
            u = {}
            U = {}
            V = {}
            w_se = {}
            b_se = {}
            for name in self.filter_names:
                if name in XS:
                    b[name] = self.b[w_type][name]
                    u[name] = self.fnl(XS[name] @ b[name], self.filter_nonlinearity[name])
                    U[name] = 1 / self.fnl(u[name], self.output_nonlinearity).flatten()
                    V[name] = np.linalg.inv(XS[name].T * U[name] @ XS[name] + P[name])
                    b_se[name] = np.sqrt(np.diag(V[name]))
                    w_se[name] = S[name] @ b_se[name]
                else:
                    w[name] = self.w[w_type][name]
                    u[name] = self.fnl(X[name] @ w[name], self.filter_nonlinearity[name])
                    U[name] = 1 / self.fnl(u[name], self.output_nonlinearity).flatten()
                    V[name] = np.linalg.inv(X[name].T * U[name] @ X[name])
                    w_se[name] = np.sqrt(np.diag(V[name])) 
        
        self.V[w_type] = V
        self.b_se[w_type] = b_se
        self.w_se[w_type] = w_se


    def _get_response_variance(self, w_type='opt', kind='train'):

        """
        Compute the variance and standard error of the predicted response.
        """ 
        
        P = self.P
        S = self.S
        X = self.X[kind]
        XS = self.XS[kind]
        w = self.w[w_type]
        V = self.V[w_type]

        y_se = {}
        y_pred_filters = {}
        y_pred_filters_upper = {}
        y_pred_filters_lower = {}
        for name in X:
            if name in XS:
                y_se[name] = np.sqrt(np.sum(XS[name] @ V[name] * XS[name], 1))
            else:
                y_se[name] = np.sqrt(np.sum(self.X[kind][name] @ V[name] * self.X[kind][name], 1))   
            
            y_pred_filters[name] = self.fnl((X[name] @ w[name]).flatten(), kind=self.filter_nonlinearity[name])
            y_pred_filters_upper[name] = self.fnl((X[name] @ w[name]).flatten() + 2 * y_se[name], kind=self.filter_nonlinearity[name])
            y_pred_filters_lower[name] = self.fnl((X[name] @ w[name]).flatten() - 2 * y_se[name], kind=self.filter_nonlinearity[name])

        y_pred = self.fnl(np.array([y_pred_filters[name] for name in X]).sum(0) + self.intercept[w_type], kind=self.output_nonlinearity)
        y_pred_upper = self.fnl(np.array([y_pred_filters_upper[name] for name in X]).sum(0) + self.intercept[w_type], kind=self.output_nonlinearity)
        y_pred_lower = self.fnl(np.array([y_pred_filters_lower[name] for name in X]).sum(0) + self.intercept[w_type], kind=self.output_nonlinearity)
        
        if not w_type in self.y_pred_lower:
            self.y_pred_lower[w_type] = {}
            self.y_pred_upper[w_type] = {}
        
        self.y_pred[w_type][kind] = y_pred
        self.y_pred_upper[w_type][kind] = y_pred_upper
        self.y_pred_lower[w_type][kind] = y_pred_lower


        
# def elementwise_add(A):
    
#     if len(A) == 1:
#         return A[0]
#     elif len(A) == 2:
#         return np.add(*A)
#     elif len(A) == 3:
#         return np.add(*[np.add(*A[:2]), A[-1]])