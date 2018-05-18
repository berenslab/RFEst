import scipy.fftpack

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc import flatten
from autograd.misc.optimizers import adam
from sklearn.utils.extmath import randomized_svd


from .._utils import *

__all__ = ['EmpiricalBayes']

class EmpiricalBayes:

    def __init__(self, X, Y, rf_dims):
        
        self.X = X
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.rf_dims = rf_dims
        
        if len(rf_dims) == 3:
            self.dims_tRF = rf_dims[2]
            self.dims_sRF = rf_dims[:2]
            self.Y = get_rdm(Y, self.dims_tRF)
        else:
            self.dims_sRF = rf_dims
            self.dims_tRF = None
            self.Y = Y
        
        self.XtX = self.X.T @ self.X
        self.XtY = self.X.T @ self.Y
        self.YtY = self.Y.T @ self.Y

        self.w_mle = np.linalg.solve(self.XtX, self.XtY)
        self.sRF_mle, self.tRF_mle = self.SVD(self.w_mle)
    
    def SVD(self, w):

        if len(self.rf_dims) == 3:
            U, S, Vt = randomized_svd(w.reshape(self.n_features,self.dims_tRF), 3)
            sRF = U[:, 0].reshape(*self.dims_sRF)
            tRF = Vt[0]
        else:
            sRF = w
            tRF = None

        return [sRF, tRF]

    def initialize_params(self):
        pass

    def update_C_prior(self, params):
        pass
    
    def update_posterior(self, params, C_prior, C_prior_inv):

        sigma = params[0]

        C_post_inv = self.XtX / sigma**2 + C_prior_inv
        C_post = np.linalg.inv(C_post_inv)
        
        m_post = C_post @ self.XtY / (sigma**2)
        
        return C_post, C_post_inv, m_post
        
    def log_evidence(self, params):
        
        sigma = params[0]
        
        (C_prior, C_prior_inv) = self.update_C_prior(params)
        
        (C_post, C_post_inv, m_post) = self.update_posterior(params, C_prior, C_prior_inv)
        
        t0 = np.log(np.abs(2 * np.pi * sigma**2)) * self.n_samples
        t1 = np.linalg.slogdet(C_prior @ C_post_inv)[1]
        t2 = m_post.T @ C_post @ m_post
        if len(np.shape(t2)) != 0:
            t2 = - np.mean(np.diag(t2))
        else:
            t2 = - t2
        t3 = self.YtY / sigma**2
        if len(np.shape(t3)) != 0:
            t3 = np.mean(np.diag(t3))
        else:
            t3 = t3
        
        return -0.5 * (t0 + t1 + t2 + t3)
    
    def objective(self, params, t):
        return -self.log_evidence(params)

    def optimize_params(self, initial_params, step_size, num_iters, bounds, callback, ):
        params = adam(grad(self.objective),
                            x0 = initial_params,
                            step_size = step_size,
                            num_iters = num_iters,
                            callback = callback)
        return params

    
    def fit(self, initial_params=None, step_size=0.001, num_iters=1, bounds=None, callback=None):

        self.step_size = step_size
        self.num_iters = num_iters
        
        if initial_params is None:
            initial_params = self.initialize_params()
        
        self.optimized_params = self.optimize_params(initial_params, step_size, num_iters, bounds, callback)

        (optimized_C_prior, 
         optimized_C_prior_inv) = self.update_C_prior(self.optimized_params)
        
        (optimized_C_post, 
         optimized_C_post_inv, 
         optimized_m_post) = self.update_posterior(self.optimized_params,
                                                   optimized_C_prior,
                                                   optimized_C_prior_inv)
        
        self.optimized_C_prior = optimized_C_prior
        self.optimized_C_post = optimized_C_post
        self.optimized_m_post = optimized_m_post
        self.w_opt = optimized_m_post
        self.sRF_opt, self.tRF_opt = self.SVD(self.w_opt)   
