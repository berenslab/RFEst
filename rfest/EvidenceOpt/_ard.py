import autograd.numpy as np
from ._base import *
from .._utils import *

__all__ = ['ARD']

class ARD(EmpiricalBayes):
    
    def initialize_params(self):
        
        sigma = np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples
        theta = np.random.rand(self.n_features)
        
        return [sigma, theta]
    
    def update_C_prior(self, params):

        theta = params[1]
        theta = np.maximum(1e-7, theta)        
        C_prior = np.diag(theta)
        C_prior += 1e-07 * np.eye(self.n_features)
        C_prior_inv = np.linalg.inv(C_prior)
    
        return C_prior, C_prior_inv    
        
class ARDFixedPoint(ARD):

    def update_theta(self, params, C_post, m_post):
        pass

    def update_sigma(self, params, C_post, m_post):
        pass

    def update_params(self, params, num_iters, callback):
        pass
   