import autograd.numpy as np
from ._base import *
from .._utils import *

__all__ = ['Ridge']

class Ridge:

    """
    Fixed Point Method
    """

    def update_theta(self, params, C_post, m_post):
        theta = params[1]
        return self.n_features - theta * np.trace(C_post) / np.sum(m_post ** 2)
    
    def update_sigma(self, params, C_post, m_post):
        
        sigma = params[0]
        theta = params[1]
        
        numerator   = np.mean(np.sum((self.Y - np.dot(self.X, m_post))**2, 0))
        denominator = self.n_samples - np.sum(1 - theta * np.diag(C_post))
        
        return numerator / denominator

    def update_params(self, params, num_iters, callback):
        
        for iteration in range(num_iters):
            
            if callback is not None:
                callback(params, iteration)
            
            (C_prior,
             C_prior_inv) = self.update_C_prior(params)

            (C_post, C_post_inv,
             m_post) = self.update_posterior(params, C_prior, C_prior_inv)
        
            sigma = self.update_sigma(params, C_post, m_post)
            theta = self.update_theta(params, C_post, m_post)
        
            params = [sigma, theta]
            
        return params