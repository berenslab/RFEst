import autograd.numpy as np
from ._base import *
from .._utils import *

__all__ = ['Ridge', 'RidgeFixedPoint']

class Ridge(EmpiricalBayes):
    
    def initialize_params(self):
        
        sigma = np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples
        theta = 10.
        
        return [sigma,theta]
    
    def update_C_prior(self, params):
        
        theta = params[1]
        C_prior = np.eye(self.n_features) * theta
        C_prior += 1e-07 * np.eye(self.n_features)
        C_prior_inv = np.linalg.inv(C_prior)
    
        return C_prior, C_prior_inv

class RidgeFixedPoint(Ridge):

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