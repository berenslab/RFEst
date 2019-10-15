import autograd.numpy as np
from ._base import *
from .._utils import *

__all__ = ['ARD']

class ARD(EmpiricalBayes):
    
    def update_C_prior(self, params):

        theta = params[1]
        theta = np.maximum(1e-7, theta)        
        C_prior = np.diag(theta)
        C_prior += 1e-07 * np.eye(self.n_features)
        C_prior_inv = np.linalg.inv(C_prior)
    
        return C_prior, C_prior_inv    
   