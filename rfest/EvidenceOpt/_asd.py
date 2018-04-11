import autograd.numpy as np
from ._base import *
from .._utils import *

__all__ = ['ASD']

class ASD(EmpiricalBayes):
    
    def __init__(self, X, Y, rf_dims):
        super().__init__(X, Y, rf_dims)
        self.D = self.squared_distance(self.dims_sRF)
        
    def squared_distance(self, rf_dims):

        if len(rf_dims) == 1:

            n = self.dims_sRF[0]

            adjM = []
            idx = np.arange(n)
            for x in range(n):
                adjM.append(np.square(x - idx))        
            adjM = np.array(adjM)

        elif len(rf_dims) > 1:

            n, m = self.dims_sRF

            xes = np.zeros([n, m])
            yes = np.zeros([n, m])
            coords = []
            counter = 0
            for x in range(n):
                for y in range(m):
                    coords.append([x, y])
            coo = np.vstack(coords)      
            adjM = np.vstack([np.sum((coo[i] - coo) ** 2, 1) for i in range(coo.shape[0])])

        return adjM 
    
    def initialize_params(self):
        
        sigma = np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples
        rho = -2.3
        delta = 1.

        return [sigma, rho, delta]
    
    
    def update_C_prior(self, params):
        
        rho = params[1]
        delta = params[2]

        C_prior = np.exp(-rho - 0.5 * self.D/delta**2)
        C_prior_inv = np.linalg.inv(C_prior)

        return C_prior, C_prior_inv
