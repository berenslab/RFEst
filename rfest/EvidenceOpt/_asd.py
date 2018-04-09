import autograd.numpy as np
from ._base import *
from .._utils import *

__all__ = ['ASD']

class ASD(EmpiricalBayes):
    
    def __init__(self, X, Y, rf_dims):
        super().__init__(X, Y, rf_dims)
        self.D = self.squared_distance(rf_dims[:2])
        
    def squared_distance(self, rf_dims):

        if len(rf_dims) == 1:

            n = rf_dims[0]
            adjM = []
            idx = np.arange(n)
            for x in range(n):
                adjM.append(np.square(x - idx))        
            adjM = np.array(adjM)

        elif len(rf_dims) == 2:
            n, m = rf_dims  

            xes = np.zeros([n, m])
            yes = np.zeros([n, m])
            coords = []
            counter = 0
            for x in range(n):
                for y in range(m):
                    coords.append([x, y])
            coo = np.vstack(coords)      
            adjM = np.vstack([np.sum((coo[i] - coo) ** 2, 1) for i in range(coo.shape[0])])

        elif len(rf_dims) == 3:
            # still need more considerations for what is the right distance matrix for 3D,
            # the current implementation assumes every voxel correlated to all other voxels,
            # which is obviously wrong, because  

            n, m, p = rf_dims  

            xes = np.zeros([n, m, p])
            yes = np.zeros([n, m, p])
            zes = np.zeros([n, m, p])
            coords = []
            counter = 0
            for x in range(n):
                for y in range(m):
                    for z in range(p):
                        coords.append([x, y, z])
            coo = np.vstack(coords)      
            adjM = np.vstack([np.sum((coo[i] - coo) ** 2, 1) for i in range(coo.shape[0])])

        return adjM 
    
    def initialize_params(self):
        
        sigma = np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples
        theta = 10.
        deltas = np.ones(len(self.rf_dims)) * 1.

        return [sigma, theta, deltas]
    
    
    def update_C_prior(self, params):
        
        theta = params[1]
        rho = -np.log(theta)
        deltas = params[2:]        
        
        if len(deltas[0]) == 1:
            delta = deltas[0][0]
            C_prior = np.exp(-rho - 0.5 * self.D/delta**2)
        elif len(deltas[0]) == 2:
            deltax, deltay = deltas[0]
            C_prior = np.exp(-rho - 0.5 * (self.D/deltax**2 + self.D/deltay**2))
        elif len(deltas[0]) == 3:
            deltax, deltay, deltat = deltas[0]
            C_prior = np.exp(-rho - 0.5 * (self.D/deltax**2 + self.D/deltay**2 + self.D/deltat**2))
    
        C_prior_inv = np.linalg.inv(C_prior)

        return C_prior, C_prior_inv
