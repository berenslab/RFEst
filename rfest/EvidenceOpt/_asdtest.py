import autograd.numpy as np
from rfest.EvidenceOpt._base import *
from rfest._utils import *

class ASDtest(EmpiricalBayes):
    
    def __init__(self, X, Y, rf_dims, interpolate=1, mode='reduced'):
        super().__init__(X, Y, rf_dims, interpolate, mode)
        self.mode = mode

    def squared_distance(self, rf_dims, mode='reduced'):

        n = rf_dims

        adjMs = []
        idx = np.arange(n)
        for x in range(n):
            adjMs.append(np.square(x - idx))        
        adjMs = np.array(adjMs)

        return adjMs

    
    def initialize_params(self):
        
        sigma = np.sqrt(np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples)
        rho = -2.3
        
        if len(self.rf_dims) > 2 and self.mode=='full':
            return [sigma, rho, 1. ,1., 1.]
        else:
            delta = 1.
            return [sigma, rho, delta]

    def update_C_prior(self, params):
        
        rho = params[1]
        
        deltax = params[2]
        deltay = params[3]
        deltat = params[4]
        # print(deltax)
        # C_prior = np.exp(-rho - 0.5 * squared_distance(self.rf_dims[0])/deltax**2 - 0.5 * self.Dy/deltay**2 - 0.5 * self.Dz/deltaz**2)
        # C_prior = np.exp(-rho - 0.5 * self.Dx/deltax**2 - 0.5 * self.Dy/deltay**2)
        # C_prior_inv = np.linalg.inv(C_prior)
        C_time = np.exp(-rho - 0.5 * self.squared_distance(self.rf_dims[2])/deltat**2)
        C_time_inv = np.linalg.inv(C_time)

        C_space_x = np.exp(-rho - 0.5 * self.squared_distance(self.rf_dims[0])/deltax**2)
        C_space_x_inv = np.linalg.inv(C_space_x)

        C_space_y = np.exp(-rho - 0.5 * self.squared_distance(self.rf_dims[1])/deltay**2)
        C_space_y_inv = np.linalg.inv(C_space_y)

        C_space = np.kron(C_space_x, C_space_y)
        C_space_inv = np.kron(C_space_x_inv, C_space_y_inv)

        C_prior = np.kron(C_time, C_space)
        C_prior_inv = np.kron(C_time_inv, C_space_inv)

        return C_prior, C_prior_inv

