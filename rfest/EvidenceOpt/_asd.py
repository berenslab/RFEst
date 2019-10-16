import jax.numpy as np
from ._base import *
from .._utils import *

__all__ = ['ASD']

class ASD(EmpiricalBayes):
    
    def __init__(self, X, Y, dims):
        super().__init__(X, Y, dims)

    def _make_1D_covariance(self, delta, ncoeff):

        incoeffices = np.arange(ncoeff)
        square_distance = (incoeffices - incoeffices.reshape(-1,1)) ** 2
        C = np.exp(-.5 * square_distance / delta ** 2)
        C_inv = np.linalg.inv(C + np.eye(ncoeff) * 1e-07)

        return C, C_inv
    
    def update_C_prior(self, params):

        rho = params[1]
        delta_time = params[2]

        C_t, C_t_inv = self._make_1D_covariance(delta_time, self.dims[0])

        if len(self.dims) == 1:
            C, C_inv = C_t, C_t_inv

        elif 1 < len(self.dims) < 3:
            delta_space = params[3]
            C_s, C_s_inv = self._make_1D_covariance(delta_space, self.dims[1])

            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)  

        elif len(self.dims) > 2:

            delta_spacey = params[3]
            delta_spacex = params[4]
        
            C_sy, C_sy_inv = self._make_1D_covariance(delta_spacey, self.dims[1])
            C_sx, C_sx_inv = self._make_1D_covariance(delta_spacex, self.dims[2])
            
            C_s = np.kron(C_sy, C_sx)
            C_s_inv = np.kron(C_sy_inv, C_sx_inv)

            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)  

        return C, C_inv

    def print_progress_header(self, params):
        
        if len(params) == 3:
            print('Iter\tσ\tρ\tδt\tcost')
        elif len(params) == 4:
            print('Iter\tσ\tρ\tδt\tδs\tcost')
        elif len(params) == 5:
            print('Iter\tσ\tρ\tδt\tδy\tδx\tcost')

    def print_progress(self, i, params, cost):
     
        if len(params) == 3:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}'.format(
                i, params[0], params[1], params[2], cost))  
        elif len(params) == 4:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}'.format(
                i, params[0], params[1], params[2], params[3], cost))  
        elif len(params) == 5:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}'.format(
                i, params[0], params[1], params[2], params[3], params[4], cost))  
