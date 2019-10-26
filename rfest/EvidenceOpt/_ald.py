import jax.numpy as np

from ._base import *
from .._utils import *

class ALD(EmpiricalBayes):
    
    def __init__(self, X, Y, dims, compute_mle=True):
        super().__init__(X, Y, dims, compute_mle)

    def _make_1D_covariance(self, params, ncoeff):
        
        chi = np.arange(ncoeff)

        taux = np.array(params[0])
        nux = np.array(params[1])
        tauf = np.array(params[2])
        nuf = np.array(params[3])

        (Uf, freq) = realfftbasis(ncoeff)

        CxSqrt = np.diag(np.exp(-0.25 * 1/taux**2 * (chi - nux)**2))

        Cf = Uf.T @ np.diag(np.exp(-0.5 * (np.abs(tauf * freq) - nuf)**2)) @ Uf
        
        C = CxSqrt @ Cf @ CxSqrt
        C_inv = np.linalg.inv(C + np.eye(ncoeff) * 1e-07)
        
        return C, C_inv

    def update_C_prior(self, params):

        rho = params[1]
        params_time = params[2:6]
        C_t, C_t_inv = self._make_1D_covariance(params_time, self.dims[0])

        if len(self.dims) == 1:
            C, C_inv = C_t, C_t_inv

        elif len(self.dims) == 2:
            params_space = params[6:10]
            C_s, C_s_inv = self._make_1D_covariance(params_space, self.dims[1])

            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)  

        elif len(self.dims) == 3:

            params_spacey = params[6:10]
            params_spacex = params[10:]
        
            C_sy, C_sy_inv = self._make_1D_covariance(params_spacey, self.dims[1])
            C_sx, C_sx_inv = self._make_1D_covariance(params_spacex, self.dims[2])
            
            C_s = np.kron(C_sy, C_sx)
            C_s_inv = np.kron(C_sy_inv, C_sx_inv)

            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)  

        return C, C_inv

    def print_progress(self, i, params, cost):
        print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}\t{8:1.3f}\t{9:1.3f}'.format(
        i, params[0], params[1], params[2], params[4], params[6], params[8], params[10], params[12], cost))   
        
    def print_progress_header(self, params):
        
        if len(params) == 6:
            print('Iter\tσ\tρ\tτs_t\tτf_t\tcost')
        elif len(params) == 10:
            print('Iter\tσ\tρ\tτs_t\tτf_t\tτs_y\tτf_y\tcost')
        elif len(params) == 14:
            print('Iter\tσ\tρ\tτs_t\tτf_t\tτs_y\tτf_y\tτs_x\tτf_x\tcost')

    def print_progress(self, i, params, cost):
     
        if len(params) == 6:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}'.format(
                i, params[0], params[1], params[2], params[4], cost))   
        elif len(params) == 10:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}'.format(
                i, params[0], params[1], params[2], params[4], params[6], params[8], cost))   
        elif len(params) == 14:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}\t{8:1.3f}\t{9:1.3f}'.format(
                i, params[0], params[1], params[2], params[4], params[6], params[8], params[10], params[12], cost))   
