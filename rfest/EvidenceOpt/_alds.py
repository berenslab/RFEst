import autograd.numpy as np
from autograd import grad
from autograd.misc.flatten import flatten_func
from scipy.optimize import minimize

from ._base import *
from .._utils import *

class ALDs(EmpiricalBayes):
    
    def __init__(self, X, Y, rf_dims):
        
        super().__init__(X, Y, rf_dims)
        
        self.chi = self.coordinates_matrix(rf_dims)    
        
    def coordinates_matrix(self, rf_dims):

        if len(rf_dims) == 1:

            n = self.dims_sRF[0]
            coo = np.arange(n)

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

        return coo.astype(np.float64)
    
    def initialize_params(self):
        
        sigma = np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples
        rho = -2.3
        
        if len(self.dims_sRF) == 1:
            nu = np.argmax(self.w_mle).astype(np.float64)
            psi_params = [25.]
        elif len(self.dims_sRF) == 2:
            RF = randomized_svd(self.w_mle, 3)[0][:, 0].reshape(self.dims_sRF)
            nu = np.array(np.where(RF == RF.max())).flatten().astype(np.float64).tolist()
            psi_params = [3., 3., 0.]

        return [sigma, rho] + nu + psi_params
#         return [sigma, rho, nu, psi_params]
    
    def update_C_prior(self, params):
        
        rho = params[1]    
 
        if len(self.dims_sRF) == 1:
            
            nu = np.array(params[2])
            diff_matrix = self.chi - nu
            
            psi = params[3]
            
            Psi_inv = 1 / psi[0]

            C_prior = np.diag(np.exp(-0.5 * Psi_inv * diff_matrix ** 2 - rho))
            C_prior += 1e-7*np.eye(self.n_features)
            C_prior_inv = np.linalg.inv(C_prior)
            
        else:
            nu = np.array(params[2:4])            
            psi = params[4:]
            
            diff_matrix = self.chi - nu
              
            Psi = np.array([[psi[0] ** 2, psi[0]* psi[1] * psi[2]],
                            [psi[0]* psi[1] * psi[2], psi[1] ** 2]])
            Psi_inv = np.linalg.inv(Psi)
        
            C = np.exp( -rho -0.5 * np.diag(diff_matrix @ Psi_inv @ diff_matrix.T))
            C_prior = np.diag(C)
            C_prior += 1e-70*np.eye(self.n_features)
            C_prior_inv = np.linalg.inv(C_prior)
        
        return C_prior, C_prior_inv
    
    def objective(self, params):
        return -self.log_evidence(params)
    
    def optimize_params(self, initial_params, step_size, num_iters, bounds, callback ):
        
        flattened_obj, unflatten, flattened_init_params =\
            flatten_func(self.objective, initial_params)
        
        results = minimize(flattened_obj, x0=flattened_init_params, jac=grad(flattened_obj),method='L-BFGS-B',
                              options={'maxiter': num_iters}, bounds=bounds, callback=callback)
        
        params = results.x
        
        return params