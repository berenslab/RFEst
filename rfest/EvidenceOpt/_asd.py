import jax.numpy as np
from ._base import EmpiricalBayes

__all__ = ['ASD']

class ASD(EmpiricalBayes):
    
    """

    Automatic Smoothness Determination (ASD).

    Reference: Sahani, M., & Linden, J. F. (2003). 

    """

    def __init__(self, X, y, dims, compute_mle=True):
        
        super().__init__(X, y, dims, compute_mle)
        self.n_hyperparams_1d = 1

    def _make_1D_covariance(self, params, ncoeff):

        """
    
        1D Squared exponential (SE) covariance.

        See eq(10) in Sahani & Linden (2003).

        """

        delta = params[0]

        grid = np.arange(ncoeff)
        square_distance = (grid - grid.reshape(-1,1)) ** 2 # pairwise squared distance
        C = np.exp(-.5 * square_distance / delta ** 2)
        C_inv = np.linalg.inv(C + np.eye(ncoeff) * 1e-07)

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
