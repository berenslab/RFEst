import jax.numpy as np
from ._base import *

__all__ = ['ALD']

class ALD(EmpiricalBayes):

    """

    Automatic Locality Determination (ALD).

    Reference: Park, M., & Pillow, J. W. (2011).

    See also: https://github.com/leaduncker/SimpleEvidenceOpt

    """
    
    def __init__(self, X, Y, dims, compute_mle=True):
        super().__init__(X, Y, dims, compute_mle)
        self.n_hyperparams_1d = 4

    def _make_1D_covariance(self, params, ncoeff):

        """
        
        1D Locality prior covariance. 

        See eq(11, 12, 13) in Park & Pillow (2011).

        """
        
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

    def print_progress_header(self, params):
        
        print('* Due to space limit, parameters for ν are not printed.')
        if len(params) == 6:
            print('Iter\tσ\tρ\tτs_t\tτf_t\tcost')
        elif len(params) == 10:
            print('Iter\tσ\tρ\tτs_t\tτf_t\tτs_y\tτf_y\tcost')
        elif len(params) == 14:
            print('Iter\tσ\tρ\tτs_t\tτf_t\tτs_y\tτf_y\tτs_x\tτf_x\tcost')

    def print_progress(self, i, params, cost):
     
        # due to space limit, parameters for \nu are not printed.
        if len(params) == 6:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}'.format(
                i, params[0], params[1], params[2], params[4], cost))   
        elif len(params) == 10:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}'.format(
                i, params[0], params[1], params[2], params[4], params[6], params[8], cost))   
        elif len(params) == 14:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}\t{8:1.3f}\t{9:1.3f}'.format(
                i, params[0], params[1], params[2], params[4], params[6], params[8], params[10], params[12], cost))  


def realfftbasis(nx):
    
    """
    Basis of sines+cosines for nn-point discrete fourier transform (DFT).
    
    Ported from MatLab code:
    https://github.com/leaduncker/SimpleEvidenceOpt/blob/master/util/realfftbasis.m
    
    """
    
    nn = nx
    
    ncos = np.ceil((nn + 1) / 2)
    nsin = np.floor((nn - 1) / 2)
    
    wvec = np.hstack([np.arange(ncos), np.arange(-nsin, 0)])
    
    wcos = wvec[wvec >= 0]
    wsin = wvec[wvec < 0]
    
    x = np.arange(nx)
    
    t0 = np.cos(np.outer(wcos * 2 * np.pi / nn, x))
    t1 = np.sin(np.outer(wsin * 2 * np.pi / nn, x))
    
    B = np.vstack([t0, t1]) / np.sqrt(nn * 0.5)
    
    return B, wvec 
