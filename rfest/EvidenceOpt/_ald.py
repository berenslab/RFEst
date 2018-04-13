import autograd.numpy as np
from autograd import grad
from autograd.misc.flatten import flatten_func
from scipy.optimize import minimize

from ._base import *
from .._utils import *

class ALD(EmpiricalBayes):
    
    def __init__(self, X, Y, rf_dims):
        
        self.X = X
        self.Xf = scipy.fftpack.rfft(X)
        
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.rf_dims = rf_dims
        
        if len(rf_dims) == 3:
            self.dims_tRF = rf_dims[2]
            self.dims_sRF = rf_dims[:2]
            self.Y = get_rdm(Y, self.dims_tRF)
        else:
            self.dims_sRF = rf_dims
            self.dims_tRF = None
            self.Y = Y
        
        self.XtX = self.X.T @ self.X
        self.XtY = self.X.T @ self.Y
        self.YtY = self.Y.T @ self.Y
                
        self.w_mle = np.linalg.solve(self.XtX, self.XtY)
        self.sRF_mle, self.tRF_mle = self.SVD(self.w_mle)
    
        self.XftXf = self.Xf.T @ self.Xf
        self.XftY = self.Xf.T @ self.Y
        
        self.w_mle_f = np.linalg.solve(self.XftXf, self.XftY)
        self.sRF_mle_f, self.tRF_mle_f = self.SVD(self.w_mle_f)
        
        self.chi = self.coordinates_matrix()    
        
    def coordinates_matrix(self):
        
        if len(self.dims_sRF) == 1:        
            n = self.dims_sRF[0]
            coo = np.arange(n)

        else:
            
            n, m = self.dims_sRF
            xes = np.zeros([n, m])
            yes = np.zeros([n, m])
            coords = []
            counter = 0
            for x in range(n):
                for y in range(m):
                    coords.append([x, y])
            coo = np.vstack(coords) 
        
        return coo

    
    def initialize_params(self):
        
        sigma = np.sum((self.Y - self.X @ self.w_mle) ** 2) / self.n_samples
        rho = -2.3
        
        if len(self.dims_sRF) == 1:
            
            nux = [np.argmax(self.w_mle).astype(np.float64)]
            nuf = [np.argmax(self.w_mle_f).astype(np.float64)]
            
            taux = [25.]
            tauf = [0.1]
        
        elif len(self.dims_sRF) == 2:
            
            nux = np.array(np.where(self.sRF_mle == self.sRF_mle.max())).flatten().astype(np.float64).tolist()
            nuf = np.array(np.where(self.sRF_mle_f == self.sRF_mle_f.max())).flatten().astype(np.float64).tolist()
            taux = [5., 5., 0.]
            tauf = [0.1, 0.1, 0.]
        
        return [sigma, rho, nux, nuf, taux, tauf]
    
    def update_C_prior(self, params):
        
        if len(self.dims_sRF) == 1:
            
            rho = np.array(params[1])
            nux = np.array(params[2])
            nuf = np.array(params[3])
            taux = np.array(params[4])
            tauf = np.array(params[5])
            
            Mx = 1 / taux**2 * np.eye(2)
            Mf = tauf * np.eye(2)
            
            (Uf, freq) = realfftbasis(self.dims_sRF[0])
            
            CxSqrt = np.diag(np.exp(-0.25 * taux**2 * (self.chi - nux)**2))
            
            Cf = Uf.T @ np.diag(np.exp(-0.5 * (np.abs(tauf * freq) - nuf)**2)) @ Uf
            
        else:
            
            rho = np.array(params[1])
            nux = np.array(params[2])
            nuf = np.array(params[3])
            taux = np.array(params[4])
            tauf = np.array(params[5])
            
            Mx = np.array([[1/taux[0]**2, -taux[2] / taux[0] * taux[1]],
                           [-taux[2]/taux[0] * taux[1], 1/taux[1]**2]])
            Mx *= 1 / (1 - taux[2]**2)

            Mf = np.array([[tauf[0], tauf[2]],
                           [tauf[2], tauf[1]]])
            
            (Ufx, freqx) = realfftbasis(self.dims_sRF[0])
            (Ufy, freqy) = realfftbasis(self.dims_sRF[1])
            
            Uf = np.kron(Ufy, Ufx)
            
            [ffy, ffx] = np.meshgrid(freqx, freqy)
            freq = np.vstack([ffx.flatten(), ffy.flatten()]).T
            
            diffx = self.chi - nux
            CxSqrt = np.diag(np.exp(- rho -0.25 * np.sum((Mx @ diffx.T)**2, 0)))
            
            diffy = abs(Mf @ freq.T).T - nuf
            Cf = Uf.T @ np.diag(np.exp(- rho -0.5 *  np.sum(diffy.T**2, 0))) @ Uf
            
            
        C_prior = CxSqrt @ Cf @ CxSqrt
        C_prior += 1e-7 * np.eye(self.n_features)
        C_prior_inv = np.linalg.inv(C_prior)
        
        return C_prior, C_prior_inv         