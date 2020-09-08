import time
import numpy as np
from tqdm.auto import trange
from ..utils import uvec_rows, softthreshold


class SparseCoding:

    """
    
     (Olshausen & Field, Nature. 1996). 
     
     Code modified from https://github.com/takyamamoto/SparseCoding-OlshausenField-Model

    """

    def __init__(self, n, m, k, random_seed=2046):

        self.n = n # number of samples
        self.m = m # number of features
        self.k = k # number of basis / subunits

        W = np.zeros([n, k])
        H = np.random.randn(m, k)

        self.W = W
        self.H = H / np.sqrt(1 / n)

    def reset_W(self):
        return np.zeros([self.n, self.k]) 

    def calculate_total_error(self, error):

        recon_error = np.mean(error ** 2)
        sparsity_W = self.beta * np.mean(np.abs(self.W))

        return recon_error + sparsity_W

    def cost(self, V):
        return V - self.W @ self.H.T

    def update_W(self, V):
        
        error = self.cost(V) 
        W = self.W + self.lr_W * error @ self.H

        return softthreshold(W, self.beta), error

    def update_H(self, V):

        error = self.cost(V)
        dH = error.T @ self.W
        H = self.H + self.lr_H * dH

        return H, error

    def fit(self, V, num_epochs, num_iters, lr_W=1e-2, lr_H=1e-2, beta=5e-3, eps=1e-2, verbose=0):

        # check input
        if len(V.shape) == 3:
            num_batches, batch_size, num_features = V.shape
            if  batch_size != self.n or num_features != self.m:
                raise ValueError('Input `V` must be in the shape of (num_batches, n, m)')

        self.lr_W = lr_W
        self.lr_H = lr_H
        self.beta = beta

        error_list = []

        for epoch in trange(num_epochs):

            self.W = self.reset_W()
            self.H = uvec_rows(self.H)

            W_prev = self.W.copy()
            V_curr = V[epoch]

            for i in range(num_iters):

                self.W, error = self.update_W(V_curr)
                dW = self.W - W_prev

                dW_norm = np.linalg.norm(dW, ord=2) / (eps + np.linalg.norm(W_prev, ord=2))
                W_prev = self.W

                if dW_norm < eps:
                    self.H, error = self.update_H(V_curr)
                    break

            error_list.append(self.calculate_total_error(error))

            if verbose:
                if epoch % verbose == verbose-1:
                    if verbose == 1:
                        cost = error_list[epoch]
                    else:
                        cost = np.mean(error_list[epoch+1-verbose:epoch+1])
                    print(f'epoch={epoch}, Cost = {cost:.3f}')

        self.error_list = error_list