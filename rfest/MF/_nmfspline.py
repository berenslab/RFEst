import numpy as np
from .._splines import build_spline_matrix

__all__ = ['NMFSpline']


class NMFSpline:

    """
    
    Spline-based Nonnegative Matrix Factorization.
    
    See: Zdunek, R. et al. (2014).
    
    """

    def __init__(self, V, dims, k, df, smooth='cr', random_seed=2046):
        
        """
        Initialize an instance of SemiNMFSpline.
        
        Parameters
        ==========
        V : array_like, shape (n_features, n_samples)
            Spike-triggered ensemble. 
            
        dims : list or array_like (ndim, )
            Dimensions or shape of the RF to estimate. Assumed order [t, sy, sx]
            
        k : int
            Number of subunits
           
        df : int or list, shape (ndim, )
            Degree of freedom for splines.
            
        smooth : str
            Spline or smooth to be used. Current supported methods include:
            * `bs`: B-spline. Fix order to 3.
            * `cr`: Cubic Regression spline
            * `tp`: (Simplified) Thin Plate regression spline           
            
        random_seed : int
            Set pseudo-random seed.
        
        """

        # store RF dimensions
        self.dims = dims
        self.ndim = len(dims)

        # store input data
        self.V = V # data
        self.S = build_spline_matrix(dims, df, smooth) # splines


        # data shape
        self.m, self.n = V.shape
        self.k = k # number of subunits
        self.b = self.S.shape[1] # number of spline coefficients

        # initialize W and H

        np.random.seed(random_seed)
        self.B = np.abs(np.random.rand(self.b, self.k))
        self.H = np.abs(np.random.randn(self.k, self.n))

    def update_B(self):
        
        V = self.V
        S = self.S
        B = self.B
        H = self.H
        
        VHt = V @ H.T
        HHt = H @ H.T
        
        upper = S.T @ VHt + 1e-7
        lower = S.T @ S @ B @ HHt + 1e-7
        
        return B * np.sqrt(upper / lower)
    
    def update_H(self):
        
        V = self.V
        S = self.S
        B = self.B
        H = self.H
            
        W = S @ B
        WtV = W.T @ V
        WtW = W.T @ W
        
        lower = WtW @ H
        
        return H * WtV / lower

    def compute_cost(self):
        V = self.V
        W = self.S @ self.B
        WH = W @ self.H
        return np.mean((V - WH)**2)

    def fit(self, num_iters=300, verbal=0):

        if verbal:
            self.cost = np.zeros(int(np.ceil(num_iters / verbal)))
            print('{}\t{}'.format('Iter', 'Cost'))
        
        # start updating
        for itr in range(num_iters):

            self.B = self.update_B()
            self.H = self.update_H()

            if verbal:
                if itr % verbal == 0:
                    self.cost[itr] = self.compute_cost()
                    print('{}\t{:.3f}'.format(itr, self.cost[itr]))  

            self.W = self.S @ self.B


