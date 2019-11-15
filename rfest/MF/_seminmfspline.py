import numpy as np
from .._splines import build_spline_matrix 

__all__ = ['SemiNMFSpline']

class SemiNMFSpline:

    """
    
    Spline-based Semi-Nonnegative Matrix Factorization.

    Smoothness only applies to the left factorized matrix.

    Inspired by multiplicative update rules from Ding el al.(2010)
    
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
            * `bs`: B-spline
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
        self.B = np.random.randn(self.b, self.k)
        self.H = np.abs(np.random.randn(self.k, self.n))

    def update_B(self):

        V = self.V
        S = self.S
        B = self.B
        H = self.H

        VHt = V @ H.T
        HHt = H @ H.T

        return np.linalg.lstsq(S, VHt @ np.linalg.pinv(HHt), rcond=None)[0]

    def update_H(self):

        V = self.V
        S = self.S
        B = self.B
        H = self.H

        def pos(A):
            return (np.abs(A) + A) / 2
        def neg(A):
            return (np.abs(A) - A) / 2

        W = S @ B
        VtW = V.T @ W
        WtW = W.T @ W

        upper = (pos(VtW) + H.T @ neg(WtW)).T + 1e-16
        lower = (neg(VtW) + H.T @ pos(WtW)).T + 1e-16

        return H * np.sqrt( upper/lower )

    def compute_cost(self):
        V = self.V
        W = self.S @ self.B
        WH = W @ self.H
        return np.mean((V - WH)**2)

    def fit(self, num_iters=300, verbal=0):

        if verbal:
            if itr % int(verbal) == 0:
                self.cost = []
                print('{}\t{}'.format('Iter', 'Cost'))
        
        # start updating
        for itr in range(num_iters):

            self.B = self.update_B()
            self.H = self.update_H()

            if verbal:
                if itr % verbal == 0:
                    self.cost.append(self.compute_cost())
                    print('{}\t{:.3f}'.format(itr, self.cost[-1]))  

            self.W = self.S @ self.B

