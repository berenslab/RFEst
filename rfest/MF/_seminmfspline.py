import numpy as np
import patsy

__all__ = ['SemiNMFSpline']

class SemiNMFSpline:

    """
    Spline-based Semi-Nonnegative Matrix Factorization.

    Smoothness only applies to the left factorized matrix.

    Inspired by multiplicative update rules from Ding el al.(2010)
    """

    def __init__(self, V, dims, df, k, random_seed=2046):

        # store RF dimensions
        self.dims = dims
        self.ndim = len(dims)

        # store input data
        self.V = V # data
        self.S = self._make_splines_matrix(df) # splines


        # data shape
        self.m, self.n = V.shape
        self.k = k # number of subunits
        self.b = self.S.shape[1] # number of spline coefficients

        # initialize W and H

        np.random.seed(random_seed)
        self.B = np.random.randn(self.b, self.k)
        self.H = np.abs(np.random.randn(self.k, self.n))


    def _make_splines_matrix(self, df):
        
        if np.ndim(df) != 0 and len(df) != self.ndim:
            raise ValueError("`df` must be an integer or an array the same length as `dims`")
        elif np.ndim(df) == 0:
            df = np.ones(self.ndim) * df
        
        if self.ndim == 1:
        
            S = patsy.cr(np.arange(self.dims[0]), df[0])
            
        elif self.ndim == 2:
        
            g0, g1 = np.meshgrid(np.arange(self.dims[0]), np.arange(self.dims[1]), indexing='ij')
            S = patsy.te(patsy.cr(g0.ravel(), df[0]), patsy.cr(g1.ravel(), df[1]))
            
        elif self.ndim == 3:
            
            g0, g1, g2 = np.meshgrid(np.arange(self.dims[0]), 
                                     np.arange(self.dims[1]), 
                                     np.arange(self.dims[2]), indexing='ij')
            S = patsy.te(patsy.cr(g0.ravel(), df[0]), 
                         patsy.cr(g1.ravel(), df[1]), 
                         patsy.cr(g2.ravel(), df[2]))
            
        return S


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

