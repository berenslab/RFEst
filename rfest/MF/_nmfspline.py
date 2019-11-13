import numpy as np
import patsy

__all__ = ['NMFSpline']


class NMFSpline:

    """
    
    Spline-based Nonnegative Matrix Factorization.
    
    See: Zdunek, R. et al. (2014).
    
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
        self.B = np.abs(np.random.rand(self.b, self.k))
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


