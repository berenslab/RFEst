import numpy as np
from ._initialize_factors import initilize_factors

__all__ = ['SemiNMF']

class SemiNMF:

    """
    
    Semi-Nonnegative Matrix Factorization.

    Multiplicative update rules from Ding el al.(2010)
    
    """

    def __init__(self, V, k, init_method='kmeans', random_seed=2046):

        # store input data
        self.V = V

        # data shape
        self.m, self.n = V.shape
        self.k = k # number of subunits

        # initialize W and H
        self.W, self.H = initilize_factors(V, k, method=init_method, random_seed=random_seed)

    def update_W(self):

        V = self.V
        W = self.W
        H = self.H

        VHt = V @ H.T
        HHt = H @ H.T

        return VHt @ np.linalg.pinv(HHt)

    def update_H(self):

        V = self.V
        W = self.W
        H = self.H

        def pos(A):
            return (np.abs(A) + A) / 2
        def neg(A):
            return (np.abs(A) - A) / 2

        VtW = V.T @ W
        WtW = W.T @ W

        upper = (pos(VtW) + H.T @ neg(WtW)).T + 1e-16
        lower = (neg(VtW) + H.T @ pos(WtW)).T + 1e-16

        return H * np.sqrt( upper/lower )

    def compute_cost(self):
        V = self.V
        WH = self.W @ self.H
        return np.mean((V - WH)**2)

    def fit(self, num_iters=300, verbal=0):

        if verbal:
            self.cost = []
            self.iter = []
            print('{}\t{}'.format('Iter', 'Cost'))

        # start updating
        for itr in range(num_iters):

            self.W = self.update_W()
            self.H = self.update_H()

            if verbal:
                if itr % verbal == 0:
                    self.cost.append(self.compute_cost())
                    self.iter.append(itr)
                    print('{}\t{:.3f}'.format(itr, self.cost[-1]))  


