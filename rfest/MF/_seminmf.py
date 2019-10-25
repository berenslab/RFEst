import numpy as np

__all__ = ['SemiNMF']

class SemiNMF:

    """
    Semi-Nonnegative Matrix Factorization.

    Multiplicative update rules from Ding el al.(2010)
    """


    def __init__(self, V, k, random_seed=2046):

        # store input data
        self.V = V

        # data shape
        self.m, self.n = V.shape
        self.k = k # number of subunits

        # initialize W and H

        np.random.seed(random_seed)
        self.W = np.random.randn(self.m, self.k)
        self.H = np.abs(np.random.randn(self.k, self.n))

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

    def fit(self, num_iters=300, compute_cost=False):

        # initialize cost object
        self.cost = np.zeros(num_iters)

        # start updating
        for itr in tqdm(range(num_iters)):

            self.W = self.update_W()
            self.H = self.update_H()

            if compute_cost:
                self.cost[itr] = self.compute_cost()
