import numpy as np

from ._initialize import initialize_factors
from .._splines import build_spline_matrix

__all__ = ['NMF']

class NMF:

    """

    Nonnegative Matrix Factorization with spline-based factors.

    Modifiled from: Zdunek, R. et al. (2014).

    """

    def __init__(self, V, k, random_seed=2046, **kwargs):

        # build basis or not
        self.build_L = kwargs['build_L'] if 'build_L' in kwargs.keys() else False
        self.build_R = kwargs['build_R'] if 'build_R' in kwargs.keys() else False

        self.dims_L = kwargs['dims_L'] if self.build_L else None
        self.df_L = kwargs['df_L'] if self.build_L else None

        self.dims_R = kwargs['dims_R'] if self.build_R else None
        self.df_R = kwargs['df_R'] if self.build_R else None

        self.smooth_L = kwargs['smooth_L'] if 'smooth_L' in kwargs.keys() else 'cr'
        self.smooth_R = kwargs['smooth_R'] if 'smooth_R' in kwargs.keys() else 'bs'

        self.L = build_spline_matrix(self.dims_L, self.df_L, self.smooth_L) if self.build_L else None
        self.R = build_spline_matrix(self.dims_R, self.df_R, self.smooth_R) if self.build_R else None


        # store input data
        self.V = V # data

        # data shape / dimension
        self.m, self.n = V.shape
        self.k = k # number of subunits
        self.b = self.L.shape[1] if self.L is not None else None
        self.d = self.R.shape[1] if self.R is not None else None

        # initialize W and H

        np.random.seed(random_seed)

        if self.L is not None:
            self.B = np.abs(np.random.rand(self.b, self.k))
            self.W = self.L @ self.B
        else:
            self.B = None
            self.W = np.abs(np.random.rand(self.m, self.k))

        if self.R is not None:
            self.D = np.abs(np.random.rand(self.d, self.k))
            self.H = self.R @ self.D
        else:
            self.D = None
            self.H = np.abs(np.random.rand(self.n, self.k))

    def update_W(self):

        # data
        V = self.V

        # factors
        H = self.H
        W = self.W

        # basis
        L = self.L
        R = self.R

        # basis coeff
        B = self.B
        D = self.D

        VH = V @ H
        HtH = H.T @ H

        if L is not None:

            upper = L.T @ VH + 1e-7
            lower = L.T @ L @ B @ HtH + 1e-7

            B *= np.sqrt(upper / lower)
            W = L @ B

        else:

            upper = VH
            lower = W @ HtH

            W *= upper / lower

        return W, B

    def update_H(self):

        # data
        V = self.V

        # factors
        H = self.H
        W = self.W

        # basis
        L = self.L
        R = self.R

        # basis coeff
        B = self.B
        D = self.D

        VtW = V.T @ W
        WtW = W.T @ W

        if R is not None:

            upper = R.T @ VtW + 1e-7
            lower = R.T @ R @ D @ WtW + 1e-7

            D *= np.sqrt(upper/lower)
            H = R @ D

        else:

            upper = VtW
            lower = H @ WtW

            H *= upper / lower

        return H, D

    def compute_cost(self):

        V = self.V
        W = self.W
        H = self.H
        WHt = W @ H.T

        return np.mean((V - WHt)**2)

    def fit(self, num_iters=300, verbal=0):


        if verbal:
            self.cost = []
            self.iter = []
            print('{}\t{}'.format('Iter', 'Cost'))

        # start updating
        for itr in range(num_iters):

            self.W, self.B = self.update_W()
            self.H, self.D = self.update_H()


            if verbal:
                if itr % verbal == 0:
                    self.cost.append(self.compute_cost())
                    self.iter.append(itr)
                    print('{}\t{:.3f}'.format(itr, self.cost[-1]))

                    if len(self.cost) >= 10 and (np.abs(np.diff(self.cost[-10:])) < 1e-7).all():
                        print('Stop: cost has been changing so small in the last ten chechpoint. Final cost = {:.3f}'.format(self.cost[-1]))
                        break
        else:
            if verbal:
                print('Stop: reached maximum iterations. Final cost = {:.3f}'.format(self.cost[-1]))
