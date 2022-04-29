import time

import numpy as np

from rfest.splines import build_spline_matrix

__all__ = ['NMF']


class NMF:
    """

    Nonnegative Matrix Factorization with spline-based factors.

    Modifiled from: Zdunek, R. et al. (2014).

    """

    def __init__(self, V, k, init_method='random', random_seed=2046, **kwargs):

        # build basis or not

        self.dims = kwargs['dims'] if 'dims' in kwargs else None
        self.df = kwargs['df'] if 'df' in kwargs else None

        self.smooth = kwargs['smooth'] if 'smooth' in kwargs else 'cr'

        self.S = build_spline_matrix(self.dims, self.df, self.smooth) if self.df is not None else None

        # store input data
        self.V = V  # data

        # data shape / dimension
        self.m, self.n = V.shape
        self.k = k  # number of subunits
        self.b = self.S.shape[1] if self.S is not None else None

        # initialize W and H

        np.random.seed(random_seed)

        if self.S is not None:
            self.B = np.abs(np.random.rand(self.b, self.k))
            self.W = self.S @ self.B

        else:
            self.B = None
            self.W = np.abs(np.random.rand(self.m, self.k))

        self.H = np.abs(np.random.rand(self.n, self.k))

    def update_W(self):

        # data
        V = self.V

        # factors
        H = self.H
        W = self.W

        # basis
        S = self.S
        B = self.B

        VH = V @ H
        HtH = H.T @ H

        if S is not None:

            upper = S.T @ VH + 1e-7
            lower = S.T @ S @ B @ HtH + 1e-7

            B *= np.sqrt(upper / lower)
            W = S @ B

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
        VtW = V.T @ W
        WtW = W.T @ W

        upper = VtW
        lower = H @ WtW

        H *= upper / lower

        return H

    def compute_cost(self):

        V = self.V
        W = self.W
        H = self.H
        WHt = W @ H.T

        return np.mean((V - WHt) ** 2)

    def fit(self, num_iters=300, verbose=0, tolerance=10):

        if verbose:
            self.cost = []
            self.iter = []
            print('{0:>1}\t{1:>10}\t{2:>10}'.format('Iter', 'Cost', 'Time (s)'))

        # start updating
        time_start = time.time()
        for itr in range(num_iters):

            self.H = self.update_H()
            self.W, self.B = self.update_W()

            time_elapsed = time.time() - time_start

            if verbose:
                if itr % verbose == 0:
                    self.cost.append(self.compute_cost())
                    self.iter.append(itr)
                    print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}'.format(itr, self.cost[-1], time_elapsed))

                    if len(self.cost) >= tolerance and (np.abs(np.diff(self.cost[-tolerance:])) < 1e-7).all():
                        total_time_elapsed = time.time() - time_start

                        print(
                            'Stop: cost has been changing so small in the last {0:03d} chechpoints. Final cost = {1:.03f}, total time elapsed = {2:.03f} s'.format(
                                tolerance, self.cost[-1], total_time_elapsed))
                        break
        else:
            if verbose:
                total_time_elapsed = time.time() - time_start

                print('Stop: reached maximum iterations. Final cost = {0:.03f}, total time elapsed = {1:.03f} s'.format(
                    self.cost[-1], total_time_elapsed))
