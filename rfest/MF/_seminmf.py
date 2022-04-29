import time

import numpy as np

from rfest.MF._initialize import initialize_factors
from rfest.splines import build_spline_matrix

__all__ = ['semiNMF']


class semiNMF:
    """
    
    Nonnegative Matrix Factorization with spline-based factors.
    
    Modified from: Ding, et al. (2010)
    
    """

    def __init__(self, V, k=2, init_method='random', random_seed=2046, **kwargs):
        # meta
        self.rcond = kwargs['rcond'] if 'rcond' in kwargs.keys() else None
        self.random_seed = random_seed

        # build basis or not 
        self.build_S = kwargs['build_S'] if 'build_S' in kwargs.keys() else False

        self.dims = kwargs['dims'] if self.build_S else None
        self.df = kwargs['df'] if self.build_S else None

        self.smooth = kwargs['smooth'] if 'smooth' in kwargs.keys() else 'cr'

        self.S = build_spline_matrix(self.dims, self.df, self.smooth) if self.build_S else None

        # store input data
        self.V = V  # data

        # data shape / dimension
        self.m, self.n = V.shape
        self.k = k  # number of subunits
        self.b = self.S.shape[1] if self.S is not None else None

        # initialize W and H

        np.random.seed(random_seed)

        self.W, self.H = initialize_factors(V, k, method=init_method, random_seed=random_seed)

        if self.S is not None:
            if init_method == 'random':
                self.B = np.random.randn(self.b, self.k)
                self.W = self.S @ self.B
            else:
                self.B = np.linalg.lstsq(self.S, self.W, rcond=self.rcond)[0]
        else:
            self.B = None

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

        try:
            VHHtHinv = VH @ np.linalg.inv(HtH)
        except:
            VHHtHinv = VH @ np.linalg.pinv(HtH)

        if S is not None:
            B = np.linalg.lstsq(S, VHHtHinv, rcond=self.rcond)[0]
            W = S @ B
        else:
            W = VHHtHinv

        # W /= np.maximum(np.linalg.norm(W, axis=0), 1e-7)

        return W, B

    def update_H(self):

        def pos(A):
            return (np.abs(A) + A) / 2

        def neg(A):
            return (np.abs(A) - A) / 2

        # data
        V = self.V

        # factors 
        H = self.H
        W = self.W

        VtW = V.T @ W
        WtW = W.T @ W

        upper = pos(VtW) + H @ neg(WtW) + 1e-7
        lower = neg(VtW) + H @ pos(WtW) + 1e-7

        H *= np.sqrt(upper / lower)

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
            # self.H, self.D = self.update_H()

            time_elapsed = time.time() - time_start

            if verbose:
                if itr % verbose == 0:
                    self.cost.append(self.compute_cost())
                    self.iter.append(itr)
                    print('{0:>5}\t{1:>10.3f}\t{2:>10.3f}'.format(itr, self.cost[-1], time_elapsed))

                    if len(self.cost) >= 10 and (np.abs(np.diff(self.cost[-10:])) < 1e-7).all():
                        total_time_elapsed = time.time() - time_start
                        print(
                            'Stop: cost has been changing so small in the last {0} check points. Final cost = {1:.03f}, total time elapsed = {2:.03f} s'.format(
                                tolerance, self.cost[-1], total_time_elapsed))
                        break
        else:
            if verbose:
                total_time_elapsed = time.time() - time_start

                print('Stop: reached {0} iterations. Final cost = {1:.03f}, total time elapsed = {1:.03f} s'.format(
                    num_iters, self.cost[-1], total_time_elapsed))
