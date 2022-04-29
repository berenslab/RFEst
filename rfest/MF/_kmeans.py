import time

import numpy as np
from scipy.spatial.distance import cdist

from rfest.splines import build_spline_matrix

__all__ = ['KMeans']


class KMeans:
    """

    Kmeans clustering. V = WH, where W is the cluster centroid, H the cluster labels.

    Modified from https://github.com/cthurau/pymf/blob/master/pymf/kmeans.py
    
    """

    def __init__(self, V, k=2, random_seed=2046, **kwargs):

        # meta
        self.rcond = kwargs['rcond'] if 'rcond' in kwargs.keys() else None
        self.random_seed = random_seed

        # build spline matrix, or not

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

        self.H = np.zeros((self.n, self.k))
        self.W = self.V[:, np.sort(np.random.choice(range(self.n), self.k))]

    def update_W(self):

        V = self.V
        W = self.W
        H = self.H
        S = self.S

        for i in range(self.k):
            # cast to bool to use H as an index for data
            idx = np.array(H[:, i], dtype=np.bool)
            n = np.sum(idx)
            if n > 0:
                W[:, i] = np.mean(V[:, idx], axis=1)

        if S is not None:

            W0 = W.copy()

            B = np.linalg.lstsq(S, W0, rcond=self.rcond)[0]
            W = S @ B

        else:
            B = None

        return W, B

    def update_H(self):

        V = self.V
        W = self.W
        H = self.H

        assigned = np.argmin(cdist(self.W.T, self.V.T), axis=0)
        H = np.zeros(H.shape)
        H[range(self.n), assigned] = 1.0

        return H

    def compute_cost(self):

        V = self.V
        W = self.W
        H = self.H
        WHt = W @ H.T

        return np.mean((V - WHt) ** 2)

    def fit(self, num_iters=300, lambd=0.05, verbose=0, tolerance=10):

        # regularzation
        self.lambd = lambd

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
                            'Stop: cost has been changing so small in the last {0} check points. Final cost = {1:.03f}, total time elapsed = {2:.03f} s'.format(
                                tolerance, self.cost[-1], total_time_elapsed))
                        break
        else:
            total_time_elapsed = time.time() - time_start
            if verbose:
                print('Stop: reached {0} iterations. Final cost = {1:.3f}, total time elapsed = {2:.3f} s'.format(
                    num_iters, self.cost[-1], total_time_elapsed))
