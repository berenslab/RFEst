import numpy as np
from .._splines import build_spline_matrix

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
        self.V = V # data
        
        # data shape / dimension
        self.m, self.n = V.shape
        self.k = k # number of subunits
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
                W[:,i] = np.mean(V[:, idx], axis=1)
        
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
        
        assigned = vq(self.W, self.V)
        H = np.zeros(H.shape)
        H[range(self.n), assigned] = 1.0        
        
        return H
    
    def compute_cost(self):
        
        V = self.V
        W = self.W
        H = self.H
        WHt = W @ H.T
        
        return np.mean((V - WHt)**2)

    def fit(self, num_iters=300, verbal=0, tolerance=10):

        if verbal:
            self.cost = []
            self.iter = []
            print('{}\t{}'.format('Iter', 'Cost'))
        
        # start updating
        for itr in range(num_iters):

            self.W, self.B = self.update_W()
            self.H = self.update_H()

            if verbal:
                if itr % verbal == 0:
                    self.cost.append(self.compute_cost())
                    self.iter.append(itr)
                    print('{}\t{:.3f}'.format(itr, self.cost[-1])) 

                    if len(self.cost) >= tolerance and (np.abs(np.diff(self.cost[-tolerance:])) < 1e-7).all():
                        print('Stop: cost has been changing so small in the last {0:03d} chechpoints. Final cost = {1:.3f}'.format(tolerance, self.cost[-1]))
                        break
        else:
            if verbal:
                print('Stop: reached maximum iterations. Final cost = {:.3f}'.format(self.cost[-1]))

def l2_distance(d, vec):    

    ret_val = np.sqrt(((d[:,:] - vec)**2.0).sum(axis=0))
            
    return ret_val.reshape((-1))

def pdist(A, B):
    # compute pairwise distance between a data matrix A (d x n) and B (d x m).
    # Returns a distance matrix d (n x m).
    d = np.zeros((A.shape[1], B.shape[1]))
    if A.shape[1] <= B.shape[1]:
        for aidx in range(A.shape[1]):
            d[aidx:aidx+1,:] = l2_distance(B[:,:], A[:,aidx:aidx+1]).reshape((1,-1))
    else:
        for bidx in range(B.shape[1]):
            d[:, bidx:bidx+1] = l2_distance(A[:,:], B[:,bidx:bidx+1]).reshape((-1,1))    
    return d

def vq(A, B):
    # assigns data samples in B to cluster centers A and
    # returns an index list [assume n column vectors, d x n]
    assigned = np.argmin(pdist(A,B), axis=0)
    return assigned
