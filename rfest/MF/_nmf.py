import numpy as np
from ._initialize_factors import initilize_factors

__all__ = ['NMF']

def normalize(A):
    return A / np.linalg.norm(A)

class NMF:

    def __init__(self, V, k, alg='mu', init_method='nndsvd', norm=True, random_seed=2046):


        """

        Nonnegative matrix factorization. 

        Parameters
        ==========

        V : array_like, shape (m, n)
            Nonnegative matrix to be factorized.
            
        k : int
            Rank of the computed factors W and H

        init_method : str
            Methods for initializing factors.
            
            Current supported initialization methods:
            
            * 'nndsvd': Nonnegative Double Singular Value Decomposition. 
                        A SVD-based initilization. See Boutsidisa & Gallopoulos, 2008
            * 'random': Initialize with random values.
            * 'one': Initialize with 1.
            * 'kmeans': initialize with K-means clustering.

        alg : str
            Algorithm for NMF.
            
            Current supported methods:

            * 'mu': multiplicative update rules with Euclidean distance
                    See eq(4) in Lee & Seung, 2001
            * 'mu-kl': multiplicative update rules with Kullback-Leibler divergence. 
                    See eq(5) in Lee & Seung, 2001

        norm : bool
            Normalize input data. 

        random_seed : int
            Random seed for pseudo random sequence.


        References
        ==========

        Boutsidis, C., & Gallopoulos, E. (2008). SVD based initialization: A head start 
            for nonnegative matrix factorization. Pattern recognition, 41(4), 1350-1362.

        Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. 
            In Advances in neural information processing systems (pp. 556-562).   

        """


        self.V = V 

        self.norm = norm
        if self.norm:
            self.V = normalize(self.V)
        
        self.m, self.n = V.shape
        self.k = k
        
        self.alg = alg
        
        self.W, self.H = initilize_factors(V, k, method=init_method, random_seed=random_seed)
        
        if self.norm:
        
            self.W = normalize(self.W)
            self.H = normalize(self.H)
        
    def update_W(self):
        
        # copy data
        V = self.V
        W = self.W
        H = self.H

        # update rules
        if self.alg == 'mu':

            W *= (V @ H.T) / (W @ H @ H.T)

        elif self.alg == 'mu-kl':

            W *=  ((V / (W @ H)) @ H.T) / (np.ones(self.m) * W.sum(0)[:, np.newaxis]).T
        

        return W + 1e-9


    def update_H(self):
        
        # copy data
        V = self.V
        W = self.W
        H = self.H
        
        # update rules
        if self.alg == 'mu':
                        
            H *= (W.T @ V) / (W.T @ W @ H)
            
        elif self.alg == 'mu-kl':
            
            H *= (W.T @ (V / (W @ H))) / (np.ones(self.n) * H.sum(1)[:, np.newaxis])
        
        else:
            
            raise NotImplementedError('`alg={}` is not implemented yet'.format(self.alg))
            
        return H + 1e-9
    
    def compute_cost(self):
        
        V = np.ravel(self.V) + 1e-9
        W = self.W
        H = self.H
        
        if self.alg == 'mu':
            
            cost = np.mean(np.square(V - np.ravel(W @ H)))
            
        elif self.alg == 'mu-kl':
            
            cost = np.sum(np.where(V != 0, V * np.log(V / np.ravel((W @ H) + 1e-9)), 0))
            
        return cost
        
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

                    if len(self.cost) >= 10 and (np.abs(np.diff(self.cost[-10:])) < 1e-7).all():
                        print('Stop: cost has been changing so small in the last ten chechpoint. Final cost = {:.3f}'.format(self.cost[-1]))
                        break

