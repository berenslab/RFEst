import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def nndsvd(V, k, option=0):
    """
    
    This function implements the NNDSVD algorithm described in [1] for
    initialization of Nonnegative Matrix Factorization Algorithms.    
    
    
    Parameters
    ==========
   
    V : array_like, shape (m, n)
        Nonnegative matrix to be factorized.
        
    k : int
        Rank of the computed factors W and H
        
    option : str
        Replace small elements with customed values
            * 0 => 0 
            * 1 => mean of V
            * 2 => random value between 0 and mean(V)/100
            
    
    Return
    ======
    
    W : array_like, shape (m, k)
        Nonnegative left factor. 
    
    H : array_like, shape (k, n)
        Nonnegative right factor. 
        
    
    Reference
    =========
    
    C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head start
        for nonnegative matrix factorization, Pattern Recognition, Elsevier
    
    """

    if sum(np.ravel(V) < 0).any():
        raise ValueError('The input matrix contains negative elements!')

    m, n = V.shape

    W = np.zeros([m, k])
    H = np.zeros([k, n])

    U, S, Vt = np.linalg.svd(V)

    W[:, 0] = np.sqrt(S[0] * np.abs(U[:, 0]))
    H[0, :] = np.sqrt(S[0] * np.abs(Vt[:, 0].T))

    for i in range(1, k):

        uu = U[:, i]
        vv = Vt[:, i]

        uup = np.maximum(uu, 0)
        uun = np.abs(np.minimum(uu, 0))
        vvp = np.maximum(vv, 0)
        vvn = np.abs(np.minimum(vv, 0))

        n_uup = np.linalg.norm(uup)
        n_uun = np.linalg.norm(uun)
        n_vvp = np.linalg.norm(vvp)
        n_vvn = np.linalg.norm(vvn)

        termp = n_uup * n_vvp
        termn = n_uun * n_vvn

        if termp >= termn:
            W[:, i] = np.sqrt(S[i] * termp) * uup / n_uup
            H[i, :] = np.sqrt(S[i] * termp) * vvp.T / n_vvp
        else:
            W[:, i] = np.sqrt(S[i] * termn) * uun / n_uun
            H[i, :] = np.sqrt(S[i] * termn) * vvn.T / n_vvn

    if option == 0:

        # fill small elements with 0
        W[W < 1e-9] = 0
        H[H < 1e-9] = 0

    elif option == 1:

        # fill small elements with mean(V)
        W[W < 1e-9] = np.mean(V)
        H[H < 1e-9] = np.mean(V)

    elif option == 2:

        # fill small elemetns with random value drawn from
        # 0 to mean(V) / 100
        W[W < 1e-9] = np.random.uniform(0, np.mean(V) / 100)
        H[H < 1e-9] = np.random.uniform(0, np.mean(V) / 100)

    else:

        raise ValueError('`option` can only be chosen within [0, 1, 2]')

    return W, H


def initialize_factors(V, k, method='nndsvd', **kwargs):
    m, n = V.shape

    if method == 'random':
        random_seed = kwargs.get('random_seed', 2046)
        np.random.seed(random_seed)

        W = np.maximum(np.random.randn(m, k), 0)
        H = np.maximum(np.random.randn(k, n), 0)

    elif method == 'one':

        W = np.ones([m, k])
        H = np.ones([k, n])

    elif method == 'nndsvd':
        option = kwargs.get('nndsvd_option', 0)
        W, H = nndsvd(V, k, option)

    elif method == 'kmeans':
        random_seed = kwargs.get('random_seed', 2046)

        pca = PCA(n_components=30).fit(V)

        km = KMeans(n_clusters=k, random_state=random_seed).fit(pca.components_.T)
        W = np.vstack([V.T[km.labels_ == i].mean(0) for i in range(k)]).T
        WtW = W.T @ W
        WtW = np.maximum(WtW, WtW.T)
        WtV = W.T @ V
        H = np.maximum(np.linalg.solve(WtW, WtV), 0)

    else:
        raise NotADirectoryError(method)

    return W, H.T
