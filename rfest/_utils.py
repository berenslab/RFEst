import numpy as np
from sklearn.decomposition import randomized_svd

def build_design_matrix(X, nlag, shift=0):

    """

    Build design matrix. 

    Parameters
    ==========

    X : array_like, shape (n_samples, 1 or n_pixels_per_frame)
        
        Input stimulus. Each row is one frame of the stimulus. For example:
        
        * Full field flicker: (n_samples, 1)
        * Flicker Bar: (n_samples, n_bars)
        * 3D noise: (n_samples, n_pixels)

    nlag: int

        Time lag, or number of frames in the temporal filters. 

    shift : int
        In case of building spike-history filter, the spike train should be shifted
        (e.g. shift=1) so that it will not predict itself.
    
    Return
    ======
    
    X_design: array_like, shape (n_samples, n_features)

    
    Examples
    ========

    >>> X = np.array([0, 1, 2])[:, np.newaxis]
    >>> np.allclose(get_stimulus_design_matrix(X, 2), np.array([[0, 0], [0, 1], [1, 2]]))
    True

    """
    
    n_sample = X.shape[0]
    n_feature = X.shape[1:]
    X = X.reshape(n_sample, np.prod(n_feature))
    X_padded = np.vstack([np.zeros([nlag+shift-1, np.prod(n_feature)]), X])
    X_design = np.hstack([X_padded[i:n_sample+i] for i in range(nlag)])
    
    return X_design

def get_spatial_and_temporal_filters(w, dims):

    """
    
    Asumming a RF is time-space separable, 
    get spatial and temporal filters using SVD. 

    Paramters
    =========

    w : array_like, shape (nt, nx, ny) or (nt, nx * ny)

        2D or 3D Receptive field. 

    dims : list or array_like, shape (ndim, )

        Number of coefficients in each dimension. 
        Assumed order [t, x, y]

    Return
    ======

    [sRF, tRF] : list, shape [2, ]
        
        Spatial and temporal filters separated by SVD. 

    """
    
    if len(dims) == 3:
        
        dims_tRF = dims[0]
        dims_sRF = dims[1:]
        U, S, Vt = randomized_svd(w.reshape(dims_tRF, np.prod(dims_sRF)), 3)
        sRF = Vt[0].reshape(*dims_sRF)
        tRF = U[:, 0]

    elif len(dims) == 2:
        dims_tRF = dims[0]
        dims_sRF = dims[1]
        U, S, Vt = randomized_svd(w.reshape(dims_tRF, dims_sRF), 3)
        sRF = Vt[0]
        tRF = U[:, 0]        
    
    return [sRF, tRF]

def softthreshold(K, lambd):
    # L1 regularization as soft thresholding.
    return np.maximum(K - lambd, 0) - np.maximum(- K - lambd, 0)

def norm(x):
    return x / np.linalg.norm(x)