import jax.numpy as np
from sklearn.decomposition import randomized_svd

# def get_stimulus_design_matrix(stim, nlag):
    
#     import numpy as np
    
#     n_samples = stim.shape[0]
#     n_features = stim.shape[1:]
#     frames = np.zeros([n_samples, np.prod(n_features) * nlag])
    
#     for i in range(n_samples):
        
#         if i < nlag-1:
#             pad = np.zeros([nlag-i-1, *n_features])
#             frame = np.ravel(np.vstack([pad, stim[:i+1]]))
#         else:
#             frame = np.ravel(stim[i-nlag+1:i+1])
        
#         frames[i] = frame
        
#     return frames

def get_stimulus_design_matrix(X, nlag):
    
    n_sample = X.shape[0]
    n_feature = X.shape[1:]
    X = X.reshape(n_sample, np.prod(n_feature))
    X_padded = np.vstack([np.zeros([nlag-1, np.prod(n_feature)]), X])
    X_design = np.hstack([X_padded[i:n_sample+i] for i in range(nlag)])
    
    return X_design

def get_spatial_and_temporal_filters(w, dims):
    
    if len(dims) != 3:
        raise ValueError("Only works for 3D receptive fields.")
        
    dims_tRF = dims[0]
    dims_sRF = dims[1:]
    U, S, Vt = randomized_svd(w.reshape(dims_tRF, np.prod(dims_sRF)), 3)
    sRF = Vt[0].reshape(*dims_sRF)
    tRF = U[:, 0]

    return [sRF, tRF]

def realfftbasis(nx):
    
    nn = nx
    
    ncos = np.ceil((nn + 1) / 2)
    nsin = np.floor((nn-1) / 2)
    
    wvec = np.hstack([np.arange(ncos), np.arange(-nsin, 0)])
    
    wcos = wvec[wvec >= 0]
    wsin = wvec[wvec < 0]
    
    x = np.arange(nx)
    
    t0 = np.cos(np.outer(wcos * 2 * np.pi / nn, x))
    t1 = np.sin(np.outer(wsin * 2 * np.pi / nn, x))
    
    B = np.vstack([t0, t1]) / np.sqrt(nn * 0.5)
    
    return B, wvec
