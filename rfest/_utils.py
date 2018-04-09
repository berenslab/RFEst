import autograd.numpy as np
import scipy

def get_sdm(stim, nlag):
    
    n_samples, *n_features = stim.shape
    n_features = np.product(*n_features)
    S = np.zeros((n_samples, n_features*nlag))
    
    sz = stim.shape
    n2 = np.product(*sz[1:])
    
    S = np.zeros((sz[0], n2*nlag))
    for j in range(n_features):
        row = np.pad([stim[0, j]], pad_width=(0, nlag-1), mode='constant')
        col = stim[:, j]
        S[:, nlag*j:nlag*(j+1)] = np.fliplr(scipy.linalg.toeplitz(col, row))
        
    return S

def lag_weights(weights,nLag):
    lagW = np.zeros([weights.shape[0],nLag])
    
    for iLag in range(nLag):
        start = iLag
        end = -nLag+iLag+1
        if end != 0:
            lagW[start:end,iLag] = weights[nLag-1:]
        else:
            lagW[start:,iLag] = weights[nLag-1:]
        
    return lagW