import autograd.numpy as np
import scipy

def get_sdm(stim, nlag):

    '''
    Get stimulus design matrix.
    '''
    
    n_samples, *n_features = stim.shape
    n_features = np.product(*n_features)
    S = np.zeros((n_samples, n_features*nlag))
    
    for j in range(n_features):
        row = np.pad([stim[0, j]], pad_width=(0, nlag-1), mode='constant')
        col = stim[:, j]
        S[:, nlag*j:nlag*(j+1)] = np.fliplr(scipy.linalg.toeplitz(col, row))
        
    return S

def get_rdm(resp, nlag):
    
    n_samples = resp.shape[0]
    R = np.zeros((n_samples, nlag))
    resp = np.pad(resp[nlag-1:].ravel(), (0, nlag-1), mode='constant')
    
    for i in range(nlag):
        if i == 0:
            R[:, i] = np.pad(resp[:].ravel(), (i, 0), mode='constant')
        else:
            R[:, i] = np.pad(resp[:-i].ravel(), (i, 0), mode='constant')
    
    return R

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