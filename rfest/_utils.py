import numpy as np
from sklearn.decomposition import randomized_svd

def build_design_matrix(X, nlag, shift=0, n_c=1):

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
    
    n_c : int
        Number of color channels.

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

    if nlag+shift > 0: 
        X_padded = np.vstack([np.zeros([nlag+shift-1, np.prod(n_feature)]), X])
    else:
        X_padded = X
        
    if shift < 0:
        X_padded = np.vstack([X_padded, np.zeros([-shift, np.prod(n_feature)])])
    
    X_design = np.hstack([X_padded[i:n_sample+i] for i in range(nlag)])
    
    if n_c > 1:
        return X_design.reshape(X_design.shape[0], -1, n_c)
    else:
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

def uvec(x):
    # turn input array into a unit vector
    return x / np.linalg.norm(x)

def fetch_data(data=None, datapath='./data/', overwrite=False):

    import urllib.request
    import os
    try:
        import h5py
    except:
        print("`h5py` is not installed. Please run `pip install h5py`.")

    if data is None:
        
        print('Available datasets: \n')
        print('\t1. A V1 Complex cell from Rust, et al., 2005. (stimulus: flicker bars; source: https://github.com/pillowlab/subunit_mele)')
        print('\t2. Salamander RGCs from Maheswaranathan et. al. 2018 (stimulus: flicker bars; source: https://github.com/baccuslab/inferring-hidden-structure-retinal-circuits)')
        print('\t3. Macaque RGCs from Uzzell & Chichilnisky, 2004 (stimulus: full-field flicker; source: https://github.com/pillowlab/GLMspiketraintutorial)')
        print('\t4. Salamander RGCs from Liu, et al., 2017 (stimulus: checkerboard; source: https://gin.g-node.org/gollischlab/Liu_etal_2017_RGC_spiketrains_for_STNMF)')
        # print('\t5. Mouse RGCs from Ran, et al. 2020 (source: htpps://github.com/berenslab/RFEst)')
        
    else:

        if not os.path.exists(datapath):
            os.makedirs(datapath)

        if data == 1:
                        
            if os.path.exists(datapath + '544l029.p21_stc.mat') is True and overwrite is False:
                print('Data is already downloaded. To re-download the same file, please set `overwrite=False`.')
                
            else:
                if overwrite is True:
                    print('Re-downloading...')
                else:
                    print('Downloading...')
                url = 'https://github.com/pillowlab/subunit_mele/blob/master/neural_data/544l029.p21_stc.mat?raw=true'
                urllib.request.urlretrieve(url, datapath + '544l029.p21_stc.mat')
                print('Done.')
                
            print('Loading data (V1 Complex cell, Rust, et al., 2005.)')
            with h5py.File(datapath + '544l029.p21_stc.mat', 'r') as f:
                data = {key:f[key][:] for key in f.keys() if key != '#refs#'}
            print('Done.')
                
        elif data == 2:
        
            if os.path.exists(datapath + 'rgc_whitenoise.h5') is True and overwrite is False:
                print('Data is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:     
                if overwrite is True:
                    print('Re-downloading...')
                else:
                    print('Downloading...')                
                url = 'https://github.com/baccuslab/inferring-hidden-structure-retinal-circuits/blob/master/rgc_whitenoise.h5?raw=true'
                urllib.request.urlretrieve(url, datapath + 'rgc_whitenoise.h5')
                print('Done.')
            
            print('Loading data (Salamander RGCs from Maheswaranathan et. al. 2018)')
            with h5py.File(datapath + 'rgc_whitenoise.h5', 'r') as f:
                data = {key:f[key][:] for key in f.keys()}
            print('Done.')
            
        elif data == 3:
        
            if os.path.exists(datapath + 'data_RGCs.zip') is True and overwrite is False:
                print('Data is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:     
                if overwrite is True:
                    print('Re-downloading...')
                else:
                    print('Downloading...')                
                url = 'http://pillowlab.princeton.edu/data/data_RGCs.zip'
                urllib.request.urlretrieve(url, datapath + 'data_RGCs.zip')
                print('Done.')

            print('Loading data (Macaque GRCs from Uzzell & Chichilnisky, 2004)')
            
            if not os.path.exists(datapath + 'data_RGCs'):            
                from zipfile import ZipFile
                archive = ZipFile(datapath + 'data_RGCs.zip', 'r')
                archive.extractall(path=datapath)

            import scipy.io
            data = {} 
            stim = scipy.io.loadmat(datapath + 'data_RGCs/Stim.mat')
            data.update({'Stim': stim['Stim'].flatten()})

            stimtime = scipy.io.loadmat(datapath + 'data_RGCs/stimtimes.mat')
            data.update({'stimtimes': stimtime['stimtimes'].flatten()})

            spiketime = scipy.io.loadmat(datapath + 'data_RGCs/SpTimes.mat')
            data.update({'SpTimes': spiketime['SpTimes']})
            print('Done.')

        elif data == 4:

            if os.path.exists(datapath + 'stnmf.zip') is True and overwrite is False:
                print('Data is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:     
                if overwrite is True:
                    print('Re-downloading...')
                else:
                    print('Downloading...')                
                url = 'https://github.com/huangziwei/data_RFEst/blob/master/stnmf.zip?raw=true'
                urllib.request.urlretrieve(url, datapath + 'stnmf.zip')
                print('Done.')

            if not os.path.exists(datapath + 'stnmf'):
                
                from zipfile import ZipFile
                archive = ZipFile(datapath + 'stnmf.zip', 'r')
                archive.extractall(path=datapath)
            
            print('Loading data (Salamander RGCs from Liu, et al., 2017)')
            with h5py.File(datapath + 'stnmf/train.h5', 'r') as f:
                train = {key:f[key][:] for key in f.keys()}

            with h5py.File(datapath + 'stnmf/test.h5', 'r') as f:
                test = {key:f[key][:] for key in f.keys()}

            data = {'train': train, 'test':test}
            print('Done.')
            
        return data
