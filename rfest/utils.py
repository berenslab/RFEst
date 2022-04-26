import numpy as np
from sklearn.decomposition import randomized_svd


def build_design_matrix(X, nlag, shift=0, n_c=1, dtype=np.float64):
    """

    Build design matrix. 

    Parameters
    ==========

    X : np.ndarray_like, shape (n_samples, 1 or n_pixels_per_frame)
        
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

    dtype : type
        Datatype of design matrix

    Return
    ======
    
    X_design: array_like, shape (n_samples, n_features)

    
    Examples
    ========

    >>> stim = np.array([0, 1, 2])[:, np.newaxis]
    >>> np.allclose(build_design_matrix(stim, 2), np.array([[0, 0], [0, 1], [1, 2]]))
    True

    """

    n_sample = X.shape[0]
    n_feature = X.shape[1:]
    X = X.reshape(n_sample, np.prod(n_feature))

    if nlag + shift > 0:
        X_padded = np.vstack([np.zeros([nlag + shift - 1, np.prod(n_feature)]), X])
    else:
        X_padded = X

    if shift < 0:
        X_padded = np.vstack([X_padded, np.zeros([-shift, np.prod(n_feature)])])

    X_design = np.hstack([X_padded[i:n_sample + i] for i in range(nlag)])

    if n_c > 1:
        return X_design.reshape((X_design.shape[0], -1, n_c)).astype(dtype)
    else:
        return X_design.astype(dtype)


def get_spatial_and_temporal_filters(w, dims):
    """
    
    Asumming a RF is time-space separable, 
    get spatial and temporal filters using SVD. 

    Paramters
    =========

    w : np.ndarray_like, shape (nt, nx, ny) or (nt, nx * ny)

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


def uvec_rows(A):
    # turn each row of a matrix into a unit vector
    return A / np.maximum(np.linalg.norm(A, axis=0, ord=2, keepdims=True), 1e-8)


def get_n_samples(t, dt):
    # get number of samples based on time (in minute) and frame rates (in second).
    return np.round(t * 60 / dt).astype(int)


def get_length(n, dt):
    # get length in minutes based on number of samples and frame rates (in second).
    return np.round(n * dt / 60, 2)


def split_data(X, y, dt, frac_train=0.8, frac_dev=0.1, verbose=1):
    """
    Split data into training, development and test set.

    Parameters
    ==========
    X : np.ndarray_like, shape (n_samples, n_features)
        Stimulus desgin matrix.

    y : np.ndarray_like, shape (n_samples, )
        Response

    dt : float
        Stimulus frame rate in second.

    len_train / len_dev : float
        length of training / dev set in fraction. 
        Test set will be the rest of n_samples.

    """

    if not X.shape[0] == y.shape[0]:
        raise ValueError('X and y must be of same length.')

    if frac_train + frac_dev > 1:
        raise ValueError('`frac_train` + `frac_dev` must be < 1.')

    n_samples = X.shape[0]
    frac_test = np.round(np.maximum(1 - frac_train - frac_dev, 0), 2)

    len_total = get_length(n_samples, dt)
    len_train = get_length(n_samples * frac_train, dt)
    len_dev = get_length(n_samples * frac_dev, dt)
    len_test = np.maximum(get_length(n_samples * frac_test, dt), 0)

    n_train = get_n_samples(len_train, dt)
    n_dev = get_n_samples(len_dev, dt)
    n_test = np.maximum(n_samples - n_train - n_dev, 0)

    if verbose:

        len_type_list = ['Total', 'Train', 'Dev', 'Test']
        data_list = [(n_samples, len_total, 1.), (n_train, len_train, frac_train),
                     (n_dev, len_dev, frac_dev), (n_test, len_test, frac_test)]
        row_format = "{:<5} {:>10} {:>10} {:>10}"
        print("SUMMARY")
        print(row_format.format('', *['N', 'Minutes', 'Fraction']))
        for len_type, row in zip(len_type_list, data_list):
            print(row_format.format(len_type, *row))

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_dev = X[n_train:n_train + n_dev]
    y_dev = y[n_train:n_train + n_dev]

    X_test = X[n_train + n_dev:]
    y_test = y[n_train + n_dev:]

    return ((X_train, y_train),
            (X_dev, y_dev),
            (X_test, y_test))


def fetch_data(data=None, datapath='./data/', overwrite=False):
    import urllib.request
    import os
    try:
        import h5py
    except:
        print("`h5py` is not installed. Please run `pip install h5py`.")

    if data is None:

        print('Available datasets: \n')
        print(
            '\t1. A V1 Complex cell from Rust, et al., 2005. (stimulus: flicker bars; source: https://github.com/pillowlab/subunit_mele)')
        print(
            '\t2. Salamander RGCs from Maheswaranathan et. al. 2018 (stimulus: flicker bars; source: https://github.com/baccuslab/inferring-hidden-structure-retinal-circuits)')
        print(
            '\t3. Macaque RGCs from Uzzell & Chichilnisky, 2004 (stimulus: full-field flicker; source: https://github.com/pillowlab/GLMspiketraintutorial)')
        print(
            '\t4. Tiger Salamander RGCs from Liu, et al., 2017 (stimulus: checkerboard; source: https://gin.g-node.org/gollischlab/Liu_etal_2017_RGC_spiketrains_for_STNMF)')
        print(
            '\t5. Mouse RGCs from Ran, et al. 2020 (stimulus: checkerboard; source: https://github.com/huangziwei/data_RFEst)')

    else:

        if not os.path.exists(datapath):
            os.makedirs(datapath)

        if data == 1:

            if os.path.exists(datapath + '544l029.p21_stc.mat') is True and overwrite is False:
                print(
                    '(Rust, et al., 2005) is already downloaded. To re-download the same file, please set `overwrite=False`.')

            else:
                if overwrite is True:
                    print('Re-downloading (Rust, et al., 2005)...')
                else:
                    print('Downloading (Rust, et al., 2005)...')
                url = 'https://github.com/pillowlab/subunit_mele/blob/master/neural_data/544l029.p21_stc.mat?raw=true'
                urllib.request.urlretrieve(url, datapath + '544l029.p21_stc.mat')
                print('Done.')

            print('Loading data...')
            with h5py.File(datapath + '544l029.p21_stc.mat', 'r') as f:
                data = {key: f[key][:] for key in f.keys() if key != '#refs#'}
            print('Done.')

        elif data == 2:
            import pickle

            if os.path.exists(datapath + 'mahesaranathan.pickle') is True and overwrite is False:
                print(
                    '(Maheswaranathan et. al. 2018) is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:
                if overwrite is True:
                    print('Re-downloading (Maheswaranathan et. al. 2018)...')
                else:
                    print(
                        'Downloading Subset of (Maheswaranathan et. al. 2018)...\nFor the complete dataset, see https://github.com/baccuslab/inferring-hidden-structure-retinal-circuits/')
                url = 'https://github.com/huangziwei/data_RFEst/blob/master/mahesaranathan.pickle?raw=true'
                urllib.request.urlretrieve(url, datapath + 'mahesaranathan.pickle')
                print('Done.')

            print('Loading data...')
            with open(datapath + 'mahesaranathan.pickle', 'rb') as f:
                data = pickle.load(f)
            print('Done.')

        elif data == 3:

            if os.path.exists(datapath + 'data_RGCs.zip') is True and overwrite is False:
                print(
                    '(Uzzell & Chichilnisky, 2004) is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:
                if overwrite is True:
                    print('Re-downloading (Uzzell & Chichilnisky, 2004)...')
                else:
                    print('Downloading (Uzzell & Chichilnisky, 2004)...')
                url = 'http://pillowlab.princeton.edu/data/data_RGCs.zip'
                urllib.request.urlretrieve(url, datapath + 'data_RGCs.zip')
                print('Done.')

            print('Loading data...')

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
                print(
                    '(Liu, et al., 2017) is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:
                if overwrite is True:
                    print('Re-downloading (Liu, et al., 2017)...')
                else:
                    print('Downloading (Liu, et al., 2017)...')
                url = 'https://github.com/huangziwei/data_RFEst/blob/master/stnmf.zip?raw=true'
                urllib.request.urlretrieve(url, datapath + 'stnmf.zip')
                print('Done.')

            if not os.path.exists(datapath + 'stnmf'):
                from zipfile import ZipFile
                archive = ZipFile(datapath + 'stnmf.zip', 'r')
                archive.extractall(path=datapath)

            print('Loading data...')
            with h5py.File(datapath + 'stnmf/train.h5', 'r') as f:
                train = {key: f[key][:] for key in f.keys()}

            with h5py.File(datapath + 'stnmf/test.h5', 'r') as f:
                test = {key: f[key][:] for key in f.keys()}

            data = {'train': train, 'test': test}
            print('Done.')

        elif data == 5:
            import pickle

            if os.path.exists(datapath + 'rgc_dendrites.pickle') is True and overwrite is False:
                print(
                    '(Ran, et al. 2020) is already downloaded. To re-download the same file, please set `overwrite=False`.')
            else:
                if overwrite is True:
                    print('Re-downloading (Ran, et al. 2020)...')
                else:
                    print('Downloading (Ran, et al. 2020)...')
                url = 'https://github.com/huangziwei/data_RFEst/blob/master/rgc_dendrites.pickle?raw=true'
                urllib.request.urlretrieve(url, datapath + 'rgc_dendrites.pickle')
                print('Done.')

            print('Loading data (Ran, et al. 2020)...')
            with open(datapath + 'rgc_dendrites.pickle', 'rb') as f:
                data = pickle.load(f)
            print('Done.')

        return data


def znorm(x):
    return (x - x.mean()) / x.std()


def upsample_data(stim, stimtime, trace, tracetime, gradient=False, threshold=False):
    """
    Upsampling the stimulus into the calcium trace sampling rate.

    Parameters
    ==========
    stim : ndarray
        Stimulus

    stimtime : ndarray
        Trigger time of the stimulus

    trace : ndarray
        Trace of the response

    tracetime : ndarray
        Time of each recorded point trace

    gradient : True
        Apply gradient to data

    threshold: bool
        Clip y at 0

    Return
    ======
    X : array
        Upsampled Stimulus

    y : array
        Trace that overlaps with stimulus, potentially the gradient and clipped.

    dt : float
        timebin size.
    """

    valid_duration = np.logical_and(tracetime > stimtime[0], tracetime < stimtime[-1])
    trace_valid = trace[valid_duration]

    y = znorm(trace_valid.copy())

    if gradient:
        y = np.gradient(y)

    frames = np.vstack([stimtime[:-1], stimtime[1:]]).T

    num_repeats = np.array([np.sum((tracetime > frame[0]) & (tracetime <= frame[1])) for i, frame in enumerate(frames)])

    X = np.repeat(stim[:len(frames)], num_repeats, axis=0)

    cut = np.min([X.shape[0], y.shape[0]])

    dt = np.mean(np.diff(tracetime))

    if threshold:
        y = np.maximum(y, 0)

    return X[:cut], y[:cut], dt


def downsample_data(stim, stimtime, trace, tracetime, gradient=False, threshold=False):
    """
    Downsampling the calcium trace to the stimulus refresh rate.

    Parameters
    ==========
    stim : ndarray
        Stimulus

    stimtime : ndarray
        Trigger time of the stimulus

    trace : ndarray
        Trace of the response

    tracetime : ndarray
        Time of each recorded point of the calcium trace.

    gradient : True
        Apply gradient to data

    threshold: bool
        Clip y at 0

    Return
    ======
    X : array
        Stimulus

    y : array
        Downsampled gradient of the calcium trace.

    dt : float
        timebin size.
    """

    from scipy.interpolate import interp1d

    y = interp1d(
        tracetime.flatten(), znorm(trace.flatten()), kind='linear', fill_value='extrapolate')(stimtime)

    if gradient:
        y = np.gradient(y)

    X = stim
    y = znorm(y)

    cut = np.min([X.shape[0], y.shape[0]])

    dt = np.mean(np.diff(stimtime))

    if threshold:
        y = np.maximum(y, 0)

    return X[:cut], y[:cut], dt


def resample_spikes(stim, stimtime, spiketime):
    """
    Resampling spikes to the stimulus refresh rate
    """

    stimtime_new = np.linspace(stimtime[0], stimtime[-1], stim.shape[0])
    y, _ = np.histogram(spiketime, bins=np.hstack([stimtime_new, stimtime_new[-1] + np.mean(np.diff(stimtime_new))]))

    dt = np.mean(np.diff(stimtime_new))

    X = stim
    cut = np.min([X.shape[0], y.shape[0]])

    return X[:cut], y[:cut], dt
