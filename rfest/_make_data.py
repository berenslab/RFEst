import autograd.numpy as np
import scipy

from ._utils import *

def gaussian(dims, sigma):
    
    if len(dims) == 1:
        
        x = dims[0]
        mu = 0.5 * (x - 1)
        xx = np.arange(x)
        
        gaussian_filter = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(xx-mu)**2/ (2*sigma**2) )
        gaussian_filter = gaussian_filter.reshape(*dims, 1)
        
    elif len(dims) > 1:
        
        x = dims[0]
        y = dims[1]
        
        sigx = sigma[0]
        sigy = sigma[1]
        
        xx, yy = np.meshgrid(np.arange(0, x), np.arange(0, y))
        mu = [0.5 * (x - 1), 0.5 * (y - 1)]
        
        gaussian_window = np.exp(-0.5 * (((xx - mu[0]) / sigx)**2 + ((yy - mu[1]) / sigy)**2))
        gaussian_filter = 0.5 * gaussian_window / (0.5 * np.pi * (sigx + sigy) ** 2)
        gaussian_filter = 0.25 * gaussian_filter / gaussian_filter.max()
        gaussian_filter = gaussian_filter.T
    
    return gaussian_filter

def mexican_hat(dims, sigma):
    
    if len(dims) == 1:
        
        x = dims[0]
        
        gaussian0 = gaussian(dims, sigma)
        gaussian1 = gaussian(dims, sigma * 0.65)
        mexican_hat = gaussian1 - gaussian0
        mexican_hat = mexican_hat.reshape(*dims, 1)

    elif len(dims) > 1:
        
        x = dims[0]
        y = dims[1]
        
        sigx = sigma[0]
        sigy = sigma[1]
        xx, yy = np.meshgrid(np.arange(0, x), np.arange(0, y))
        mu = [0.5 * (x - 1), 0.5 * (y - 1)]

        gaussian_window = gaussian(dims, sigma).T
        mexican_hat = 1 / ( np.pi * sigx ** 2) * (1 - (((xx - mu[0]) / sigy)**2 + ((yy - mu[1]) / sigy)**2)) * gaussian_window
        mexican_hat = mexican_hat.T

    mexican_hat = mexican_hat / np.sqrt(np.sum(mexican_hat**2))
    
    return mexican_hat
        
    
def gabor(dims, sigma, theta=np.pi/4, phi=np.pi/2, sf=1/6):
    
    if len(dims) == 1:
        
        x = dims[0]
        xx = np.arange(x)
        guassian_window = gaussian(dims, sigma)
        sinusoidal_wave = np.sin(2 * np.pi * sf * xx  - phi)
        gabor_filter = guassian_window * sinusoidal_wave
        gabor_filter = gabor_filter.reshape(*dims, 1)

    elif len(dims) > 1:
        
        x = dims[0]
        y = dims[1]
        
        sigx = sigma[0]
        sigy = sigma[1]
        xx, yy = np.meshgrid(np.arange(0, x), np.arange(0, y))
        mu = [0.5 * (x - 1), 0.5 * (y - 1)]
        
        gaussian_window = gaussian(dims, sigma).T
        sinusoidal_wave = np.sin(2 * np.pi * sf * yy  - phi)
        gabor_filter = (gaussian_window * sinusoidal_wave).T
        
    gabor_filter = gabor_filter / np.sqrt(np.sum(gabor_filter**2))
        
    return gabor_filter

def make_true_filter(dims, sigma, filter_type='gaussian'):

    if filter_type == 'gaussian':
        w_true = gaussian(dims, sigma)
    elif filter_type == 'mexican_hat':
        w_true = mexican_hat(dims, sigma)
    elif filter_type == 'gabor':
        w_true = gabor(dims, sigma)
    else:
        print('Please choose from "gaussian", "mexican_hat", "gabor".')
        
    if len(dims) ==3:
        
        nT = dims[2]
        temporal_filter = np.gradient(gaussian((nT*10,), 6).flatten())
        temporal_filter = temporal_filter / np.linalg.norm(temporal_filter)
        spatial_filter = w_true.flatten()
        
        w_true = np.array([spatial_filter * temporal_filter[i] for i in range(nT*10)]).T
        w_true = w_true[:, ::10]
        w_true = w_true.reshape(*dims)

    return w_true


def make_stimulus(n_samples, dims, nsevar, seed):
    np.random.seed(seed)
    if len(dims) == 1:
        
        stimulus = np.random.randn(*dims, n_samples).reshape(n_samples, *dims)

    elif len(dims) > 1:

        stimulus = np.random.randn(n_samples, np.product([*dims[:2]]))

    return stimulus


def make_response(stimulus, w_true, nsevar):

    if len(w_true.shape) == 1:
        response = stimulus @ w_true + np.random.randn(stimulus.shape[0], 1) * nsevar
    elif len(w_true.shape) == 2:
        response = stimulus @ w_true.reshape(np.product([*w_true.shape]), 1) + np.random.randn(stimulus.shape[0], 1) * nsevar
    elif len(w_true.shape) == 3:

        nlag = w_true.shape[2]

        stimulus = get_sdm(stimulus, nlag)
        w_true = w_true.reshape(np.product([*w_true.shape]), 1)
        response = stimulus @ w_true + np.random.randn(stimulus.shape[0], 1) * nsevar

    return response

def make_data(dims, sigma, n_samples, nsevar, preloaded_stimulus=None, filter_type='gaussian', seed=2046):
    
    """
    a simple linear gaussian data generator.

    Paramters
    =========
    dims: array-like
        dimension of the receptive field, e.g.
            1D: (nX,)
            2D: (nX, nY)
            3D: (nX, nY, nT)

    sigma: array-like
        the deviation of the spatial receptive field, e.g.
            1D: (sig,)
            2D/3D: (sigx, sigy)

    n_samples: int
        number of samples/length of stimulus

    nsevar: float
        the noisiness of the response

    preloaded_stimulus: None or array-like
        You can load preexisiting stimulus instead of generating new one

    filter_type: str
        There are three types of filter to be choose:
        'gaussian', 'mexican_hat', 'gabor'

    seed: int
        Seed the random number generator.
    
    Returns
    =======
    a tuple of 
     ((Stimulus_training, Reponse_training),
      (Stimulus_testing, Response_testing),
      true_receptive_field,
      )

    """

    w_true = make_true_filter(dims, sigma, filter_type)

    if preloaded_stimulus is None: 
        stimulus = make_stimulus(n_samples, dims, nsevar, seed)
    else:
        stimulus = preloaded_stimulus
    response = make_response(stimulus, w_true, nsevar)

    n_test = int(n_samples * 0.25)
    n_train = int(n_samples - n_test)

    return ((stimulus[:n_train, :], response[:n_train]),
            (stimulus[n_train:], response[:n_train]),
            w_true)


