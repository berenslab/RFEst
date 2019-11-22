import numpy as onp
import scipy

from ._utils import *

def gaussian(dims, sigma):
    
    if len(dims) == 1:
        
        x = dims[0]
        mu = 0.5 * (x - 1)
        xx = onp.arange(x)
        
        gaussian_filter = 1/(sigma*onp.sqrt(2*onp.pi))*onp.exp(-1*(xx-mu)**2/ (2*sigma**2) )
        gaussian_filter = gaussian_filter.reshape(*dims, 1)
        
    else:
        
        if 1 < len(dims) < 3:
        
            x = dims[0]
            y = dims[1]
            
        elif len(dims) > 2:
            
            y = dims[1]
            x = dims[2]
        
        sigx = sigma[0]
        sigy = sigma[1]
        
        xx, yy = onp.meshgrid(onp.arange(0, y), onp.arange(0, x))
        mu = [0.5 * (y - 1), 0.5 * (x - 1)]
        
        gaussian_window = onp.exp(-0.5 * (((xx - mu[0]) / sigx)**2 + ((yy - mu[1]) / sigy)**2))
        gaussian_filter = 0.5 * gaussian_window / (0.5 * onp.pi * (sigx + sigy) ** 2)
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

    else:

        if 1 < len(dims) < 3:
        
            x = dims[0]
            y = dims[1]
            
        elif len(dims) > 2:
            
            y = dims[1]
            x = dims[2]
        
        sigx = sigma[0]
        sigy = sigma[1]
        xx, yy = onp.meshgrid(onp.arange(0, y), onp.arange(0, x))
        mu = [0.5 * (y - 1), 0.5 * (x - 1)]

        gaussian_window = gaussian(dims, sigma).T
        mexican_hat = 1 / ( onp.pi * sigx ** 2) * (1 - (((xx - mu[0]) / sigy)**2 + ((yy - mu[1]) / sigy)**2)) * gaussian_window
        mexican_hat = mexican_hat.T

    mexican_hat = mexican_hat / onp.sqrt(onp.sum(mexican_hat**2))
    
    return mexican_hat
        
    
def gabor(dims, sigma, theta=onp.pi/4, phi=onp.pi/2, sf=1/6):
    
    if len(dims) == 1:
        
        x = dims[0]
        xx = onp.arange(x)
        guassian_window = gaussian(dims, sigma)
        sinusoidal_wave = onp.sin(2 * onp.pi * sf * xx  - phi)
        gabor_filter = guassian_window * sinusoidal_wave
        gabor_filter = gabor_filter.reshape(*dims, 1)

    elif len(dims) > 1:
        
        x = dims[0]
        y = dims[1]
        
        sigx = sigma[0]
        sigy = sigma[1]
        xx, yy = onp.meshgrid(onp.arange(0, x), onp.arange(0, y))
        mu = [0.5 * (x - 1), 0.5 * (y - 1)]
        
        gaussian_window = gaussian(dims, sigma).T
        sinusoidal_wave = onp.sin(2 * onp.pi * sf * yy  - phi)
        gabor_filter = (gaussian_window * sinusoidal_wave).T
        
    gabor_filter = gabor_filter / onp.sqrt(onp.sum(gabor_filter**2))
        
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
        
        nT = dims[0]
        tRF = onp.gradient(gaussian((nT*10,), 6).ravel())[::]
        tRF = tRF / onp.linalg.norm(tRF)
        tRF = tRF[::10]
        sRF = w_true.ravel()
        
        w_true = onp.kron(tRF, sRF.ravel())

    return w_true.reshape(dims)


def make_stimulus(n_samples, dims, nsevar, seed):
    onp.random.seed(seed)
    if len(dims) == 1:
        
        stimulus = onp.random.randn(*dims, n_samples).reshape(n_samples, *dims)

    elif len(dims) > 1:

        stimulus = onp.random.randn(n_samples, onp.product([*dims[1:]]))

    return stimulus


def make_response(stimulus, w_true, nsevar):

    if len(w_true.shape) == 1:
        response = stimulus @ w_true.ravel() + onp.random.randn(stimulus.shape[0], ) * nsevar
    elif len(w_true.shape) == 2:
        response = stimulus @ w_true.ravel() + onp.random.randn(stimulus.shape[0], ) * nsevar
    elif len(w_true.shape) == 3:
        nlag = w_true.shape[0]
        stimulus = get_sdm(stimulus, nlag)
        response = stimulus @ w_true.ravel() + onp.random.randn(stimulus.shape[0], ) * nsevar

    return response

def make_data(dims, sigma, n_samples, nsevar, preloaded_stimulus=None, filter_type='gaussian', seed=2046):
    
    """
    a simple linear gaussian data generator.

    Paramters
    =========
    dims: array-like
        dimension of the receptive field, e.g.
            1D: (nT,)
            2D: (nT, nY)
            3D: (nT, nY, nX)

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
            (stimulus[n_train:], response[n_train:]),
            w_true)


