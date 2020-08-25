import numpy as np
import scipy.signal
import scipy.stats

from .utils import build_design_matrix, uvec


__all__ = ['gaussian1d', 'gaussian2d', 'gaussian2d',
           'mexicanhat1d', 'mexicanhat2d', 'mexicanhat3d',
           'gabor2d', 'gabor3d', 'V1complex_2d',
           'get_stimulus']

def gaussian1d(dim=200, std=15):
    return uvec(scipy.signal.gaussian(dim, std=std))

def gaussian2d(dims=[25,25], std=[2,2]):
    gaussian_x = gaussian1d(dims[0], std=std[0])    
    gaussian_y = gaussian1d(dims[1], std=std[1]) 
    return uvec(np.kron(gaussian_x, gaussian_y)).reshape(dims)

def gaussian3d(dims, std):
    gaussian_t = np.gradient(gaussian1d(dims[0], std[0]))
    gaussian_s = gaussian2d(dims[1:], std[1:]).flatten()
    return uvec(np.kron(gaussian_t, gaussian_s)).reshape(dims)

def mexicanhat1d(dims=200, std=15, a=0.8):
    g0 = gaussian1d(dims, std)
    g1 = gaussian1d(dims, std*a)
    m = g1 - 0.65 * g0
    return uvec(m)

def mexicanhat2d(dims=[25,25], std=[3,3], a=0.55):
    g0 = gaussian2d(dims, std)
    g1 = gaussian2d(dims, np.array(std) * a)
    m = g1 - 0.65 * g0
    return uvec(m)

def mexicanhat3d(dims, std, a=0.3):
    g_t = np.gradient(gaussian1d(dims[0], std[0]))
    m_s = mexicanhat2d(dims[1:], std[1:], a).flatten()
    m = np.kron(g_t, m_s)  
    return uvec(m).reshape(dims)

def gabor2d(dims=[25, 25], omega=0.5, theta=np.pi/6, func=np.cos, K=1.):
    radius = (int(dims[1]/2.0), int(dims[0]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    
    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    
    return uvec(gabor)[:dims[0], :dims[1]]

def gabor3d(dims, std, omega, theta, func=np.cos, K=np.pi):
    g_t = np.gradient(gaussian1d(dims[0], std))
    g_s = gabor2d(dims[1:], omega, theta, func, K).flatten()
    g = np.kron(g_t, g_s)  
    return uvec(g).reshape(dims)    

def V1complex_2d(dims=[30, 40], scale=[.025, .03]):

    dt = 1/60 # time bin size
    nt = dims[0]
    nx = dims[1]
    tt = np.arange(-nt*dt, 0, dt)

    kt1 = scipy.stats.gamma.pdf(-tt, dims[0]/7.5, scale=scale[0])
    kt2 = scipy.stats.gamma.pdf(-tt, dims[1]/6, scale=scale[1])
    kt1 /= np.linalg.uvec(kt1)
    kt2 /= -np.linalg.uvec(kt2)

    kt = np.vstack([kt1, kt2]).T

    xx = np.linspace(-2, 2, nx)

    kx1 = np.cos(2*np.pi*xx/2 + np.pi/5) * np.exp(-1/(2*0.35**2) * xx**2)
    kx2 = np.sin(2*np.pi*xx/2 + np.pi/5) * np.exp(-1/(2*0.35**2) * xx**2)

    kx1 /= np.linalg.uvec(kx1)
    kx2 /= np.linalg.uvec(kx2)

    kx = np.vstack([kx1, kx2])

    k = kt @ kx
    
    return uvec(k)

def get_stimulus(n_samples, dims, kind='3dnoise', delta=1000, random_seed=1990):

    """
    Parameters
    ==========
    n_samples: int
        number of frames
    
    dims: list or array_like
        RF size
    
    delta: float
        size of the gaussian kernel. 
        larger delta means stronger correlation in the stimulus. 
    """

    def kernel(ncoeff, delta):
        grid = np.arange(ncoeff)
        square_distance = np.sqrt((grid - grid.reshape(-1,1))**2) 
        C = np.exp(-square_distance / (ncoeff/delta))
        return C


    np.random.seed(random_seed)

    if len(dims) == 1:
        
        Sigma = kernel(dims[0], delta)
        Stim = np.random.multivariate_normal(np.zeros(len(Sigma)), Sigma, n_samples) 
        X = Stim

    elif len(dims) == 2 and kind=='2dbar':

        Sigma = kernel(dims[1], delta)
        Stim = np.random.multivariate_normal(np.zeros(len(Sigma)), Sigma, n_samples)
        X = build_design_matrix(Stim, dims[0])

    elif len(dims) == 2 and kind=='2dnoise':
        
        Sigma = np.kron(kernel(dims[0], delta), kernel(dims[1], delta))
        Stim = np.random.multivariate_normal(np.zeros(len(Sigma)), Sigma, n_samples) 
        X = Stim

    elif len(dims) == 3 and kind=='3dnoise':
        
        Sigma = np.kron(kernel(dims[1], delta), kernel(dims[2], delta))
        Stim = np.random.multivariate_normal(np.zeros(len(Sigma)), Sigma, n_samples) 
        X = build_design_matrix(Stim, dims[0])
        
    return X
