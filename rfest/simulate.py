import numpy as np
import scipy.signal
import scipy.stats

from .utils import build_design_matrix, uvec
from .nonlinearities import * 

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
    kt1 /= np.linalg.norm(kt1)
    kt2 /= -np.linalg.norm(kt2)

    kt = np.vstack([kt1, kt2]).T

    xx = np.linspace(-2, 2, nx)

    kx1 = np.cos(2*np.pi*xx/2 + np.pi/5) * np.exp(-1/(2*0.35**2) * xx**2)
    kx2 = np.sin(2*np.pi*xx/2 + np.pi/5) * np.exp(-1/(2*0.35**2) * xx**2)

    kx1 /= np.linalg.norm(kx1)
    kx2 /= np.linalg.norm(kx2)

    kx = np.vstack([kx1, kx2])

    k = kt @ kx
    
    return uvec(k)

def get_fullfield_flicker(n_samples, dims=None, shift=0, beta=None, noise='gaussian', design_matrix=False, random_seed=2046):

    np.random.seed(random_seed)

    if noise == 'gaussian':
        
        X = np.random.randn(n_samples)[:, np.newaxis]
        
    elif noise == 'binary':
        
        X = np.random.choice([-1, 1], size=n_samples)[:, np.newaxis]
        
    
    if beta is not None:
        X = colornoise1d(1, dims=n_samples, beta=beta, phi=X.flatten(), random_seed=2046)[:, np.newaxis]
        X = (X - X.mean()) / X.std()
        
    if design_matrix:
        
        if dims is None:
            raise ValueError('`dims` is needed for building stimulus design matrix.')
        X = build_design_matrix(X, dims, shift)
        
    return X

def get_flicker_bar(n_samples, dims, shift=0, beta=None, noise='gaussian', design_matrix=False, random_seed=2046):
    
    nt, nx = dims
    
    if noise == 'gaussian':
        
        X = np.random.randn(n_samples, nx)
        
    elif noise == 'binary': 
        
        X = np.random.choice([-1, 1], size=[n_samples, nx])
        
    if beta is not None:
        if noise != 'gaussian': 
            raise ValueError('1/f noise only applis to Gaussian noise.')
        X = colornoise1d(n_samples=n_samples, dims=nx, beta=beta, phi=X, random_seed=random_seed)
        X = (X - X.mean()) / X.std()        
            
    if design_matrix:
        
        X = build_design_matrix(X, nt, shift)
        
    return X

def get_checkerboard(n_samples, dims, shift=0, beta=None, noise='gaussian', design_matrix=False, random_seed=2046):
    
    if len(dims) == 2:
        nt = None
        
    elif len(dims) == 3:
        nt = dims[0]
        dims = dims[1:]
    
    if noise == 'gaussian':

        X = np.random.randn(n_samples, *dims)
        
    elif noise == 'binary':
        
        X = np.random.choice([-1, 1], size=[n_samples, *dims])
        
    if beta is not None:
        if noise != 'gaussian': 
            raise ValueError('1/f noise only applis to Gaussian noise.')
            
        X = colornoise2d(n_samples=n_samples, dims=dims, beta=beta, phi=X, random_seed=random_seed)
        X = (X - X.mean()) / X.std()
        
    if design_matrix:
        if nt is None:
            pass
        else:
            X = build_design_matrix(X, nt, shift)
            
    return X
        
def colornoise1d(n_samples, dims, beta=1, phi=None, random_seed=2046):
    
    u = np.fft.fftfreq(dims)
    Sf = (u ** 2) ** (- beta / 2)
    Sf[np.isinf(Sf)] = 0
    
    np.random.seed(random_seed)
    phi = np.random.randn(n_samples, dims) if phi is None else phi
    
    f = np.cos(2 * np.pi * phi, dtype=complex) 
    f.imag = np.sin(2 * np.pi * phi)

    x = np.fft.fft(Sf ** 0.5 * f)
    
    return x.real        
    
    
def colornoise2d(n_samples, dims, beta=1, phi=None, random_seed=2046):
        
    u = np.fft.fftfreq(dims[0])[:, np.newaxis]
    u = np.tile(u, dims[1])
    v = np.fft.fftfreq(dims[1])[:, np.newaxis]
    v = np.tile(v.T, (dims[0], 1))

    Sf = (u ** 2 + v ** 2) ** (- beta / 2)
    Sf[np.isinf(Sf)] = 0

    np.random.seed(random_seed)
    phi = np.random.randn(n_samples, *dims) if phi is None else phi
    
    f = np.cos(2 * np.pi * phi, dtype=complex) 
    f.imag = np.sin(2 * np.pi * phi)

    x = np.fft.fft2(Sf ** 0.5 * f)
    
    return x.real

def get_response(X, w, intercept=0, dt=1, R=10, random_seed=2046,
                distr='gaussian', nonlinearity='none'):

    np.random.seed(random_seed)
    if nonlinearity == 'softplus':
        fnl = softplus
    elif nonlinearity == 'exponential':
        fnl = np.exp
    elif nonlinearity == 'none':
        fnl = lambda x : x

    r = dt * R * fnl(X @ w + intercept)

    if distr == 'gaussian':
        return np.random.normal(r)

    elif distr == 'poisson':
        r = np.maximum(dt * r, 1e-17) # avoid 0.
        return np.random.poisson(r)


