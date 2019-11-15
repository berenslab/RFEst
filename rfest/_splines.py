import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def bs(x, df, degree=3):
    
    """
    B-spline basis. Simplified from `patsy.bs`.
    """
    
    from scipy.interpolate import BSpline
    import numpy as np
    
    def _get_all_sorted_knots(x, df, degree):

        order = degree + 1
        n_inner_knots = df - order
        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1] * 100
        inner_knots = np.percentile(x, knot_quantiles)
        all_knots = np.hstack(([np.min(x), np.max(x)] * order, inner_knots))
        all_knots = np.sort(all_knots)
        
        return all_knots
    
    x = np.asarray(x)
    df = np.asarray(df)

    knots = _get_all_sorted_knots(x, df, degree)
    n_bases = len(knots) - (degree + 1) 
    coeff = np.eye(n_bases)
    basis = np.vstack([BSpline(knots, coeff[i], degree)(x) for i in range(n_bases)]).T
    
    return basis

def cr(x, df):
    
    """
    Natural cubic regression splines. Simplified from `patsy.cr`.
    """
    
    import numpy as np
    
    def _get_all_sorted_knots(x, df):

        n_inner_knots = df-2

        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1] * 100
        inner_knots = np.percentile(np.unique(x), knot_quantiles)

        all_knots = np.concatenate(([np.min(x), np.max(x)], inner_knots))
        all_knots = np.unique(all_knots)

        return all_knots

    knots = _get_all_sorted_knots(x, df)
    
    j = np.maximum(np.searchsorted(knots, x) - 1, 0)
    h = np.mean(knots[1:] - knots[:-1]) # constant

    ajm = (knots[j+1] - x) / h
    ajp = (x - knots[j]) / h
    cjm = ((knots[j+1] - x)**3 / h - h * (knots[j+1] - x)) / 6
    cjp = ((x - knots[j])  **3 / h - h * (x - knots[j]))   / 6
    
    B = np.sum([np.diag([1, 2, 1][i] * h * np.ones(df-[3, 2, 3][i]), 
                        [-1, 0, 1][i]) / [6, 3, 6][i] 
                    for i in range(3)], 0)
    D = np.sum([[1, -2, 1][i] * np.pad(np.eye(df-2) / h, 
                        pad_width=((0, 0), (i, 2-i)), mode='constant') 
                    for i in range(3)], 0)


    f = np.vstack([np.zeros(df), np.linalg.solve(B, D), np.zeros(df)])

    i = np.eye(df)
    basis = ajm * i[j,:].T + ajp * i[j+1,:].T + cjm * f[j,:].T + cjp * f[j+1,:].T
    
    return basis.T

def tp(x, df):

    """
    
    Simplified implementation of the truncated Thin Plate (TP) regression spline.

    """
    
    def eta(r):
        return r**2 * np.log(r + 1e-10)

    E = eta(np.abs(x.ravel() - x.ravel().reshape(-1,1)))
    U, _, _ = np.linalg.svd(E)
    basis = U[:, :int(df)]
    
    return basis / np.linalg.norm(basis)

def te(*args):

    """

    Tensor Product smooth. Numericially the same as `patsy.te`.

    """
    
    As = list(args)
    
    def columnwise_product(A2, A1):
        return np.hstack([A2 * A1[:, i][:, np.newaxis] for i in range(A1.shape[1])])    

    if len(As)==1:
        return As[0]
    
    return columnwise_product(te(*As[:-1]), As[-1])


def build_spline_matrix(dims, df, smooth):

    ndim = len(dims)
    
    # initialize list of degree of freedom for each dimension
    if np.ndim(df) != 0 and len(df) != ndim:
        raise ValueError("`df` must be an integer or an array the same length as `dims`")
    elif np.ndim(df) == 0:
        df = np.ones(ndim) * df
        df = df.astype(int)

    # choose smooth basis
    if smooth =='bs': 
        basis = bs # B-spline (order=3)
    elif smooth == 'cr':
        basis = cr  # Natural cubic regression spline
    elif smooth == 'tp':
        basis = tp  # Thin plate regression spline
    else:
        raise ValueError("Input method `{}` is not supported.".format(smooth))

    # build spline matrix
    if ndim == 1:

        S = basis(np.arange(dims[0]), df[0])

    elif ndim == 2:

        g0, g1 = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing='ij')
        S = te(basis(g0.ravel(), df[0]), 
               basis(g1.ravel(), df[1]))

    elif ndim == 3:

        g0, g1, g2 = np.meshgrid(np.arange(dims[0]), 
                                 np.arange(dims[1]), 
                                 np.arange(dims[2]), indexing='ij')
        S = te(basis(g0.ravel(), df[0]), 
               basis(g1.ravel(), df[1]), 
               basis(g2.ravel(), df[2]))

    return S
