import numpy as np

from rfest.utils import uvec


def bs(x, df, degree=3):
    """
    
    B-spline basis. Knots placed equally by percentile.
    
    Simplified from `patsy.bs`:
    https://github.com/pydata/patsy/blob/master/patsy/splines.py
    
    """

    from scipy.interpolate import BSpline

    def _get_all_sorted_knots(_x, _df, _degree):
        order = _degree + 1
        n_inner_knots = _df - order
        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1] * 100
        inner_knots = np.percentile(_x, knot_quantiles)
        all_knots = np.hstack(([np.min(_x), np.max(_x)] * order, inner_knots))
        all_knots = np.sort(all_knots)

        return all_knots

    x = np.asarray(x)
    df = np.asarray(df)

    knots = _get_all_sorted_knots(x, df, degree)
    n_bases = len(knots) - (degree + 1)
    coeff = np.eye(n_bases)
    basis = np.vstack([BSpline(knots, coeff[i], degree)(x) for i in range(n_bases)]).T

    return uvec(basis)


def cr(x, df):
    """
    
    Natural cubic regression splines. Knots placed equally by percentile. 
    
    Simplified from `patsy.cr`:
    https://github.com/pydata/patsy/blob/master/patsy/mgcv_cubic_splines.py
    
    """

    def _get_all_sorted_knots(_x, _df):
        n_inner_knots = _df - 2
        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1] * 100
        inner_knots = np.percentile(np.unique(_x), knot_quantiles)
        all_knots = np.concatenate(([np.min(_x), np.max(_x)], inner_knots))
        all_knots = np.unique(all_knots)

        return all_knots

    knots = _get_all_sorted_knots(x, df)

    j = np.maximum(np.searchsorted(knots, x) - 1, 0)
    h = np.mean(knots[1:] - knots[:-1])  # constant

    ajm = (knots[j + 1] - x) / h
    ajp = (x - knots[j]) / h
    cjm = ((knots[j + 1] - x) ** 3 / h - h * (knots[j + 1] - x)) / 6
    cjp = ((x - knots[j]) ** 3 / h - h * (x - knots[j])) / 6

    B = np.sum([np.diag([1, 2, 1][i] * h * np.ones(df - [3, 2, 3][i]),
                        [-1, 0, 1][i]) / [6, 3, 6][i]
                for i in range(3)], 0)
    D = np.sum([[1, -2, 1][i] * np.pad(np.eye(df - 2) / h,
                                       pad_width=((0, 0), (i, 2 - i)), mode='constant')
                for i in range(3)], 0)

    f = np.vstack([np.zeros(df), np.linalg.solve(B, D), np.zeros(df)])
    i = np.eye(df)

    basis = ajm * i[j, :].T + ajp * i[j + 1, :].T + cjm * f[j, :].T + cjp * f[j + 1, :].T

    return uvec(basis.T)


def cc(x, df):
    """

    Cyclic cubic regression splines. Knots placed equally by percentile.

    Simplified from `patsy.cc`:
    https://github.com/pydata/patsy/blob/master/patsy/mgcv_cubic_splines.py

    """

    def _map_cyclic(_x, lbound, ubound):
        _x = np.copy(_x)
        _x[_x > ubound] = lbound + (_x[_x > ubound] - ubound) % (ubound - lbound)
        _x[_x < lbound] = ubound - (lbound - _x[_x < lbound]) % (ubound - lbound)

        return _x

    def _get_all_sorted_knots(_x, _df):
        n_inner_knots = _df - 2

        knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1] * 100
        inner_knots = np.percentile(np.unique(_x), knot_quantiles)

        all_knots = np.concatenate(([np.min(_x), np.max(_x)], inner_knots))
        all_knots = np.unique(all_knots)

        return all_knots

    def _get_cyclic_f(_knots):
        kd = _knots[1:] - _knots[:-1]
        n = _knots.size - 1
        b = np.zeros((n, n))
        d = np.zeros((n, n))

        b[0, 0] = (kd[n - 1] + kd[0]) / 3.
        b[0, n - 1] = kd[n - 1] / 6.
        b[n - 1, 0] = kd[n - 1] / 6.

        d[0, 0] = -1. / kd[0] - 1. / kd[n - 1]
        d[0, n - 1] = 1. / kd[n - 1]
        d[n - 1, 0] = 1. / kd[n - 1]

        for i in range(1, n):
            b[i, i] = (kd[i - 1] + kd[i]) / 3.
            b[i, i - 1] = kd[i - 1] / 6.
            b[i - 1, i] = kd[i - 1] / 6.

            d[i, i] = -1. / kd[i - 1] - 1. / kd[i]
            d[i, i - 1] = 1. / kd[i - 1]
            d[i - 1, i] = 1. / kd[i - 1]

        return np.linalg.solve(b, d)

    knots = _get_all_sorted_knots(x, df)  # length = df

    x = _map_cyclic(x, min(knots), max(knots))
    df -= 1

    j = np.maximum(np.searchsorted(knots, x) - 1, 0)
    h = np.mean(knots[1:] - knots[:-1])  # constant

    ajm = (knots[j + 1] - x) / h
    ajp = (x - knots[j]) / h
    cjm = ((knots[j + 1] - x) ** 3 / h - h * (knots[j + 1] - x)) / 6
    cjp = ((x - knots[j]) ** 3 / h - h * (x - knots[j])) / 6

    f = _get_cyclic_f(knots)

    i = np.eye(df)
    j1 = j + 1
    j1[j1 == df] = 0

    basis = ajm * i[j, :].T + ajp * i[j1, :].T + cjm * f[j, :].T + cjp * f[j1, :].T

    return uvec(basis.T)


def tp(x, df):
    """
    
    Simplified implementation of the truncated Thin Plate (TP) regression spline.
    See Wood, S. (2017) p.216-217

    """

    def eta(r):
        return r ** 2 * np.log(r + 1e-10)

    E = eta(np.abs(x.ravel() - x.ravel().reshape(-1, 1)))
    U, _, _ = np.linalg.svd(E)
    basis = U[:, :int(df)]

    return uvec(basis)


def te(*args):
    """

    Tensor Product smooth. See Wood, S. (2017) p.227-229
    
    Numericially the same as `patsy.te`:
    https://github.com/pydata/patsy/blob/master/patsy/mgcv_cubic_splines.py

    """

    def columnwise_product(A2, A1):
        return np.hstack([A2 * A1[:, i][:, np.newaxis] for i in range(A1.shape[1])])

    As = list(args)

    if len(As) == 1:
        return As[0]

    return columnwise_product(te(*As[:-1]), As[-1])


def build_spline_matrix(dims, df, smooth, dtype=np.float64):
    """
    
    Building spline matrix for n-dimensional RF (n=[1,2,3]) with tensor product smooth.
    
    Parameters
    ==========
    
    dims : list or array_like, shape (d, )
        Dimensions or shape of the RF to estimate. Assumed order [t, sx, sy]
        
    df : list or array_like, shape (d,) 
        Degree of freedom for spline / smooth basis. Same length as dims.
        
    smooth : str
        Spline or smooth to be used. Current supported methods include:
        * `bs`: B-spline
        * `cr`: Cubic Regression spline
        * `tp`: (Simplified) Thin Plate regression spline 

    dtype : dtype
        Data type S will be cast to before returning

    Return
    ======
    
    S : array_like, shape (n_features, n_spline_coef)
        Spline matrix. Each column is one basis. 

    Note
    ====

    ---outdated
    A mesh-free (actually simpler) way to do this is to get the spline bases for each dimension, 
    then calculate the kronecker product of them, for example:

    >>> dims, df = (10, 10, 10), (3, 3, 3)
    >>> St, Sy, Sx = [basis(np.arange(d), f), for d, f in zip(dims, df)]
    >>> S = np.kron(St, np.kron(Sy, Sx)) 
    
    Here we use a mesh-based `te` approach to keep consistent with the Patsy inplementation / Wood, S. (2017).
    ----

    Now we switched to the mesh-free inmplemtation.
        
    """

    ndim = len(dims)  # store RF dimemsion

    # initialize list of degree of freedom for each dimension
    if len(df) != ndim:
        raise ValueError("`df` must have the same length as `dims`")

    if smooth == 'cr':
        basis = cr  # Natural cubic regression spline
    elif smooth == 'cc':
        basis = cc  # cyclic cubic regression spline
    elif smooth == 'bs':
        basis = bs  # b-spline
    elif smooth == 'tp':
        basis = tp  # thin-plate spline
    else:
        raise ValueError("Input method `{}` is not supported.".format(smooth))

    # build spline matrix
    if ndim == 1:

        g0 = np.arange(dims[0])
        S = basis(g0.ravel(), df[0])

    elif ndim == 2:

        g0 = np.arange(dims[0])
        g1 = np.arange(dims[1])

        St = basis(g0.ravel(), df[0])
        Sx = basis(g1.ravel(), df[1])

        S = np.kron(St, Sx)

    elif ndim == 3:

        g0 = np.arange(dims[0])
        g1 = np.arange(dims[1])
        g2 = np.arange(dims[2])

        St = basis(g0.ravel(), df[0])
        Sx = basis(g1.ravel(), df[1])
        Sy = basis(g2.ravel(), df[2])

        S = np.kron(St, np.kron(Sx, Sy))

    else:
        raise NotImplementedError(ndim)

    return uvec(S).astype(dtype)
