import jax.numpy as np
from jax import config, grad, jit

try:
    from jax.example_libraries import optimizers
except ImportError:
    from jax.experimental import optimizers

config.update("jax_enable_x64", True)

__all__ = ["fASD"]


class fASD:
    """

    Fast Automatic Smoothness Determination (fASD)

    Reference:  Aoi & Pillow, bioRxiv 2017.

    """

    def __init__(self, X, y, dims, p0, compute_mle=False):

        self.optimized_C_prior = None
        self.optimized_C_post = None
        self.optimized_params = None
        self.w_opt = None
        self.num_iters = None
        self.dims = dims
        self.n_samples, self.n_features = X.shape

        B, freq, freq_each_dim = fourier_transform(dims, p0=p0)
        Z = X @ B

        self.XtY = X.T @ y
        if np.array_equal(y, y.astype(bool)):  # if y is spikes
            self.w_sta = self.XtY / sum(y)
        else:  # if y is not spike
            self.w_sta = self.XtY / len(y)

        if compute_mle:  # maximum likelihood estimation
            self.XtX = X.T @ X
            self.w_mle = np.linalg.solve(self.XtX, self.XtY)

        self.X = np.array(X)
        self.y = np.array(y)

        self.n_b = B.shape[1]
        self.B = B
        self.freq = freq
        self.freq_each_dim = freq_each_dim
        self.Z = Z
        self.ZtZ = Z.T @ Z
        self.ZtY = Z.T @ y
        self.YtY = y.T @ y

        self.p0 = p0

    def update_C_prior(self, params):
        """

        Using kronecker product to construct high-dimensional prior covariance.
        Given RF dims = [t, y, x], the prior covariance:
            C = kron(Ct, kron(Cy, Cx))
            Cinv = kron(Ctinv, kron(Cyinv, Cxinv))

        """

        rho = params[1]
        params_time = params[2]

        # Covariance Matrix in Time
        C_t, C_t_inv = asdf_cov(params_time, self.dims[0], self.freq_each_dim[0])

        if len(self.dims) == 1:

            C, C_inv = rho * C_t, (1 / rho) * C_t_inv

        elif len(self.dims) == 2:

            # Covariance Matrix in Space
            params_space = params[3]
            C_s, C_s_inv = asdf_cov(params_space, self.dims[1], self.freq_each_dim[1])

            # Build 2D Covariance Matrix
            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)

        elif len(self.dims) == 3:

            # Covariance Matrix in Space
            params_spacey = params[3]
            params_spacex = params[4]

            C_sy, C_sy_inv = asdf_cov(
                params_spacey, self.dims[1], self.freq_each_dim[1]
            )
            C_sx, C_sx_inv = asdf_cov(
                params_spacex, self.dims[2], self.freq_each_dim[2]
            )

            C_s = np.kron(C_sy, C_sx)
            C_s_inv = np.kron(C_sy_inv, C_sx_inv)

            # Build 3D Covariance Matrix
            C = rho * np.kron(C_t, C_s)
            C_inv = (1 / rho) * np.kron(C_t_inv, C_s_inv)
        else:
            raise NotImplementedError(len(self.dims))

        return np.diag(C), np.diag(C_inv)

    def update_C_posterior(self, params, C_prior_inv):
        """
        See eq(9) in Park & Pillow (2011).
        """

        sigma = params[0]

        C_post_inv = self.ZtZ / sigma**2 + C_prior_inv
        C_post = np.linalg.inv(C_post_inv)

        m_post = C_post @ self.ZtY / (sigma**2)

        return C_post, C_post_inv, m_post

    def negative_log_evidence(self, params):
        """

        See eq(10) in Park & Pillow (2011).
        """

        sigma = params[0]

        (C_prior, C_prior_inv) = self.update_C_prior(params)

        (C_post, C_post_inv, m_post) = self.update_C_posterior(params, C_prior_inv)

        t0 = np.log(np.abs(2 * np.pi * sigma**2)) * self.n_samples
        t1 = np.linalg.slogdet(C_prior @ C_post_inv)[1]
        t2 = -m_post.T @ C_post @ m_post
        t3 = self.YtY / sigma**2

        return 0.5 * (t0 + t1 + t2 + t3)

    @staticmethod
    def print_progress_header(params):

        if len(params) == 3:
            print("Iter\tσ\tρ\tδt\tcost")
        elif len(params) == 4:
            print("Iter\tσ\tρ\tδt\tδs\tcost")
        elif len(params) == 5:
            print("Iter\tσ\tρ\tδt\tδy\tδx\tcost")

    @staticmethod
    def print_progress(i, params, cost):
        if len(params) == 3:
            print(
                "{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}".format(
                    i, params[0], params[1], params[2], cost
                )
            )
        elif len(params) == 4:
            print(
                "{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}".format(
                    i, params[0], params[1], params[2], params[3], cost
                )
            )
        elif len(params) == 5:
            print(
                "{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}".format(
                    i, params[0], params[1], params[2], params[3], params[4], cost
                )
            )

    def optimize_params(self, p0, num_iters, step_size, tolerance, verbose, atol=1e-5):
        """

        Perform gradient descent using JAX optimizers.
        """

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(p0)

        @jit
        def step(_i, _opt_state):
            p = get_params(_opt_state)
            g = grad(self.negative_log_evidence)(p)
            return opt_update(_i, g, _opt_state)

        cost_list = []
        params_list = []

        if verbose:
            self.print_progress_header(p0)

        for i in range(num_iters):

            opt_state = step(i, opt_state)
            params_list.append(get_params(opt_state))
            cost_list.append(self.negative_log_evidence(params_list[-1]))

            if verbose:
                if i % verbose == 0:
                    self.print_progress(i, params_list[-1], cost_list[-1])

            if len(params_list) > tolerance:

                if np.all((np.array(cost_list[1:])) - np.array(cost_list[:-1]) > 0):
                    params = params_list[0]
                    if verbose:
                        print(
                            "Stop: cost has been monotonically increasing for {} steps.".format(
                                tolerance
                            )
                        )
                    break
                elif np.all(np.array(cost_list[:-1]) - np.array(cost_list[1:]) < atol):
                    params = params_list[-1]
                    if verbose:
                        print(
                            "Stop: cost has been stop changing for {} steps.".format(
                                tolerance
                            )
                        )
                    break
                else:
                    params_list.pop(0)
                    cost_list.pop(0)

        else:

            params = params_list[-1]
            if verbose:
                print(
                    "Stop: reached {0} steps, final cost={1:.5f}.".format(
                        num_iters, cost_list[-1]
                    )
                )

        return params

    def fit(self, num_iters=20, step_size=1e-2, tolerance=10, verbose=True):
        """
        Parameters
        ==========
        num_iters : int
            Max number of optimization iterations.

        step_size : float
            Initial step size for Jax optimizer.

        tolerance : int
            Set early stop tolerance. Optimization stops when cost monotonically
            increases or stop increases for tolerance=n steps.

        verbose: int
            When `verbose=0`, progress is not printed. When `verbose=n`,
            progress will be printed in every n steps.
        """

        self.num_iters = num_iters
        self.optimized_params = self.optimize_params(
            self.p0, num_iters, step_size, tolerance, verbose
        )

        (optimized_C_prior, optimized_C_prior_inv) = self.update_C_prior(
            self.optimized_params
        )

        (optimized_C_post, optimized_C_post_inv, optimized_m_post) = (
            self.update_C_posterior(self.optimized_params, optimized_C_prior_inv)
        )

        self.optimized_C_prior = optimized_C_prior
        self.optimized_C_post = optimized_C_post
        self.w_opt = self.B @ optimized_m_post


def asdf_cov(delta, ncoeff, freq, ext=1.25):
    """
    1D fourier transformed smooth prior.
    """

    ncoeff_ext = np.floor(ncoeff * ext).astype(int)

    const = (2 * np.pi / ncoeff_ext) ** 2
    freq *= const
    C_prior = np.sqrt(2 * np.pi) * np.exp(-0.5 * delta * freq**2)
    C_prior_inv = 1 / (C_prior + 1e-7)

    return C_prior, C_prior_inv


def fourierfreq(ncoeff, delta, CONDTHRESH=1e8):
    maxfreq = np.floor(
        ncoeff / (np.pi * delta) * np.sqrt(0.5 * np.log(CONDTHRESH))
    ).astype(int)
    # wvec = np.hstack([np.arange(maxfreq+1), np.arange(-maxfreq+1, 0)])
    if maxfreq < ncoeff / 2:
        wvec = np.hstack([np.arange(maxfreq + 1), np.arange(-maxfreq + 1, 0)])
    else:
        ncos = np.ceil((ncoeff + 1) / 2)
        nsin = np.floor((ncoeff - 1) / 2)
        wvec = np.hstack([np.arange(ncos), np.arange(-nsin, 0)])

    return wvec


def realfftbasis(ncoeff, ncoeff_circular=None, wvec=None):
    if ncoeff_circular is None:
        ncoeff_circular = ncoeff

    if wvec is None:
        ncos = np.ceil((ncoeff_circular + 1) / 2)
        nsin = np.floor((ncoeff_circular - 1) / 2)
        wvec = np.hstack([np.arange(ncos), np.arange(-nsin, 0)])

    wcos = wvec[wvec >= 0]
    wsin = wvec[wvec < 0]

    x = np.arange(ncoeff)

    t0 = np.cos(np.outer(wcos * 2 * np.pi / ncoeff_circular, x))
    t1 = np.sin(np.outer(wsin * 2 * np.pi / ncoeff_circular, x))

    B = np.vstack([t0, t1]) / np.sqrt(ncoeff_circular * 0.5)

    return B, wvec


def fourier_transform(dims, p0, ext=1.25):
    dims_tRF = dims[0]
    dims_sRF = dims[1:]

    params_time = p0[2]
    params_space = p0[3:]

    ncoeff_ext_t = np.floor(dims_tRF * ext).astype(int)

    wvec_t = fourierfreq(ncoeff_ext_t, params_time)
    U_t, freq_t = realfftbasis(dims_tRF, ncoeff_circular=ncoeff_ext_t, wvec=wvec_t)
    freq_t *= (2 * np.pi / ncoeff_ext_t) ** 2

    freq_each_dim = [freq_t]

    if len(dims_sRF) == 0:
        Uf = U_t
        freq_comb = (2 * np.pi / ncoeff_ext_t**2) * freq_t**2

    else:
        if len(dims_sRF) == 1:

            ncoeff_ext_s = np.floor(dims_sRF[0] * ext).astype(int)
            wvec_s = fourierfreq(ncoeff_ext_s, params_space[0])
            U_s, freq_s = realfftbasis(
                dims_sRF[0], ncoeff_circular=ncoeff_ext_s, wvec=wvec_s
            )
            freq_s *= (2 * np.pi / ncoeff_ext_s) ** 2

            [ww0, ww1] = np.meshgrid(freq_t, freq_s)
            ww0 = np.transpose(ww0, [1, 0])
            ww1 = np.transpose(ww1, [1, 0])

            freq_comb = np.vstack([ww0.flatten(), ww1.flatten()]).T

            Uf = np.kron(U_t, U_s)

            freq_each_dim.append(freq_s)

        elif len(dims_sRF) == 2:

            params_spacex = params_space[0]
            params_spacey = params_space[1]

            ncoeff_ext_sx = np.floor(dims_sRF[0] * ext).astype(int)
            wvec_sx = fourierfreq(ncoeff_ext_sx, params_spacex)
            U_sx, freq_sx = realfftbasis(
                dims_sRF[0], ncoeff_circular=ncoeff_ext_sx, wvec=wvec_sx
            )
            freq_sx *= (2 * np.pi / ncoeff_ext_sx) ** 2

            ncoeff_ext_sy = np.floor(dims_sRF[1] * 1.25).astype(int)
            wvec_sy = fourierfreq(ncoeff_ext_sy, params_spacey)
            U_sy, freq_sy = realfftbasis(
                dims_sRF[1], ncoeff_circular=ncoeff_ext_sy, wvec=wvec_sy
            )
            freq_sy *= (2 * np.pi / ncoeff_ext_sy) ** 2

            [ww0, ww1, ww2] = np.meshgrid(freq_t, freq_sx, freq_sy)
            ww0 = np.transpose(ww0, [1, 0, 2])
            ww1 = np.transpose(ww1, [1, 0, 2])
            ww2 = np.transpose(ww2, [1, 0, 2])

            freq_comb = np.vstack([ww0.flatten(), ww1.flatten(), ww2.flatten()]).T

            Uf = np.kron(U_t, np.kron(U_sx, U_sy))

            freq_each_dim.append(freq_sx)
            freq_each_dim.append(freq_sy)
        else:
            raise NotADirectoryError(len(dims_sRF))

    return Uf.T, freq_comb, freq_each_dim
