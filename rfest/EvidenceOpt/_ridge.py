import jax.numpy as np
from sklearn.metrics import mean_squared_error

from rfest.EvidenceOpt._base import EmpiricalBayes

__all__ = ['Ridge', 'RidgeFixedPoint']


class Ridge(EmpiricalBayes):
    """
    Ridge Regression
    """

    def __init__(self, X, Y, dims, compute_mle=True):
        super().__init__(X, Y, dims, compute_mle, time='ridge', space='ridge', n_hp_time=1, n_hp_space=1)


class RidgeFixedPoint:
    """
    Ridge regression updated with iterative fixed-point algorithm.
    """

    def __init__(self, X, y, dims):

        self.w_opt = None
        self.optimized_C_post = None
        self.optimized_C_prior = None
        self.optimized_params = None
        self.X = np.array(X)  # stimulus design matrix
        self.Y = np.array(y)  # response

        self.dims = dims  # assumed order [t, y, x]
        self.n_samples, self.n_features = X.shape

        self.XtX = X.T @ X
        self.XtY = X.T @ y
        self.YtY = y.T @ y

        self.w_mle = np.linalg.solve(self.XtX, self.XtY)

    def update_params(self, params, C_post, m_post):

        # sigma = params[0]
        theta = params[1]

        theta = (self.n_features - theta * np.trace(C_post)) / np.sum(m_post ** 2)

        upper = np.sum(self.YtY - 2 * self.XtY * m_post + m_post.T @ self.XtX @ m_post)
        lower = self.n_features - np.sum(1 - theta * np.diag(C_post))
        sigma = upper / lower

        return np.asarray([sigma, theta])

    def update_C_prior(self, params):

        # sigma = params[0]
        theta = params[1]

        C_prior = np.identity(self.n_features) * 1 / theta
        C_prior_inv = np.identity(self.n_features) * theta
        return C_prior, C_prior_inv

    def update_C_posterior(self, params, C_prior_inv):

        sigma = params[0]

        C_post_inv = self.XtX / sigma ** 2 + C_prior_inv
        C_post = np.linalg.inv(C_post_inv)

        m_post = C_post @ self.XtY / (sigma ** 2)

        return C_post, C_post_inv, m_post

    def fit(self, p0, num_iters=100, threshold=1e-6, MAXTHETA=1e6, verbose=True):

        params = p0

        if verbose:
            print('Iter\tσ\tθ')
            print('{0}\t{1:.3f}\t{2:.3f}'.format(0, params[0], params[1]))

        i = 0
        for i in np.arange(1, num_iters + 1):

            params0 = params

            (C_prior, C_prior_inv) = self.update_C_prior(params)
            (C_post, C_post_inv, m_post) = self.update_C_posterior(params, C_prior_inv)

            params = self.update_params(params, C_post, m_post)

            dparams = np.linalg.norm(params - params0)

            if dparams < threshold:
                if verbose:
                    print('{0}\t{1:.3f}\t{2:.3f}'.format(i, params[0], params[1]))
                    print('Finished: Converged in {} steps'.format(i))
                break
            elif params[1] > MAXTHETA:
                if verbose:
                    print('{0}\t{1:.3f}\t{2:.3f}'.format(i, params[0], params[1]))
                    print('Finished: ridge regression: filter is all-zeros.')
                break
        else:
            if verbose:
                print('{0}\t{1:.3f}\t{2:.3f}'.format(i, params[0], params[1]))
                print('Finished: reached maxiter = {}.'.format(num_iters))

        self.optimized_params = params

        (optimized_C_prior,
         optimized_C_prior_inv) = self.update_C_prior(self.optimized_params)

        (optimized_C_post,
         optimized_C_post_inv,
         optimized_m_post) = self.update_C_posterior(self.optimized_params,
                                                     optimized_C_prior_inv)

        self.optimized_C_prior = optimized_C_prior
        self.optimized_C_post = optimized_C_post
        self.w_opt = optimized_m_post

    @staticmethod
    def _rcv(w, wSTA_test, X_test, y_test):

        """Relative Mean Squared Error"""

        a = mean_squared_error(y_test, X_test @ w)
        b = mean_squared_error(y_test, X_test @ wSTA_test)

        return a - b

    def measure_prediction_performance(self, X_test, y_test):

        wSTA_test = np.linalg.solve(X_test.T @ X_test, X_test.T @ y_test)

        w = self.w_opt.ravel()

        return self._rcv(w, wSTA_test, X_test, y_test)
