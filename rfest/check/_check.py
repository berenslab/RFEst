import numpy as np
import scipy.stats


def eval_model_score(model, X, y, stim=True, history=True, metric='corrcoef', w_type='opt'):
    """Evaluate model score for given X and y"""
    if stim and history:
        X_dict = {'stimulus': X, 'history': y}
    elif stim and not history:
        X_dict = {'stimulus': X}
    elif not stim and history:
        X_dict = {'history': y}
    else:
        raise ValueError()
    return model.score(X_test=X_dict, y_test=y, metric=metric, w_type=w_type)


def compute_permutation_test(model, X_test, y_test, n_perm=100, history=True, metric='corrcoef', w_type='opt'):
    """Compare model performance to performance for permuted stimuli.
    If permuting the stimulus does not decrease the model performance, the fit might be pure autoregression.
    """
    score_trueX = eval_model_score(model=model, X=X_test, y=y_test, stim=True, history=history, metric=metric,
                                   w_type=w_type)

    score_permX = np.full(n_perm, np.nan)
    for i in range(n_perm):
        permX = X_test[np.random.permutation(np.arange(X_test.shape[0]))]
        score_permX[i] = eval_model_score(model=model, X=permX, y=y_test, stim=True, history=history, metric=metric,
                                          w_type=w_type)

    return score_trueX, score_permX


def significance(model, w_type='opt', show_results=False):
    W_values = {}
    p_values = {}

    for name in model.filter_names:
        W = np.squeeze(model.p[w_type][name].T @ np.linalg.inv(model.V[w_type][name]) @ model.p[w_type][name])
        p_value = 1 - scipy.stats.chi2.cdf(x=W, df=sum(model.df[name]))

        W_values[name] = W
        p_values[name] = p_value

        if show_results:
            if p_value < 0.05:
                print(f'{name}: \n\tsignificant \n\tW={W:.3f}, p_value={p_value:.3f}')
            else:
                print(f'{name}: \n\tnot significant \n\tW={W:.3f}, p_value={p_value:.3f}')

    return W_values, p_values


def residuals_pearson(y, y_pred):
    rsd = y - y_pred
    ri = rsd / np.sqrt(y_pred)
    return ri


def residuals_deviance(y, y_pred):
    quo = y / y_pred
    rsd = y - y_pred
    ri = np.sign(rsd) * np.sqrt(2 * (y * np.log(quo, out=np.zeros_like(quo), where=(quo != 0)) - rsd))

    return ri


