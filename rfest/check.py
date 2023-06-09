import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter

from rfest.utils import get_n_samples, uvec, get_spatial_and_temporal_filters


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
    """Compare model performace to performance for permuted stimuli.
    If permuting the stimulus does not decrease the model performance, the fit imight be pure autoregression.
    """
    score_trueX = eval_model_score(model=model, X=X_test, y=y_test, stim=True, history=history, metric=metric,
                                   w_type=w_type)

    score_permX = np.full(n_perm, np.nan)
    for i in range(n_perm):
        permX = X_test[np.random.permutation(np.arange(X_test.shape[0]))]
        score_permX[i] = eval_model_score(model=model, X=permX, y=y_test, stim=True, history=history, metric=metric,
                                          w_type=w_type)

    return score_trueX, score_permX


def plot_permutation_test(model, X_test, y_test, metric='corrcoef',
                          n_perm=100, q=99, history=True, ax=None, figsize=None, w_type='opt'):
    """Plot test results"""
    score_trueX, score_permX = compute_permutation_test(
        model, X_test, y_test, n_perm=n_perm, history=history, metric=metric, w_type=w_type)

    q = int(q)
    perc = np.percentile(score_permX, q=q)
    is_greater = score_trueX > perc

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    line = ax.axhline(score_trueX, c='r', zorder=0)
    line_q = ax.axhline(perc, c='b', ls=':', zorder=0)
    bp = ax.boxplot(score_permX, positions=[0], notch=True)
    ax.set(ylabel=metric, xticks=[], title='permutation test', xlim=(-0.4, 0.4))
    ax.legend(handles=[line, bp["boxes"][0], line_q], labels=['true X', 'perm. X', f'q{q}'], loc='lower right')
    ax.plot(-0.3, score_trueX, marker='o', color='lightgray', ms=20, zorder=2, alpha=1)
    ax.plot(-0.3, score_trueX, marker="*" if is_greater else "_", color='r', ms=10, zorder=3)

    return ax


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


def plot_residuals(y, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(y_pred, y - y_pred, s=1, color='black')
    ax.set_title('Residuals')

    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted')


def plot_residuals_pearson(y, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(y_pred, residuals_pearson(y, y_pred), s=1, color='black')
    ax.set_title('Perason residuals')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted')


def plot_residuals_deviance(y, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(y_pred, residuals_deviance(y, y_pred), s=1, color='black')
    ax.set_title('Deviance residuals')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Fitted')


def plot_prediction_testset(model, X_test, y_test, display_window, w_type, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    dt = model.dt
    length = get_n_samples(display_window / 60, dt)
    y = y_test

    _ = model.predict(X_test, w_type)

    y = y[model.burn_in:]
    y_pred = model.y_pred[w_type]['test']
    y_pred_lower = model.y_pred_lower[w_type]['test']
    y_pred_upper = model.y_pred_upper[w_type]['test']

    s = model.compute_score(y, model.forwardpass(model.p[w_type], 'test'), model.metric)

    tt = np.arange(len(y[:length])) * dt
    ax.plot(tt, y[:length], color='black')
    ax.plot(tt, y_pred[:length], color='red')

    ax.fill_between(tt, y_pred_upper[:length], y_pred_lower[:length], color='red', alpha=0.5)

    ax.set_xlabel('Time (second)', fontsize=12)
    ax.set_ylabel('spikes counts')
    ax.set_title(f'{model.metric}(test) = {s:.2f}')


def plot_permutation(model, kind='train', w_type='opt', q=99, num_repeat=100, ax=None, legend=False):
    if ax is None:
        _, ax = plt.subplots()

    X0 = model.X[kind].copy()
    XS0 = model.XS[kind].copy()

    score_true = model.compute_score(model.y[kind], model.forwardpass(model.p[w_type], kind), model.metric)

    # start permutation
    s_perm = []
    for _ in range(num_repeat):
        for name in model.filter_names:
            if name in model.S:
                model.XS[kind][name] = np.random.permutation(model.XS[kind][name])
            else:
                model.X[kind][name] = np.random.permutation(model.X[kind][name])

        s_perm.append(model.compute_score(model.y[kind], model.forwardpass(model.p[w_type], kind), model.metric))
    else:
        for name in model.filter_names:
            if name in model.S:
                model.XS[kind][name] = XS0[name]
            else:
                model.X[kind][name] = X0[name]

    q = int(q)
    perc = np.percentile(s_perm, q=q)
    is_greater = score_true > perc

    line = ax.axhline(score_true, c='r', zorder=0)
    line_q = ax.axhline(perc, c='b', ls=':', zorder=0)

    bp = ax.boxplot(s_perm, positions=[0])
    # ax.axhline(score_true, color='black', linestyle='--')
    # ax.plot(-0.3, score_true, marker='o', color='lightgray', ms=20, zorder=2, alpha=1)
    # ax.plot(-0.3, score_true, marker="*" if is_greater else "_", color='r', ms=10, zorder=3)
    t_value, p_value = scipy.stats.ttest_1samp(s_perm, score_true, alternative='less')
    stars = '*'
    if p_value < 0.001:
        stars *= 3
        p_result = f'p<0.001{stars}'
    elif p_value < 0.01:
        stars *= 2
        p_result = f'p<0.01{stars}'
    elif p_value < 0.05:
        stars *= 1
        p_result = f'p<0.05{stars}'
    else:
        stars = '[n.s.]'
        p_result = f'p>0.05{stars}'

    if legend:
        ax.legend(handles=[line, bp["boxes"][0], line_q], labels=['model pred.', 'permuted pred.', f'q{q}'])
    ax.set_title(f'Prediction on permuted {kind} set\n{p_result}')
    ax.set_xlim([-0.4, 0.4])


def plot_permutation_testset(model, X_test, y_test, w_type, q=99, num_repeat=100, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    model.y['test'] = y_test[model.burn_in:]
    model.X['test'] = {}
    model.XS['test'] = {}

    if type(X_test) is dict:
        for name in X_test:
            model.add_design_matrix(X_test[name], dims=model.dims[name], shift=model.shift[name], name=name,
                                    kind='test')
    else:
        # if X is np.array, assumed it's the stimulus.
        model.add_design_matrix(X_test, dims=model.dims['stimulus'], shift=model.shift['stimulus'], name='stimulus',
                                kind='test')

    if model.num_subunits != 1:
        for name in model.X['train']:
            if 'stimulus' in name:
                model.X['test'][name] = model.X['test']['stimulus']
                model.XS['test'][name] = model.XS['test']['stimulus']
        model.X['test'].pop('stimulus')
        model.XS['test'].pop('stimulus')

    Xtest0 = model.X['test'].copy()
    XStest0 = model.XS['test'].copy()

    # metric_test_true = model.compute_score(model.y['test'], model.y_pred[w_type]['test'], model.metric)
    score_true = model.compute_score(model.y['test'], model.forwardpass(model.p[w_type], 'test'), model.metric)
    s_perm = []
    for _ in range(num_repeat):
        for name in model.X['test']:
            if name in model.S:
                model.XS['test'][name] = np.random.permutation(model.XS['test'][name])
            else:
                model.X['test'][name] = np.random.permutation(model.X['test'][name])
        s_perm.append(model.compute_score(model.y['test'], model.forwardpass(model.p[w_type], 'test'), model.metric))
    else:
        for name in model.X['test']:
            if name in model.S:
                model.XS['test'][name] = XStest0[name]
            else:
                model.X['test'][name] = Xtest0[name]
    q = int(q)
    perc = np.percentile(s_perm, q=q)

    ax.axhline(score_true, c='r', zorder=0)
    ax.axhline(perc, c='b', ls=':', zorder=0)

    ax.boxplot(s_perm, positions=[0])

    t_value, p_value = scipy.stats.ttest_1samp(s_perm, score_true, alternative='less')
    stars = '*'
    if p_value < 0.001:
        stars *= 3
        p_result = f'p<0.001{stars}'
    elif p_value < 0.01:
        stars *= 2
        p_result = f'p<0.01{stars}'
    elif p_value < 0.05:
        stars *= 1
        p_result = f'p<0.05{stars}'
    else:
        stars = '[n.s.]'
        p_result = f'p>0.05{stars}'
    ax.set_title(f'Prediction on permutated test set\n{p_result}')
    ax.set_xlim([-0.4, 0.4])


def plot_cost(model, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(model.cost_train, color='black', label='Cost (train)')
    if 'dev' in model.y:
        ax.plot(model.cost_dev, color='gray', label='Cost (dev)')
    if model.distr == 'poisson':
        ax.set_ylabel('negLogLikelihood')
    else:
        ax.set_ylabel('MSE')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.legend(frameon=False)


def plot_metric(model, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(model.metric_train, color='black', label=f'{model.metric} (train)')
    if 'dev' in model.y:
        ax.plot(model.metric_dev, color='gray', label=f'{model.metric} (dev)')
    ax.set_xscale('log')
    ax.set_ylabel(f'{model.metric}')
    ax.set_xlabel('Iteration')
    ax.legend(frameon=False)


def plot1d(model, w_type='opt', w_true=None, figsize=None, ):
    """
    Parameters
    ----------

    w_type: str:
        'opt' or 'mle'

    w_true: np.array or dict
        Ground true.

    figsize: float

    """

    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    if w_true is not None:
        if type(w_true) is not dict:
            w_true = {'stimulus': w_true}

    dt = model.dt
    ncols = 5 if len(model.filter_names) < 5 else len(model.filter_names)
    nrows = 1

    if model.compute_ci:
        W_score, p_values = significance(model, w_type)

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows + 1, figure=fig)

    # plot filters
    filts = []
    for i, name in enumerate(model.filter_names):
        filts.append(fig.add_subplot(spec[0, i]))

        w = model.w[w_type][name].flatten()
        w_se = model.w_se[w_type][name]
        vmax = np.max(
            [np.abs(w.max()), np.abs(w.min()), np.abs(w.max() + 2 * w_se.max()), np.abs(w.min() - 2 * w_se.min())])
        tt = np.linspace(-model.dims[name][0] - model.shift[name], 0 - model.shift[name], model.dims[name][0]) * dt

        if w_true is not None and name in w_true:
            filts[i].plot(tt, w_true[name], color='black')

        filts[i].plot(tt, w, color=f'C{i}')
        filts[i].fill_between(tt,
                              w + 2 * w_se,
                              w - 2 * w_se, color=f'C{i}', alpha=0.5)
        filts[i].axhline(0, color='black', linestyle='--')
        filts[i].axvline(0, color='black', linestyle='--')

        if model.compute_ci:
            p = p_values[name]
            stars = '*'

            if p < 0.001:
                stars *= 3
            elif p < 0.01:
                stars *= 2
            elif p < 0.05:
                stars *= 1
            else:
                stars = '[n.s.]'
            filts[i].set_title(f'{name} \n p={p:.02f}{stars} ')
        filts[i].set_xlim(tt[0], dt)
        filts[i].set_ylim(-vmax, vmax)
        if i == 0:
            filts[0].set_xlabel('Time Lag (s)')
            filts[0].set_ylabel('Amplitude [A. U.]')

    fig.tight_layout()


def plot2d(model, w_type='opt', figsize=None, return_stats=False):
    """
    Parameters
    ----------

    model: Base
        Model object

    w_type: str
        weight types. 'opt', 'mle' or 'init'.

    figsize: tuple
        Figure size.

    return_stats : bool
        Return RF statistics?

    """

    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    dt = model.dt
    dims = model.dims
    shift = model.__dict__.get("shift", 0)
    compute_ci = model.__dict__.get("compute_ci", False)

    # rf
    ws = model.w[w_type]
    if compute_ci:
        ws_se = model.w_se[w_type]

    ncols = 4
    nrows = sum(['stimulus' in name for name in model.filter_names]) + 1
    figsize = figsize if figsize is not None else (16 * ncols / 4, 4 * nrows)
    fig = plt.figure(figsize=figsize)

    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    # plot RF and get stats
    stats = {}
    RF_data = {}
    ax_data = {}

    if compute_ci:
        W_score, p_values = significance(model, w_type)

    for i, name in enumerate(ws):

        if 'stimulus' in name:

            RF_data[name] = {
                "sRFs_min": [],
                "sRFs_max": [],
                "tRFs": [],
                "sRFs_min_cntr": [],
                "sRFs_max_cntr": [],
            }
            ax_data[name] = {
                "axes_sRF_min": [],
                "axes_sRF_max": [],
                "axes_tRF": [],
            }

            stats[name] = {
                'tRF_time_min': [],
                'tRF_time_max': [],
                'tRF_activation_min': [],
                'tRF_activation_max': [],
                'tRF_time_diff': [],
                'tRF_activation_diff': [],
                'sRF_size_min': [],
                'sRF_size_max': [],
                'sRF_size_diff': [],
            }

            t_tRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]

            w = ws[name].flatten()

            if compute_ci:
                w_se = ws_se[name].flatten()
                wu = w + 2 * w_se
                wl = w - 2 * w_se
                vmax = np.max([np.abs(w.max()), np.abs(w.min()), np.abs(wu.max()), np.abs(wu.min()), np.abs(wl.max()),
                               np.abs(wl.min())])
            else:
                vmax = np.max([np.abs(w.max()), np.abs(w.min())])

            w_uvec = uvec(ws[name].flatten()).reshape(dims[name])

            w = w.reshape(dims[name])
            sRF, tRF = get_spatial_and_temporal_filters(w, model.dims[name])
            ref = [sRF[2:-2].max(), sRF[2:-2].min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
            max_coord = np.where(sRF == ref)

            tRF = w[:, max_coord].flatten()

            if compute_ci:
                wu = wu.reshape(dims[name])
                wl = wl.reshape(dims[name])
                tRFu = wu[:, max_coord].flatten()
                tRFl = wl[:, max_coord].flatten()
            else:
                tRFu = None
                tRFl = None

            tRF_max = np.argmax(tRF)
            sRF_max = w[tRF_max]
            sRF_max_uvec = w_uvec[tRF_max]

            tRF_min = np.argmin(tRF)
            sRF_min = w[tRF_min]
            sRF_min_uvec = w_uvec[tRF_min]

            RF_data[name]['sRFs_max'].append(sRF_max_uvec)
            RF_data[name]['sRFs_min'].append(sRF_min_uvec)
            RF_data[name]['tRFs'].append(tRF)

            xrnge = np.linspace(-len(sRF_min) / 2, len(sRF_min) / 2, len(sRF_min))

            ax_RF = fig.add_subplot(spec[i, 0])
            ax_RF.imshow(w, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)

            ax_sRF_min = fig.add_subplot(spec[i, 1])
            ax_data[name]['axes_sRF_min'].append(ax_sRF_min)

            ax_sRF_min.plot(xrnge, sRF_min, color='C0')
            if compute_ci:
                ax_sRF_min.fill_between(xrnge, wu[tRF_min], wl[tRF_min], color='C0', alpha=0.5)
            ax_sRF_min.axhline(0, color='gray', linestyle='--')
            ax_sRF_min.axvline(xrnge[np.argmin(sRF_min)], color='gray', linestyle='--')

            ax_sRF_max = fig.add_subplot(spec[i, 2])
            ax_data[name]['axes_sRF_max'].append(ax_sRF_max)
            ax_sRF_max.plot(xrnge, sRF_max, color='C3')
            if compute_ci:
                ax_sRF_max.fill_between(xrnge, wu[tRF_max], wl[tRF_max], color='C3', alpha=0.5)
            ax_sRF_max.axhline(0, color='gray', linestyle='--')
            ax_sRF_max.axvline(xrnge[np.argmax(sRF_max)], color='gray', linestyle='--')

            ax_tRF = fig.add_subplot(spec[i, 3])
            ax_data[name]['axes_tRF'].append(ax_tRF)
            ax_tRF.plot(t_tRF, tRF, color='black')
            if compute_ci:
                ax_tRF.fill_between(t_tRF, tRFu, tRFl, color='gray', alpha=0.5)

            ax_tRF.axhline(0, color='gray', linestyle='--')
            ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
            ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
            if compute_ci:
                ax_tRF.set_yticks([tRFl.min(), 0, tRFu.max()])
            else:
                ax_tRF.set_yticks([tRF.min(), 0, tRF.max()])
            ax_tRF.set_ylim(-vmax, vmax)

            stats[name]['tRF_time_min'].append(t_tRF[tRF_min])
            stats[name]['tRF_time_max'].append(t_tRF[tRF_max])
            stats[name]['tRF_activation_min'].append(float(tRF[tRF_min]))
            stats[name]['tRF_activation_max'].append(float(tRF[tRF_max]))

            stats[name]['tRF_time_diff'].append(np.abs(t_tRF[tRF_max] - t_tRF[tRF_min]))
            stats[name]['tRF_activation_diff'].append(np.abs(tRF[tRF_max] - tRF[tRF_min]))

            if i == 0:
                ax_sRF_min.set_title('Spatial (min)', fontsize=14)
                ax_sRF_max.set_title('Spatial (max)', fontsize=14)
                ax_tRF.set_title('Temporal', fontsize=14)

            if compute_ci:
                p = p_values[name]

                stars = '*'

                if p < 0.001:
                    stars *= 3
                elif p < 0.01:
                    stars *= 2
                elif p < 0.05:
                    stars *= 1
                else:
                    stars = '[n.s.]'

                ax_RF.set_ylabel(f'{name} \n p={p:.02f}{stars} ', fontsize=14)
            ax_RF.axvline(max_coord, color='gray', linestyle='--', alpha=0.5)
            ax_RF.axhline(tRF_max, color='C3', linestyle='--', alpha=0.5)
            ax_RF.axhline(tRF_min, color='C0', linestyle='--', alpha=0.5)

            for ax in [ax_sRF_max, ax_sRF_min, ax_tRF]:
                asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
                ax.set_aspect(asp)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        else:

            h = ws[name].flatten()
            h_se = ws_se[name].flatten()

            hu = h + 2 * h_se
            hl = h - 2 * h_se
            t_hRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]
            ax_hRF = fig.add_subplot(spec[0, -1])
            ax_hRF.plot(t_hRF, h, color='black')
            ax_hRF.fill_between(t_hRF, hu, hl, color='gray', alpha=0.5)
            ax_hRF.spines['top'].set_visible(False)
            ax_hRF.spines['right'].set_visible(False)
            ax_hRF.set_yticks([hl.min(), 0, hu.max()])
            ax_hRF.set_title(name.capitalize(), fontsize=14)

    fig.tight_layout()

    if return_stats:
        return RF_data, stats


def plot3d(model, w_type='opt', contour=0.1, pixel_size=30, figsize=None, return_stats=False):
    """
    Parameters
    ----------

    model: Base
        Model object

    w_type: str
        weight types. 'opt', 'mle' or 'init'.

    contour: float
        > 0. The contour level. Default is 0.015.

    pixel_size:
        The size of pixel for calculating the contour size.

    figsize: tuple
        Figure size.

    return_stats: bool
        Return model stats?

    """
    import matplotlib.gridspec as gridspec
    import warnings

    try:
        import cv2
    except ImportError:
        cv2 = None
        warnings.warn('Failed to import cv2: Cannot compute contour areas.')

    warnings.filterwarnings("ignore")

    dt = model.dt
    dims = model.dims
    shift = model.shift

    # rf
    ws = model.w[w_type]
    if model.compute_ci:
        ws_se = model.w_se[w_type]

    ncols = 4
    nrows = sum(['stimulus' in name for name in model.filter_names]) + 1
    figsize = figsize if figsize is not None else (16 * ncols / 4, 4 * nrows)
    fig = plt.figure(figsize=figsize)

    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    # plot RF and get stats
    stats = {}
    RF_data = {}
    ax_data = {}

    if model.compute_ci:
        W_score, p_values = significance(model, w_type)

    for i, name in enumerate(ws):

        if 'stimulus' in name:

            RF_data[name] = {

                "sRFs_min": [],
                "sRFs_max": [],
                "tRFs": [],
                "sRFs_min_cntr": [],
                "sRFs_max_cntr": [],
            }
            ax_data[name] = {
                "axes_sRF_min": [],
                "axes_sRF_max": [],
                "axes_tRF": [],
            }

            stats[name] = {
                'tRF_time_min': [],
                'tRF_time_max': [],
                'tRF_activation_min': [],
                'tRF_activation_max': [],
                'tRF_time_diff': [],
                'tRF_activation_diff': [],
                'sRF_size_min': [],
                'sRF_size_max': [],
                'sRF_size_diff': [],
            }

            t_tRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]

            w = ws[name].flatten()
            w_uvec = uvec(ws[name].flatten())
            if model.compute_ci:
                w_se = ws_se[name].flatten()
                wu = w + 2 * w_se
                wl = w - 2 * w_se
                vmax = np.max([np.abs(w.max()), np.abs(w.min()), np.abs(wu.max()), np.abs(wu.min()), np.abs(wl.max()),
                               np.abs(wl.min())])
            else:
                vmax = np.max([np.abs(w.max()), np.abs(w.min())])

            w = w.reshape(dims[name])
            w_uvec = w_uvec.reshape(dims[name])
            if model.compute_ci:
                wu = wu.reshape(dims[name])
                wl = wl.reshape(dims[name])

            sRF, tRF = get_spatial_and_temporal_filters(w, model.dims[name])
            ref = [sRF[2:, 2:].max(), sRF[2:, 2:].min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
            max_coord = np.where(sRF == ref)

            tRF = w[:, max_coord[0], max_coord[1]].flatten()
            if model.compute_ci:
                tRFu = wu[:, max_coord[0], max_coord[1]].flatten()
                tRFl = wl[:, max_coord[0], max_coord[1]].flatten()

            tRF_max = np.argmax(tRF)
            sRF_max = w[tRF_max]
            sRF_max_uvec = w_uvec[tRF_max]
            tRF_min = np.argmin(tRF)
            sRF_min = w[tRF_min]
            sRF_min_uvec = w_uvec[tRF_min]

            RF_data[name]['sRFs_max'].append(sRF_max_uvec)
            RF_data[name]['sRFs_min'].append(sRF_min_uvec)
            RF_data[name]['tRFs'].append(tRF)

            ax_sRF_min = fig.add_subplot(spec[i, 0])
            ax_data[name]['axes_sRF_min'].append(ax_sRF_min)

            ax_sRF_min.imshow(sRF_min.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, aspect="auto")
            ax_sRF_min.set_xticks([])
            ax_sRF_min.set_yticks([])

            ax_sRF_max = fig.add_subplot(spec[i, 1])
            ax_data[name]['axes_sRF_max'].append(ax_sRF_max)
            ax_sRF_max.imshow(sRF_max.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, aspect="auto")
            ax_sRF_max.set_xticks([])
            ax_sRF_max.set_yticks([])

            ax_tRF = fig.add_subplot(spec[i, 2])
            ax_data[name]['axes_tRF'].append(ax_tRF)
            ax_tRF.plot(t_tRF, tRF, color='black')
            if model.compute_ci:
                ax_tRF.fill_between(t_tRF, tRFu, tRFl, color='gray', alpha=0.5)
                ax_tRF.set_yticks([tRFl.min(), 0, tRFu.max()])
            else:
                ax_tRF.set_yticks([tRF.min(), 0, tRF.max()])

            ax_tRF.axhline(0, color='gray', linestyle='--')
            ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
            ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
            ax_tRF.spines['top'].set_visible(False)
            ax_tRF.spines['right'].set_visible(False)

            ax_tRF.set_ylim(-vmax, vmax)

            stats[name]['tRF_time_min'].append(t_tRF[tRF_min])
            stats[name]['tRF_time_max'].append(t_tRF[tRF_max])
            stats[name]['tRF_activation_min'].append(float(tRF[tRF_min]))
            stats[name]['tRF_activation_max'].append(float(tRF[tRF_max]))

            stats[name]['tRF_time_diff'].append(np.abs(t_tRF[tRF_max] - t_tRF[tRF_min]))
            stats[name]['tRF_activation_diff'].append(np.abs(tRF[tRF_max] - tRF[tRF_min]))

            if i == 0:
                ax_sRF_min.set_title('Spatial (min)', fontsize=14)
                ax_sRF_max.set_title('Spatial (max)', fontsize=14)
                ax_tRF.set_title('Temporal', fontsize=14)

            if model.compute_ci:
                p = p_values[name]

                stars = '*'

                if p < 0.001:
                    stars *= 3
                elif p < 0.01:
                    stars *= 2
                elif p < 0.05:
                    stars *= 1
                else:
                    stars = '[n.s.]'

                ax_sRF_min.set_ylabel(f'{name} \n p={p:.02f}{stars} ', fontsize=14)

        else:

            h = ws[name].flatten()
            h_se = ws_se[name].flatten()

            hu = h + 2 * h_se
            hl = h - 2 * h_se
            t_hRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]
            ax_hRF = fig.add_subplot(spec[0, -1])
            ax_hRF.plot(t_hRF, h, color='black')
            if model.compute_ci:
                ax_hRF.fill_between(t_hRF, hu, hl, color='gray', alpha=0.5)
                ax_hRF.set_yticks([hl.min(), 0, hu.max()])
            else:
                ax_hRF.set_yticks([h.min(), 0, h.max()])

            ax_hRF.spines['top'].set_visible(False)
            ax_hRF.spines['right'].set_visible(False)
            ax_hRF.set_title(name.capitalize(), fontsize=14)

    # contour and more stats

    for name in ws:

        if 'stimulus' in name:
            sRFs_min = RF_data[name]["sRFs_min"]
            sRFs_max = RF_data[name]["sRFs_max"]
            tRFs = RF_data[name]["tRFs"]

            axes_sRF_min = ax_data[name]["axes_sRF_min"]
            axes_sRF_max = ax_data[name]["axes_sRF_max"]
            axes_tRF = ax_data[name]["axes_tRF"]

            color_min = 'lightsteelblue'
            color_max = 'lightcoral'

            contour_size_min = []
            contour_size_max = []

            i = 0
            CS_min = axes_sRF_min[i].contour(sRFs_min[i].T, levels=[-contour], colors=[color_min], linestyles=['-'],
                                             linewidths=3, alpha=1)
            CS_max = axes_sRF_max[i].contour(sRFs_max[i].T, levels=[contour], colors=[color_max], linestyles=['-'],
                                             linewidths=3, alpha=1)

            cntrs_min = [p.vertices for p in CS_min.collections[0].get_paths()]
            cntrs_max = [p.vertices for p in CS_max.collections[0].get_paths()]

            RF_data[name]["sRFs_min_cntr"].append(cntrs_min[0])
            RF_data[name]["sRFs_max_cntr"].append(cntrs_max[0])

            if cv2 is not None:

                cntrs_size_min = [cv2.contourArea(cntr.astype(np.float32)) * pixel_size ** 2 / 1000 for cntr in
                                  cntrs_min]
                axes_sRF_min[i].set_xlabel(f'cntr size = {cntrs_size_min[0]:.03f} 10^3 μm^2')

                cntrs_size_max = [cv2.contourArea(cntr.astype(np.float32)) * pixel_size ** 2 / 1000 for cntr in
                                  cntrs_max]
                axes_sRF_max[i].set_xlabel(f'cntr size = {cntrs_size_max[0]:.03f} 10^3 μm^2')

                contour_size_min += cntrs_size_min
                contour_size_max += cntrs_size_max

                stats[name]['sRF_size_min'].append(cntrs_size_min[0])
                stats[name]['sRF_size_max'].append(cntrs_size_max[0])
                stats[name]['sRF_size_diff'].append(np.abs(cntrs_size_min[0] - cntrs_size_max[0]))

    fig.tight_layout()

    if return_stats:
        return RF_data, stats


def plot_diagnostics(model, X_test=None, y_test=None, w_type='opt', metric='corrcoef', display_window=100,
                     plot_rsd=True, figsize=None, num_repeat=100, random_seed=2046):
    """
    Parameters
    ----------

    display_window: float
        Seconds of prediction to display.

    """

    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(random_seed)
    dt = model.dt
    ncols = 5 if len(model.filter_names) < 5 else len(model.filter_names)
    nrows = 3

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows + 1, figure=fig)

    diagnostics = []
    diagnostics.append(fig.add_subplot(spec[0, 0]))
    if w_type == 'opt':
        plot_cost(model, ax=diagnostics[0])
    else:
        diagnostics[0].axis('off')

    diagnostics.append(fig.add_subplot(spec[0, 1]))
    if w_type == 'opt':
        plot_metric(model, ax=diagnostics[1])
    else:
        diagnostics[1].axis('off')

    # check if metric exists
    if model.__dict__.get('metric', None) is None:
        model.metric = metric

    length = get_n_samples(display_window / 60, dt)
    y_train = model.y['train']
    y_pred_train = model.y_pred[w_type]['train']
    if model.compute_ci:
        y_pred_train_lower = model.y_pred_lower[w_type]['train']
        y_pred_train_upper = model.y_pred_upper[w_type]['train']

    if model.distr != 'gaussian':
        diagnostics.append(fig.add_subplot(spec[0, 2]))
        if plot_rsd:
            plot_residuals_pearson(y_train, y_pred_train, ax=diagnostics[2])
        else:
            diagnostics[2].axis('off')
        diagnostics.append(fig.add_subplot(spec[0, 3]))
        if plot_rsd:
            plot_residuals_deviance(y_train, y_pred_train, ax=diagnostics[3])
        else:
            diagnostics[3].axis('off')
    else:
        diagnostics.append(fig.add_subplot(spec[0, 2]))
        if plot_rsd:
            plot_residuals(y_train, y_pred_train, ax=diagnostics[2])
        else:
            diagnostics[2].axis('off')
        diagnostics.append(fig.add_subplot(spec[0, 3]))
        diagnostics[3].axis('off')

    diagnostics.append(fig.add_subplot(spec[0, 4]))
    plot_permutation(model, kind='train', w_type=w_type, num_repeat=num_repeat, ax=diagnostics[4])

    if 'dev' in model.y:
        diagnostics.append(fig.add_subplot(spec[1, 4]))
        plot_permutation(model, kind='dev', w_type=w_type, num_repeat=num_repeat, ax=diagnostics[5])
    # else:
    # diagnostics[-1].axis('off')

    if 'dev' in model.y:
        y_dev = model.y['dev']
        y_pred_dev = model.y_pred[w_type]['dev']

        y = np.hstack([y_train[-int(length / 2):], y_dev[:int(length / 2)]])
        y_pred = np.hstack([y_pred_train[-int(length / 2):], y_pred_dev[:int(length / 2)]])

        if model.compute_ci:
            y_pred_dev_lower = model.y_pred_lower[w_type]['dev']
            y_pred_dev_upper = model.y_pred_upper[w_type]['dev']

            y_pred_lower = np.hstack([y_pred_train_lower[-int(length / 2):], y_pred_dev_lower[:int(length / 2)]])
            y_pred_upper = np.hstack([y_pred_train_upper[-int(length / 2):], y_pred_dev_upper[:int(length / 2)]])
    else:
        y = y_train[-int(length):]
        y_pred = y_pred_train[-int(length):]
        if model.compute_ci:
            y_pred_lower = y_pred_train_lower[-int(length):]
            y_pred_upper = y_pred_train_upper[-int(length):]

    tt = np.arange(len(y)) * dt

    # plot train and dev prediction
    axes_pred = fig.add_subplot(spec[1, :-1])
    axes_pred.plot(tt, y, color='black')
    axes_pred.plot(tt, y_pred, color='red')
    if model.compute_ci:
        axes_pred.fill_between(tt, y_pred_lower, y_pred_upper, color='red', alpha=0.5)

    if len(y) == length:
        axes_pred.axvline(tt[int(length / 2)], color='black', linestyle='--')
        axes_pred.set_xlabel('(train set)  <-    Time (second)  ->     (dev set)', fontsize=12)
    else:
        axes_pred.axvline(tt[len(y_train[-int(length / 2):])], color='black', linestyle='--')
        axes_pred.set_xlabel('Time (second)', fontsize=12)

    axes_pred.set_ylabel('Data')

    metric_train = model.compute_score(model.y['train'], model.forwardpass(model.p[w_type], 'train'), model.metric)
    # metric_train = model.compute_score(model.y['train'], model.y_pred[w_type]['train'], model.metric)
    if 'dev' in model.y:
        metric_dev = model.compute_score(model.y['dev'], model.forwardpass(model.p[w_type], 'dev'), model.metric)
        # metric_dev = model.compute_score(model.y['dev'], model.y_pred[w_type]['dev'], model.metric)
        axes_pred.set_title(f'{model.metric}(train) = {metric_train:.2f} |  {model.metric}(dev) = {metric_dev:.2f}')
    else:
        axes_pred.set_title(f'{model.metric}(train) = {metric_train:.2f}')

        # test set prediction
    if X_test is not None:
        axes_pred_test = fig.add_subplot(spec[2, :-1])
        plot_prediction_testset(model, X_test, y_test, display_window, w_type, ax=axes_pred_test)

        diagnostics.append(fig.add_subplot(spec[2, 4]))
        plot_permutation_testset(model, X_test, y_test, w_type=w_type, num_repeat=num_repeat, ax=diagnostics[6])

    fig.tight_layout()


def plot_subunits3d(model, X_test, y_test, dt=None, shift=None, model_name=None, response_type='spike', len_time=1,
                    contour=None, figsize=None):
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(dims[0] - shift) * dt, shift * dt, dims[0] + 1)[1:]

    ws = uvec(model.w_opt)

    num_subunits = ws.shape[1]

    sRFs_max = []
    sRFs_min = []
    tRFs = []
    for i in range(num_subunits):
        w = ws[:, i].reshape(dims)
        sRF, tRF = get_spatial_and_temporal_filters(w, dims)

        ref = [sRF[2:, 2:].max(), sRF[2:, 2:].min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
        max_coord = np.where(sRF == ref)
        tRF = w[:, max_coord[0], max_coord[1]].flatten()
        tRF_max = np.argmax(tRF)
        sRF_max = w[tRF_max]
        sRFs_max.append(sRF_max)
        tRF_min = np.argmin(tRF)
        sRF_min = w[tRF_min]
        sRFs_min.append(sRF_min)

        tRFs.append(tRF)

    sRFs_max = np.stack(sRFs_max)
    sRFs_min = np.stack(sRFs_min)

    vmax = np.max([np.abs(np.max(sRFs_max)), np.abs(np.min(sRFs_max)),
                   np.abs(np.max(sRFs_min)), np.abs(np.min(sRFs_min))])

    # fig = plt.figure(figsize=(8, 4))

    ncols = num_subunits if num_subunits > 5 else 5
    nrows = 3
    if model.__dict__.get('fnl_fitted', None) is not None:
        nrows += 1

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows + 1, figure=fig)
    axs = []
    ax_sRF_mins = []
    ax_sRF_maxs = []
    for i in range(num_subunits):
        ax_sRF_min = fig.add_subplot(spec[0, i])
        if model.__dict__.get('w_spl', None) is not None:
            ax_sRF_min.imshow(sRFs_min[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        else:
            ax_sRF_min.imshow(gaussian_filter(sRFs_min[i].T, sigma=1), cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        ax_sRF_min.set_xticks([])
        ax_sRF_min.set_yticks([])
        ax_sRF_min.set_title(f'S{i}')

        ax_sRF_max = fig.add_subplot(spec[1, i])
        if model.__dict__.get('w_spl', None) is not None:
            ax_sRF_max.imshow(sRFs_max[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        else:
            ax_sRF_max.imshow(gaussian_filter(sRFs_max[i].T, sigma=1), cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)

        ax_sRF_max.set_xticks([])
        ax_sRF_max.set_yticks([])

        ax_tRF = fig.add_subplot(spec[2, i])
        ax_tRF.plot(t_tRF, tRFs[i], color='black')
        ax_tRF.spines['top'].set_visible(False)
        ax_tRF.spines['right'].set_visible(False)

        tRF_max = np.argmax(tRFs[i])
        tRF_min = np.argmin(tRFs[i])
        ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
        ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)

        if i == 0:
            ax_sRF_min.set_ylabel('Min Frame')
            ax_sRF_max.set_ylabel('Max Frame')
            ax_tRF.set_ylabel('Temporal')

        ax_sRF_mins.append(ax_sRF_min)
        ax_sRF_maxs.append(ax_sRF_max)

        if model.__dict__.get('nl_params_opt', None) is not None:
            ax_nl = fig.add_subplot(spec[3, i])
            xrng = model.nl_xrange
            nl_opt = model.fnl_fitted(model.nl_params_opt[i], model.nl_xrange)
            ax_nl.plot(xrng, nl_opt, color='black', linewidth=3)
            ax_nl.plot(xrng, model.nl_basis * model.nl_params_opt[i], color='grey', alpha=0.5)

            ax_nl.spines['top'].set_visible(False)
            ax_nl.spines['right'].set_visible(False)
            if i == 0:
                ax_nl.set_ylabel('Fitted Nonlinearity')

    if contour is not None:  # then plot contour
        for i in range(num_subunits):
            for j in range(num_subunits):
                if i != j:
                    color = 'grey'
                    alpha = 0.5
                    style = '--'
                else:
                    color = 'black'
                    alpha = 1
                    style = '--'
                ax_sRF_mins[i].contour(sRFs_min[j].T, levels=[-contour], colors=[color], linestyles=[style],
                                       alpha=alpha)
                ax_sRF_maxs[i].contour(sRFs_max[j].T, levels=[contour], colors=[color], linestyles=[style], alpha=alpha)

    if (model.__dict__.get('h_opt', None) is not None) and (model.__dict__.get('fnl_fitted', None) is None):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h + 1) * dt, -1 * dt, dims_h + 1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)

        ax_pred = fig.add_subplot(spec[nrows, :-1])

    elif (model.__dict__.get('h_opt', None) is None) and (model.__dict__.get('fnl_fitted', None) is not None):

        if model.output_nonlinearity == 'spline' or model.output_nonlinearity == 'nn':
            ax_nl = fig.add_subplot(spec[nrows, -1])
            nl = model.fnl_fitted(model.nl_params_opt[-1], model.nl_xrange)
            xrng = model.nl_xrange

            ax_nl.plot(xrng, nl)
            ax_nl.plot(xrng, model.nl_basis * model.nl_params_opt[-1], color='grey', alpha=0.5)

            ax_nl.set_title('Fitted output nonlinearity')
            ax_nl.spines['top'].set_visible(False)
            ax_nl.spines['right'].set_visible(False)

        ax_pred = fig.add_subplot(spec[nrows, :-1])

    elif (model.__dict__.get('h_opt', None) is not None) and (model.__dict__.get('fnl_fitted', None) is not None):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h + 1) * dt, -1 * dt, dims_h + 1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -2])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black', linewidth=3)
        ax_h_opt.plot(t_hRF, model.Sh * model.bh_opt, color='grey')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)

        ax_nl = fig.add_subplot(spec[nrows, -1])
        xrng = model.nl_xrange
        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)
        ax_nl.plot(xrng, nl0)

        if model.__dict__.get('nl_params_opt', None) is not None:

            if model.output_nonlinearity == 'spline' or model.output_nonlinearity == 'nn':
                nl_opt = model.fnl_fitted(model.nl_params_opt[-1], model.nl_xrange)
                ax_nl.plot(xrng, nl_opt, color='black', linewidth=3)
                ax_nl.plot(xrng, model.nl_basis * model.nl_params_opt[-1], color='grey', alpha=0.5)
            else:
                ax_nl.axis('off')

        ax_nl.set_title('Fitted output nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)

        ax_pred = fig.add_subplot(spec[nrows, :-2])

    else:
        ax_pred = fig.add_subplot(spec[nrows, :])

    y_pred = model.predict(X_test, y_test)

    if len_time is not None:
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    t_pred = np.arange(n)

    pred_score = model.score(X_test, y_test)

    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                                                       markerfmt='none', use_line_collection=True,
                                                       label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline, 'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')

    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left", frameon=False)
    ax_pred.set_title('Prediction performance')

    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'{model_name}', fontsize=14)


def plot3dn(model, w_type='opt', contour=0.1, pixel_size=30, figsize=None, return_stats=False):
    """
    plot3dn(ew).


    Parameters
    ----------

    model: object
        Model object

    w_type: str
        weight types. 'opt', 'mle' or 'init'.

    contour: float
        > 0. The contour level. Default is 0.015.

    pixel_size:
        The size of pixel for calculating the contour size.

    figsize: tuple
        Figure size.

    """

    import matplotlib.gridspec as gridspec
    import warnings

    try:
        import cv2
    except ImportError:
        cv2 = None
        warnings.warn('Failed to import cv2: Cannot compute contour areas.')

    warnings.filterwarnings("ignore")

    dt = model.dt
    dims = model.dims
    shift = model.shift

    # rf
    ws = model.w[w_type]
    if model.compute_ci:
        ws_se = model.w_se[w_type]

    ncols = 4
    nrows = sum(['stimulus' in name for name in model.filter_names]) + 1
    figsize = figsize if figsize is not None else (16 * ncols / 4, 4 * nrows)
    fig = plt.figure(figsize=figsize)

    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    # plot RF and get stats
    stats = {}
    RF_data = {}
    ax_data = {}

    if model.compute_ci:
        W_score, p_values = significance(model, w_type)

    for i, name in enumerate(ws):

        if 'stimulus' in name:

            RF_data[name] = {

                "sRFs_min": [],
                "sRFs_max": [],
                "tRFs_min": [],
                "tRFs_max": [],
                "sRFs_min_cntr": [],
                "sRFs_max_cntr": [],
            }
            ax_data[name] = {
                "axes_sRF_min": [],
                "axes_sRF_max": [],
                "axes_tRF": [],
            }

            stats[name] = {
                'tRF_time_min': [],
                'tRF_time_max': [],
                'tRF_activation_min': [],
                'tRF_activation_max': [],
                'tRF_time_diff': [],
                'tRF_activation_diff': [],
                'sRF_size_min': [],
                'sRF_size_max': [],
                'sRF_size_diff': [],
            }

            t_tRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]

            w = ws[name].flatten()
            w_uvec = uvec(ws[name].flatten())
            if model.compute_ci:
                w_se = ws_se[name].flatten()
                wu = w + 2 * w_se
                wl = w - 2 * w_se
                vmax = np.max([np.abs(w.max()), np.abs(w.min()), np.abs(wu.max()), np.abs(wu.min()), np.abs(wl.max()),
                               np.abs(wl.min())])
            else:
                vmax = np.max([np.abs(w.max()), np.abs(w.min())])

            w = w.reshape(dims[name])
            w_uvec = w_uvec.reshape(dims[name])
            if model.compute_ci:
                wu = wu.reshape(dims[name])
                wl = wl.reshape(dims[name])

            min_coord = np.hstack(np.where(w == w.min()))
            max_coord = np.hstack(np.where(w == w.max()))

            tRF_max = w[:, max_coord[1], max_coord[2]].flatten()
            tRF_min = w[:, min_coord[1], min_coord[2]].flatten()
            if model.compute_ci:
                tRFu_max = wu[:, max_coord[1], max_coord[2]].flatten()
                tRFl_max = wl[:, max_coord[1], max_coord[2]].flatten()
                tRFu_min = wu[:, min_coord[1], min_coord[2]].flatten()
                tRFl_min = wl[:, min_coord[1], min_coord[2]].flatten()

            sRF_max = w[max_coord[0]]
            sRF_max_uvec = w_uvec[max_coord[0]]
            sRF_min = w[min_coord[0]]
            sRF_min_uvec = w_uvec[min_coord[0]]

            RF_data[name]['sRFs_max'].append(sRF_max_uvec)
            RF_data[name]['sRFs_min'].append(sRF_min_uvec)
            RF_data[name]['tRFs_max'].append(tRF_max)
            RF_data[name]['tRFs_min'].append(tRF_min)

            ax_sRF_min = fig.add_subplot(spec[i, 0])
            ax_data[name]['axes_sRF_min'].append(ax_sRF_min)

            ax_sRF_min.imshow(sRF_min.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, aspect="auto")
            ax_sRF_min.set_xticks([])
            ax_sRF_min.set_yticks([])

            ax_sRF_max = fig.add_subplot(spec[i, 1])
            ax_data[name]['axes_sRF_max'].append(ax_sRF_max)
            ax_sRF_max.imshow(sRF_max.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, aspect="auto")
            ax_sRF_max.set_xticks([])
            ax_sRF_max.set_yticks([])

            ax_tRF = fig.add_subplot(spec[i, 2])
            ax_data[name]['axes_tRF'].append(ax_tRF)
            ax_tRF.plot(t_tRF, tRF_max, color='red')
            ax_tRF.plot(t_tRF, tRF_min, color='blue')
            if model.compute_ci:
                ax_tRF.fill_between(t_tRF, tRFu_max, tRFl_max, color='C3', alpha=0.5)
                ax_tRF.fill_between(t_tRF, tRFu_min, tRFl_min, color='C0', alpha=0.5)
                ax_tRF.set_yticks([tRFl_max.min(), 0, tRFu_max.max()])
            else:
                ax_tRF.set_yticks([tRF_max.min(), 0, tRF_max.max()])

            ax_tRF.axhline(0, color='gray', linestyle='--')
            ax_tRF.axvline(t_tRF[max_coord[0]], color='C3', linestyle='--', alpha=0.6)
            ax_tRF.axvline(t_tRF[min_coord[0]], color='C0', linestyle='--', alpha=0.6)
            ax_tRF.spines['top'].set_visible(False)
            ax_tRF.spines['right'].set_visible(False)

            ax_tRF.set_ylim(-vmax, vmax)

            stats[name]['tRF_time_min'].append(t_tRF[min_coord[0]])
            stats[name]['tRF_time_max'].append(t_tRF[max_coord[0]])
            stats[name]['tRF_activation_min'].append(float(tRF_min[min_coord[0]]))
            stats[name]['tRF_activation_max'].append(float(tRF_max[max_coord[0]]))

            stats[name]['tRF_time_diff'].append(np.abs(t_tRF[max_coord[0]] - t_tRF[min_coord[0]]))
            stats[name]['tRF_activation_diff'].append(np.abs(tRF_max[max_coord[0]] - tRF_min[min_coord[0]]))

            if i == 0:
                ax_sRF_min.set_title('Spatial (min)', fontsize=14)
                ax_sRF_max.set_title('Spatial (max)', fontsize=14)
                ax_tRF.set_title('Temporal', fontsize=14)

            if model.compute_ci:
                p = p_values[name]

                stars = '*'

                if p < 0.001:
                    stars *= 3
                elif p < 0.01:
                    stars *= 2
                elif p < 0.05:
                    stars *= 1
                else:
                    stars = '[n.s.]'

                ax_sRF_min.set_ylabel(f'{name} \n p={p:.02f}{stars} ', fontsize=14)

        else:

            h = ws[name].flatten()
            h_se = ws_se[name].flatten()

            hu = h + 2 * h_se
            hl = h - 2 * h_se
            t_hRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]
            ax_hRF = fig.add_subplot(spec[0, -1])
            ax_hRF.plot(t_hRF, h, color='black')
            if model.compute_ci:
                ax_hRF.fill_between(t_hRF, hu, hl, color='gray', alpha=0.5)
                ax_hRF.set_yticks([hl.min(), 0, hu.max()])
            else:
                ax_hRF.set_yticks([h.min(), 0, h.max()])

            ax_hRF.spines['top'].set_visible(False)
            ax_hRF.spines['right'].set_visible(False)
            ax_hRF.set_title(name.capitalize(), fontsize=14)

    # contour and more stats

    for name in ws:

        if 'stimulus' in name:
            sRFs_min = RF_data[name]["sRFs_min"]
            sRFs_max = RF_data[name]["sRFs_max"]
            tRFs_min = RF_data[name]["tRFs_min"]
            tRFs_max = RF_data[name]["tRFs_max"]

            axes_sRF_min = ax_data[name]["axes_sRF_min"]
            axes_sRF_max = ax_data[name]["axes_sRF_max"]
            axes_tRF = ax_data[name]["axes_tRF"]

            color_min = 'lightsteelblue'
            color_max = 'lightcoral'

            contour_size_min = []
            contour_size_max = []

            i = 0
            CS_min = axes_sRF_min[i].contour(sRFs_min[i].T, levels=[-contour], colors=[color_min], linestyles=['-'],
                                             linewidths=3, alpha=1)
            CS_max = axes_sRF_max[i].contour(sRFs_max[i].T, levels=[contour], colors=[color_max], linestyles=['-'],
                                             linewidths=3, alpha=1)

            cntrs_min = [p.vertices for p in CS_min.collections[0].get_paths()]
            cntrs_max = [p.vertices for p in CS_max.collections[0].get_paths()]

            RF_data[name]["sRFs_min_cntr"].append(cntrs_min[0])
            RF_data[name]["sRFs_max_cntr"].append(cntrs_max[0])

            if cv2 is not None:
                cntrs_size_min = [cv2.contourArea(cntr.astype(np.float32)) * pixel_size ** 2 / 1000 for cntr in
                                  cntrs_min]
                axes_sRF_min[i].set_xlabel(f'cntr size = {cntrs_size_min[0]:.03f} 10^3 μm^2')

                cntrs_size_max = [cv2.contourArea(cntr.astype(np.float32)) * pixel_size ** 2 / 1000 for cntr in
                                  cntrs_max]
                axes_sRF_max[i].set_xlabel(f'cntr size = {cntrs_size_max[0]:.03f} 10^3 μm^2')

                contour_size_min += cntrs_size_min
                contour_size_max += cntrs_size_max


                stats[name]['sRF_size_min'].append(cntrs_size_min[0])
                stats[name]['sRF_size_max'].append(cntrs_size_max[0])
                stats[name]['sRF_size_diff'].append(np.abs(cntrs_size_min[0] - cntrs_size_max[0]))

    fig.tight_layout()

    if return_stats:
        return RF_data, stats


def plot3d_allframes(model, figsize=None, transpose=False):
    """
    Plot all frames of a 3D RF. Not for subunits yet.
    """

    dt = model.dt
    dims = model.dims['stimulus']
    w = model.w['opt']['stimulus'].reshape(dims)
    tt = np.linspace(-dims[0] * dt, 0, dims[0])
    vmax = np.max([np.abs(w.min()), w.max()])

    fig, ax = plt.subplots(1, dims[0], figsize=figsize)
    for i in range(dims[0]):
        if transpose:
            ax[i].imshow(w[i].T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
        else:
            ax[i].imshow(w[i], cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if i == 0:
            ax[i].set_title(f't={tt[i]:.02f}')
        else:
            ax[i].set_title(f'{tt[i]:.02f}')
    fig.tight_layout()
