import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .utils import get_n_samples, uvec, get_spatial_and_temporal_filters


def eval_model_score(model, X, y, stim=True, history=True, metric='corrcoef'):
    """Evaluate model score for given X and y"""
    if stim and history:
        X_dict = {'stimulus': X, 'history': y}
    elif stim and not history:
        X_dict = {'stimulus': X}
    elif not stim and history:
        X_dict = {'history': y}
    else:
        raise ValueError()
    return model.score(X_test=X_dict, y_test=y, metric=metric)


def compute_permutation_test(model, X_test, y_test, n_perm=100, history=True, metric='corrcoef'):
    """Compare model performace to performance for permuted stimuli.
    If permuting the stimulus does not decrease the model performance, the fit imight be pure autoregression.
    """
    score_trueX = eval_model_score(model=model, X=X_test, y=y_test, stim=True, history=history, metric=metric)

    score_permX = np.full(n_perm, np.nan)
    for i in range(n_perm):
        permX = X_test[np.random.permutation(np.arange(X_test.shape[0])), :, :]
        score_permX[i] = eval_model_score(model=model, X=permX, y=y_test, stim=True, history=history, metric=metric)

    return score_trueX, score_permX


def plot_permutation_test(model, X_test, y_test, metric='corrcoef',
                          n_perm=100, q=99, history=True, ax=None, figsize=None):
    """Plot test results"""
    score_trueX, score_permX = compute_permutation_test(
        model, X_test, y_test, n_perm=n_perm, history=history, metric=metric)

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


def plot3d(model, X_test=None, y_test=None, metric='corrcoef', window=None,
           contour=0.1, pixel_size=30, figsize=None):
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    dt = model.dt
    dims = model.dims
    shift = model.shift

    # rf
    ws = [model.w_opt[name] for name in model.w_opt if 'stimulus' in name]
    n_stimulus_filter = len(ws)
    n_subunits = [w.shape[1] for w in ws]

    ncols = 3
    nrows = sum(n_subunits) + 1
    figsize = figsize if figsize is not None else (8, 8 * nrows / ncols)
    fig = plt.figure(figsize=figsize)

    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    # plot RF and get stats
    stats = {}
    RF_data = {}
    ax_data = {}
    for i, name in enumerate(model.w_opt):

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

            w = uvec(model.w_opt[name])
            vmax = max([w.max(), abs(w.min())])
            n_subunits = w.shape[1]

            for j in range(n_subunits):

                s = w[:, j].reshape(dims[name])
                sRF, tRF = get_spatial_and_temporal_filters(s, model.dims[name])
                ref = [sRF[2:, 2:].max(), sRF[2:, 2:].min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
                max_coord = np.where(sRF == ref)

                tRF = s[:, max_coord[0], max_coord[1]].flatten()
                tRF_max = np.argmax(tRF)
                sRF_max = s[tRF_max]
                tRF_min = np.argmin(tRF)
                sRF_min = s[tRF_min]

                RF_data[name]['sRFs_max'].append(sRF_max)
                RF_data[name]['sRFs_min'].append(sRF_min)
                RF_data[name]['tRFs'].append(tRF)

                ax_sRF_min = fig.add_subplot(spec[i * n_subunits + j, 0])
                ax_data[name]['axes_sRF_min'].append(ax_sRF_min)

                ax_sRF_min.imshow(sRF_min.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, aspect="auto")
                ax_sRF_min.set_xticks([])
                ax_sRF_min.set_yticks([])

                ax_sRF_max = fig.add_subplot(spec[i * n_subunits + j, 1])
                ax_data[name]['axes_sRF_max'].append(ax_sRF_max)
                ax_sRF_max.imshow(sRF_max.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax, aspect="auto")
                ax_sRF_max.set_xticks([])
                ax_sRF_max.set_yticks([])

                ax_tRF = fig.add_subplot(spec[i * n_subunits + j, 2])
                ax_data[name]['axes_tRF'].append(ax_tRF)
                ax_tRF.plot(t_tRF, tRF, color='black')

                ax_tRF.axhline(0, color='gray', linestyle='--')
                ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
                ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
                ax_tRF.spines['top'].set_visible(False)
                ax_tRF.spines['right'].set_visible(False)
                ax_tRF.set_yticks([0])
                ax_tRF.set_ylim(-vmax - 0.01, vmax + 0.01)

                stats[name]['tRF_time_min'].append(t_tRF[tRF_min])
                stats[name]['tRF_time_max'].append(t_tRF[tRF_max])
                stats[name]['tRF_activation_min'].append(float(tRF[tRF_min]))
                stats[name]['tRF_activation_max'].append(float(tRF[tRF_max]))

                stats[name]['tRF_time_diff'].append(np.abs(t_tRF[tRF_max] - t_tRF[tRF_min]))
                stats[name]['tRF_activation_diff'].append(np.abs(tRF[tRF_max] - tRF[tRF_min]))

                if i == 0 and j == 0:
                    ax_sRF_min.set_title('Spatial (min)', fontsize=14)
                    ax_sRF_max.set_title('Spatial (max)', fontsize=14)
                    ax_tRF.set_title('Temporal', fontsize=14)

                if n_subunits > 1:
                    ax_sRF_min.set_ylabel(f'{name} \n Subunits {j}', fontsize=14)
                else:
                    ax_sRF_min.set_ylabel(f'{name}', fontsize=14)

        elif 'history' in name:
            h = uvec(model.w_opt[name])
            t_hRF = np.linspace(-(dims[name][0] + shift[name]) * dt, -shift[name] * dt, dims[name][0] + 1)[1:]
            ax_hRF = fig.add_subplot(spec[-1, 2])
            ax_hRF.plot(t_hRF, h, color='black')
            ax_hRF.spines['top'].set_visible(False)
            ax_hRF.spines['right'].set_visible(False)
            ax_hRF.set_yticks([0])
            ax_hRF.set_title('History', fontsize=14)
    #             ax_hRF.set_ylim(-vmax-0.01, vmax+0.01)

    # contour and more stats

    for name in model.w_opt:

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

            for i in range(n_subunits):

                CS_min = axes_sRF_min[i].contour(sRFs_min[i].T, levels=[-contour], colors=[color_min], linestyles=['-'],
                                                 linewidths=3, alpha=1)
                cntrs_min = [p.vertices for p in CS_min.collections[0].get_paths()]
                cntrs_size_min = [cv2.contourArea(cntr.astype(np.float32)) * pixel_size ** 2 / 1000 for cntr in
                                  cntrs_min]

                axes_sRF_min[i].set_xlabel(f'cntr size = {cntrs_size_min[0]:.03f} 10^3 μm^2')

                CS_max = axes_sRF_max[i].contour(sRFs_max[i].T, levels=[contour], colors=[color_max], linestyles=['-'],
                                                 linewidths=3, alpha=1)
                cntrs_max = [p.vertices for p in CS_max.collections[0].get_paths()]
                cntrs_size_max = [cv2.contourArea(cntr.astype(np.float32)) * pixel_size ** 2 / 1000 for cntr in
                                  cntrs_max]

                axes_sRF_max[i].set_xlabel(f'cntr size = {cntrs_size_max[0]:.03f} 10^3 μm^2')

                contour_size_min += cntrs_size_min
                contour_size_max += cntrs_size_max

                RF_data[name]["sRFs_min_cntr"].append(cntrs_min[0])
                RF_data[name]["sRFs_max_cntr"].append(cntrs_max[0])
                stats[name]['sRF_size_min'].append(cntrs_size_min[0])
                stats[name]['sRF_size_max'].append(cntrs_size_max[0])
                stats[name]['sRF_size_diff'].append(np.abs(cntrs_size_min[0] - cntrs_size_max[0]))

                if n_subunits > 1:
                    for j in np.delete(np.arange(n_subunits), i):
                        axes_sRF_min[i].contour(sRFs_min[j].T, levels=[-contour], colors=['gray'], linestyles=['--'],
                                                alpha=0.4)
                        axes_sRF_max[i].contour(sRFs_max[j].T, levels=[contour], colors=['gray'], linestyles=['--'],
                                                alpha=0.4)

    if X_test is not None:
        if 'history' in model.w_opt:
            ax_pred = fig.add_subplot(spec[-1, :2])
        else:
            ax_pred = fig.add_subplot(spec[-1, :])

        stats[metric], y_pred = model.score(X_test, y_test, metric, return_prediction=True)

        if window is not None:
            n = get_n_samples(window / 60, dt)
        else:
            n = y_test.shape[0]

        t_pred = np.arange(n)
        ax_pred.plot(t_pred * dt, y_test[model.burn_in:][:n], color='black')
        ax_pred.plot(t_pred * dt, y_pred[:n], color='C3', label=f'Predict (cc={stats[metric]:.02f})')
        ax_pred.legend(frameon=False)
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)

    fig.tight_layout()

    return RF_data, stats


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

    vmax = np.max([np.abs(sRFs_max.max()), np.abs(sRFs_max.min()), np.abs(sRFs_min.max()), np.abs(sRFs_min.min())])

    # fig = plt.figure(figsize=(8, 4))

    ncols = num_subunits if num_subunits > 5 else 5
    nrows = 3
    if hasattr(model, 'fnl_fitted'):
        nrows += 1

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows + 1, figure=fig)
    axs = []
    ax_sRF_mins = []
    ax_sRF_maxs = []
    for i in range(num_subunits):
        ax_sRF_min = fig.add_subplot(spec[0, i])
        if hasattr(model, 'w_spl'):
            ax_sRF_min.imshow(sRFs_min[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        else:
            ax_sRF_min.imshow(gaussian_filter(sRFs_min[i].T, sigma=1), cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        ax_sRF_min.set_xticks([])
        ax_sRF_min.set_yticks([])
        ax_sRF_min.set_title(f'S{i}')

        ax_sRF_max = fig.add_subplot(spec[1, i])
        if hasattr(model, 'w_spl'):
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

        if hasattr(model, 'nl_params_opt'):
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

    if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h + 1) * dt, -1 * dt, dims_h + 1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)

        ax_pred = fig.add_subplot(spec[nrows, :-1])

    elif not hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'):

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

    elif hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'):

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

        if hasattr(model, 'nl_params_opt'):

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
