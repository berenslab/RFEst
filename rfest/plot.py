import numpy as np
import matplotlib.pyplot as plt

from .utils import get_n_samples, uvec, get_spatial_and_temporal_filters

def plot1d(models, X_test, y_test, model_names=None, figsize=None, vmax=0.5, response_type='spike', dt=None, len_time=None):
    
    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    plot_w_spl = any([hasattr(model, 'w_spl') for model in models])
    plot_w_opt = any([hasattr(model, 'w_opt') for model in models])
    plot_nl = any([hasattr(model, 'fnl_fitted') for model in models])
    plot_h_opt = any([hasattr(model, 'h_opt') for model in models])
    
    nrows = len(models) + 1 # add row for prediction
    ncols = 3
    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)   
    
    ax_pred = fig.add_subplot(spec[nrows-1, :])
    dt = models[0].dt if dt is None else dt
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    t_pred = np.arange(n)
    
    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.set_xlabel('Time (s)')
    
    for idx, model in enumerate(models):
                
        dims = model.dims
        ax_w_rf = fig.add_subplot(spec[idx, 0])
        if idx == 0:
            ax_w_rf.set_title('RF', fontsize=14)

        w_sta = uvec(model.w_sta.reshape(dims))
        ax_w_rf.plot(w_sta, color='C0', label='STA')
        ax_w_rf.spines['top'].set_visible(False)
        ax_w_rf.spines['right'].set_visible(False)
        ax_w_rf.set_ylabel(model_names[idx], fontsize=14)
        
        if hasattr(model, 'w_spl'):
            w_spl = uvec(model.w_spl.reshape(dims))
            ax_w_rf.plot(w_spl, color='C1', label='SPL')
                        
        if hasattr(model, 'w_opt'):
            w_opt = uvec(model.w_opt.reshape(dims))
            ax_w_rf.plot(w_opt, color='C2', label='OPT')
        
        if plot_h_opt:
            ax_h_opt = fig.add_subplot(spec[idx, 1])
            
            if hasattr(model, 'h_opt'):
                
                h_opt = model.h_opt
                dims_h = len(model.h_opt)
                t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

                ax_h_opt.plot(t_hRF, h_opt, color='C2')
                ax_h_opt.spines['top'].set_visible(False)
                ax_h_opt.spines['right'].set_visible(False)
            else:
                ax_h_opt.axis('off')
                
            if idx == 0:
                ax_h_opt.set_title('History Filter')
                
        if plot_nl:
            if plot_h_opt:
                ax_nl = fig.add_subplot(spec[idx, 2])
            else:
                ax_nl = fig.add_subplot(spec[idx, 1])

            if hasattr(model, 'fnl_fitted'):
                
                nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)                
                nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
                xrng = model.nl_xrange
                ax_nl.plot(xrng, nl0, color='black', label='init')
                ax_nl.plot(xrng, nl_opt, color='red', label='fitted')
                ax_nl.spines['top'].set_visible(False)
                ax_nl.spines['right'].set_visible(False)
            else:
                ax_nl.axis('off')
                
            if idx == 0:
                ax_nl.set_title('Fitted nonlinearity')

            ax_nl.legend(frameon=False)
                
        y_pred = model.predict(X_test, y_test)
        pred_score = model.score(X_test, y_test)
    
        ax_pred.plot(t_pred * dt, y_pred[t_pred], color=f'C{idx}', linewidth=2,
            label=f'{model_names[idx]} = {pred_score:.3f}')
        ax_pred.legend(frameon=False)
        ax_w_rf.legend(frameon=False)
        

    fig.tight_layout()   

def plot2d(models, X_test=None, y_test=None, model_names=None, figsize=None, vmax=0.5, response_type='spike', dt=None, len_time=None):
    
    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    # plot_w_spl = any([hasattr(model, 'w_spl') for model in models])
    plot_w_opt = any([hasattr(model, 'w_opt') for model in models])
    plot_nl = any([hasattr(model, 'fnl_fitted') for model in models])
    plot_h_opt = any([hasattr(model, 'h_opt') for model in models])

    
    nrows = len(models) + 1 # add row for prediction
    ncols = 1 + sum([1, plot_w_opt, plot_nl, plot_h_opt])
    figsize = figsize if figsize is not None else (2 * ncols, nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)   
    
    if X_test is not None:
        ax_pred = fig.add_subplot(spec[nrows-1, :])
        dt = models[0].dt if dt is None else dt
        if len_time is not None: 
            n = get_n_samples(len_time / 60, dt)
        else:
            n = y_test.shape[0]

        t_pred = np.arange(n)
        
        if response_type == 'spike':
            markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                                markerfmt='none', use_line_collection=True, label=f'{response_type}')
            markerline.set_markerfacecolor('none')
            plt.setp(baseline,'color', 'none')
        else:
            ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.set_xlabel('Time (s)')
    
    for idx, model in enumerate(models):
                
        dims = model.dims
        ax_w_sta = fig.add_subplot(spec[idx, 0])
        w_sta = uvec(model.w_sta.reshape(dims))
        ax_w_sta.imshow(w_sta, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
        ax_w_sta.set_xticks([])
        ax_w_sta.set_yticks([])
        ax_w_sta.set_ylabel(model_names[idx], fontsize=14)
        
        if hasattr(model, 'w_spl'):
            ax_w_spl = fig.add_subplot(spec[idx, 1])
            w_spl = uvec(model.w_spl.reshape(dims))
            ax_w_spl.imshow(w_spl, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
            ax_w_spl.set_xticks([])
            ax_w_spl.set_yticks([])
        
        if idx == 0:
            ax_w_sta.set_title('STA', fontsize=14)
            if hasattr(model, 'w_spl'): 
                ax_w_spl.set_title('SPL', fontsize=14)    
                
        if hasattr(model, 'w_opt'):
            ax_w_opt = fig.add_subplot(spec[idx, 2])
            w_opt = uvec(model.w_opt.reshape(dims))
            ax_w_opt.imshow(w_opt, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
            ax_w_opt.set_xticks([])
            ax_w_opt.set_yticks([])
            if idx == 0:
                ax_w_opt.set_title('OPT', fontsize=14)
        
        if plot_h_opt:
            ax_h_opt = fig.add_subplot(spec[idx, 3])
            
            if hasattr(model, 'h_opt'):
                
                h_opt = model.h_opt
                ax_h_opt.plot(h_opt)
                ax_h_opt.spines['top'].set_visible(False)
                ax_h_opt.spines['right'].set_visible(False)
            else:
                ax_h_opt.axis('off')
                
            if idx == 0:
                ax_h_opt.set_title('History Filter')
                
        if plot_nl:
            if plot_h_opt:
                ax_nl = fig.add_subplot(spec[idx, 4])
            else:
                ax_nl = fig.add_subplot(spec[idx, 3])

            if hasattr(model, 'fnl_fitted'):
                
                xrng = model.nl_xrange
                nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
                ax_nl.plot(xrng, nl0)

                if hasattr(model, 'nl_params_opt'):
                    nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
                    ax_nl.plot(xrng, nl_opt)

                ax_nl.spines['top'].set_visible(False)
                ax_nl.spines['right'].set_visible(False)

            else:
                ax_nl.axis('off')
        

            if idx == 0:
                ax_nl.set_title('Fitted nonlinearity')
            
        if X_test is not None:
            y_pred = model.predict(X_test, y_test)
            pred_score = model.score(X_test, y_test)
        
            ax_pred.plot(t_pred * dt, y_pred[t_pred], color=f'C{idx}', linewidth=2,
                label=f'{model_names[idx]} = {pred_score:.3f}')
            ax_pred.legend(frameon=False)

    fig.tight_layout()
    
def plot3d(model, X_test=None, y_test=None, dt=None, 
        shift=None,  response_type='spike', len_time=None, figsize=None, 
        plot_extra=False, model_name=None, title=None):
        
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift

    if hasattr(model, 'w_opt'):
        w = uvec(model.w_opt.reshape(dims))
    elif hasattr(model, 'w_spl'):
        w = uvec(model.w_spl.reshape(dims))
    elif hasattr(model, 'w_mle'):
        w = uvec(model.w_mle.reshape(dims))
    else:
        w = uvec(model.w_sta.reshape(dims))

    vmax = np.max([np.abs(w.max()), np.abs(w.min())])

    sRF, tRF = get_spatial_and_temporal_filters(w, dims)
    ref = [sRF.max(), sRF.min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
    max_coord = np.where(sRF == ref)
    tRF = w[:,max_coord[0], max_coord[1]].flatten()
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]

    # fig = plt.figure(figsize=(8, 6))
    
    figsize = figsize if figsize is not None else (8, 6)
    fig = plt.figure(figsize=figsize)
    if plot_extra:
        spec = gridspec.GridSpec(ncols=8, nrows=3, figure=fig)
    else:
        spec = gridspec.GridSpec(ncols=8, nrows=2, figure=fig)
    
    ax_sRF_min = fig.add_subplot(spec[0, 0:2])
    ax_sRF_max = fig.add_subplot(spec[0, 2:4])
    ax_tRF = fig.add_subplot(spec[0, 4:6])
    ax_hRF = fig.add_subplot(spec[0, 6:])

    tRF_max = np.argmax(tRF)
    sRF_max = w[tRF_max]
    tRF_min = np.argmin(tRF)
    sRF_min = w[tRF_min]

    ax_sRF_max.imshow(sRF_max.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
    ax_sRF_min.imshow(sRF_min.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
    ax_sRF_min.set_title('Spatial filter (min)')
    ax_sRF_max.set_title('Spatial filter (max)')

    ax_tRF.plot(t_tRF, tRF, color='black', lw=3, alpha=0.8)
    ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
    ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
    ax_tRF.set_title('Temporal filter (center)')
    ax_tRF.spines['top'].set_visible(False)
    ax_tRF.spines['right'].set_visible(False)
    ax_tRF.set_xlabel('Time (s)')

    if hasattr(model, 'h_opt'):
        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]
        ax_hRF.plot(t_hRF, model.h_opt, color='black', lw=3, alpha=0.8)
        ax_hRF.plot(t_hRF, model.Sh * model.bh_opt, color='gray', alpha=0.5) 
        ax_hRF.set_title('Response-history filter')
        ax_hRF.spines['top'].set_visible(False)
        ax_hRF.spines['right'].set_visible(False)
        ax_hRF.set_xlabel('Time (s)')
    else:
        ax_hRF.axis('off')
    
    if X_test is not None:

        if hasattr(model, 'fnl_fitted') and not hasattr(model, 'nl_params_opt'):
            ax_nl = fig.add_subplot(spec[1, -2:])
            xrng = model.nl_xrange
            nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
            ax_nl.plot(xrng, nl0, lw=4, alpha=0.5)
            ax_nl.plot(xrng, model.nl_basis * model.nl_params, color='grey', alpha=0.5) 

            # if hasattr(model, 'nl_params_opt'):
        elif hasattr(model, 'nl_params_opt'): 
            ax_nl = fig.add_subplot(spec[1, -2:])
            xrng = model.nl_xrange
            nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
            ax_nl.plot(xrng, nl_opt, lw=4, alpha=0.5)
            ax_nl.plot(xrng, model.nl_basis * model.nl_params_opt, color='grey', alpha=0.5) 
            
            ax_nl.set_title('Fitted nonlinearity')
            ax_nl.spines['top'].set_visible(False)
            ax_nl.spines['right'].set_visible(False) 
            ax_nl.set_xlabel('Filter output')
            ax_pred = fig.add_subplot(spec[1, :-2])

        else:
            ax_pred = fig.add_subplot(spec[1, :])

        y_pred = model.predict(X_test, y_test)

        if len_time is not None: 
            n = get_n_samples(len_time / 60, dt)
        else:
            n = y_test.shape[0]

        t_pred = np.arange(n)

        pred_score = model.score(X_test, y_test)

        if response_type == 'spike':
            markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                                markerfmt='none', use_line_collection=True, label=f'{response_type}')
            markerline.set_markerfacecolor('none')
            plt.setp(baseline,'color', 'none')
        else:
            ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
        
        ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{model_name}={pred_score:.3f}')
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        ax_pred.set_xlabel('Time (s)')
        ax_pred.legend(loc="upper left" , frameon=False)
        ax_pred.set_title('Prediction performance')
        
    if plot_extra:

        ax_cost = fig.add_subplot(spec[2, :4])
        ax_metric = fig.add_subplot(spec[2, 4:])
        
        ax_cost.plot(model.cost_train, color='black', label='train')
        ax_cost.plot(model.cost_dev, color='red', label='dev')
        ax_cost.set_title('cost')

        if 'LG' in model_name:
            ax_cost.set_ylabel('MSE')
        else:
            ax_cost.set_ylabel('nLL')

        ax_cost.legend(frameon=False)

        ax_metric.plot(model.metric_train, color='black', label='train')
        ax_metric.plot(model.metric_dev, color='red', label='dev')
        ax_metric.set_title('metric')
        ax_metric.set_ylabel('corrcoef')

        for ax in [ax_metric, ax_cost]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    for ax in [ax_sRF_min, ax_sRF_max]:
        ax.set_xticks([])
        ax.set_yticks([])
                
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title, fontsize=14)

    return fig

def plot3d_frames(model, shift=None):
    
    dims = model.dims
    
    dt = model.dt
    nt = dims[0] # number of time frames
    ns = model.n_s if hasattr(model, 'n_s') else 1# number of subunits
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(nt-shift)*dt, shift*dt, nt+1)[1:]    
    
    fig, ax = plt.subplots(ns, nt, figsize=(1.5 * nt, 2*ns))
    if ns == 1:
        w = uvec(model.w_opt.reshape(dims))
        vmax = np.max([np.abs(w.max()), np.abs(w.min())])

        for i in range(nt):
            ax[i].imshow(w[i].T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f'{t_tRF[i]:.3f} s', fontsize=18)
    else:
        for k in range(ns):
            w = uvec(model.w_opt[:, k].reshape(dims))
            vmax = np.max([np.abs(w.max()), np.abs(w.min())])
            for i in range(nt):
                ax[k, i].imshow(w[i], cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
                ax[k, i].set_xticks([])
                ax[k, i].set_yticks([]) 
                ax[0, i].set_title(f'{t_tRF[i]:.3f} s', fontsize=18)
            ax[k, 0].set_ylabel(f'S{k}', fontsize=18)

    fig.tight_layout()
        
def plot_prediction(models, X_test, y_test, dt=None, len_time=None, 
                    response_type='spike', model_names=None):
    
    """
    Parameters
    ==========

    models : a model object or a list of models
        List of models. 

    X_test, y_test : array_likes
        Test set.

    dt : float
        Stimulus frame rate in second.

    length : None or float
        Length of y_test to display (in second). 
        If it's None, then use the whole y_test.

    response_type : str
        Plot y_test as `spike` or others.
    """

    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]

    dt = models[0].dt if dt is None else dt
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    trange = np.arange(n)

    fig, ax = plt.subplots(figsize=(12,3))

    if response_type == 'spike':
        markerline, stemlines, baseline = ax.stem(trange * dt, y_test[trange], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax.plot(trange * dt, y_test[trange], color='black', label=f'{response_type}')
    
    for i, model in enumerate(models):
        
        y_pred = model.predict(X_test, y_test)
        pred_score = model.score(X_test, y_test)
    
        ax.plot(trange * dt, y_pred[trange], color=f'C{i}', linewidth=2,
            label=f'{model_names[i]} = {pred_score:.3f}')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax.tick_params(axis='y', colors='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.legend(loc="upper right" , frameon=False, bbox_to_anchor=(1., 1.))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Prediction - Pearson\'s r', fontsize=14)

def plot_learning_curves(models, model_names=None):

    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]    

    len_iters = []
    fig, ax = plt.subplots(len(models),2, figsize=(8, len(models)*2))
    ax = ax.reshape(len(models), 2)

    for i, model in enumerate(models):
        
        ax[i, 0].plot(model.cost_train, label='train', color='black', linewidth=3)
        ax[i, 0].plot(model.cost_dev, label='dev', color='red', linewidth=3)
        

        ax[i, 1].plot(model.metric_train, label='train', color='black', linewidth=3)
        ax[i, 1].plot(model.metric_dev, label='dev', color='red', linewidth=3)
        if i < len(models)-1:
            ax[i, 0].set_xticks([])
            ax[i, 1].set_xticks([])

        len_iters.append(len(model.metric_train))
            

        ax[i, 1].set_ylim(0, 1) 

        ax[i, 0].set_ylabel(f'{model_names[i]}', fontsize=14)
        ax[i, 0].set_yticks([])

        ax[i, 0].spines['top'].set_visible(False)
        ax[i, 0].spines['right'].set_visible(False)
        ax[i, 1].spines['top'].set_visible(False)
        ax[i, 1].spines['right'].set_visible(False)

    for i, model in enumerate(models):
        ax[i, 0].set_xlim(-100, max(len_iters))
        ax[i, 1].set_xlim(-100, max(len_iters))
        
    ax[0, 0].set_title('Cost')
    ax[0, 1].set_title('Performance')
        
    ax[-1, 0].set_xlabel('Iteration')
    ax[-1, 1].set_xlabel('Iteration')
    
    ax[0, 0].legend(frameon=False)
    
    fig.tight_layout()

def plot_subunits2d(model, X_test, y_test, dt=None, shift=None, model_name=None, response_type='spike', len_time=30, ncols=5, figsize=None):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 
    
    ws = uvec(model.w_opt)
    dims = model.dims
    num_subunits = ws.shape[1]
    
    vmax = np.max([np.abs(ws.max()), np.abs(ws.min())])
    t_hRF = np.linspace(-(dims[0]+1)*dt, -1*dt, dims[0]+1)[1:]
    
    nrows = np.ceil(num_subunits/ncols).astype(int)
    if num_subunits % ncols != 0:
        num_left = ncols - num_subunits % ncols
    else:
        num_left = 0

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    axs = []
    
    for j in range(nrows):
        for i in range(ncols):
            ax_subunits = fig.add_subplot(spec[j, i])
            axs.append(ax_subunits)
            
    for i in range(num_subunits):
        w = ws[:, i].reshape(dims)
        axs[i].imshow(w, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    else:
        if num_left > 0:
            for j in range(1, num_left+1):
                axs[i+j].axis('off')
            
    if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif not hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'): 
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        xrng = model.nl_xrange
        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
        ax_nl.plot(xrng, nl0)

        if hasattr(model, 'nl_params_opt'):
            nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
            ax_nl.plot(xrng, nl_opt)
        
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'):
        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]
        ax_h_opt = fig.add_subplot(spec[nrows, -2])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)    
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        xrng = model.nl_xrange
        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
        ax_nl.plot(xrng, nl0)

        if hasattr(model, 'nl_params_opt'):
            nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
            ax_nl.plot(xrng, nl_opt)

        ax_nl.set_title('Fitted nonlinearity')
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
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')    
        
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')
    
    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')

    fig.tight_layout()

def plot_subunits3d(model, X_test, y_test, dt=None, shift=None, model_name=None, response_type='spike', len_time=1, contour=None, figsize=None):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]

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
        tRF = w[:,max_coord[0], max_coord[1]].flatten()
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

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    axs = []
    ax_sRF_mins= []
    ax_sRF_maxs = []
    for i in range(num_subunits):
        ax_sRF_min = fig.add_subplot(spec[0, i])       
        ax_sRF_min.imshow(sRFs_min[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        ax_sRF_min.set_xticks([])
        ax_sRF_min.set_yticks([])
        ax_sRF_min.set_title(f'S{i}')

        ax_sRF_max = fig.add_subplot(spec[1, i])       
        ax_sRF_max.imshow(sRFs_max[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
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

        ax_sRF_mins.append(ax_sRF_min)
        ax_sRF_maxs.append(ax_sRF_max)

    if contour is not None: # then plot contour
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
                ax_sRF_mins[i].contour(sRFs_min[j].T, levels=[-contour], colors=[color], linestyles=[style], alpha=alpha)
                ax_sRF_maxs[i].contour(sRFs_max[j].T, levels=[contour], colors=[color], linestyles=[style], alpha=alpha)
            
    if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif not hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'): 
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        nl = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
        xrng = model.nl_xrange
        
        ax_nl.plot(xrng, nl)
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -2])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)    
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        xrng = model.nl_xrange
        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
        ax_nl.plot(xrng, nl0)

        if hasattr(model, 'nl_params_opt'):
            nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
            ax_nl.plot(xrng, nl_opt)

        ax_nl.set_title('Fitted nonlinearity')
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
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')    
        
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')    
        
    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'{model_name}', fontsize=14)

def plot_multicolors3d(model, X_test, y_test, dt=None, shift=None, response_type='spike', len_time=1,
                       model_name=None, cmaps=None, figsize=None):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]

    ws = uvec(model.w_opt)
    
    num_colors = model.n_c
    cmaps = cmaps if cmaps is not None else [plt.cm.bwr for _ in range(num_colors)]
    
    sRFs = []
    tRFs = []
    for i in range(num_colors):
        sRF, tRF = get_spatial_and_temporal_filters(ws[:, i], dims)
        sRFs.append(sRF)
        tRFs.append(tRF)
    
    sRFs = np.stack(sRFs)
    
    vmax = np.max([np.abs(sRFs.max()), np.abs(sRFs.min())])

    
    ncols = num_colors if num_colors > 4 else 4    
    nrows = 2
    
    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    axs = []
    
    for i in range(num_colors):
        ax_sRF = fig.add_subplot(spec[0, i])       
        ax_sRF.imshow(sRFs[i].T, cmap=cmaps[i], vmax=vmax, vmin=-vmax)
        ax_sRF.set_xticks([])
        ax_sRF.set_yticks([])
#         ax_sRF.set_title(f'S{i}')
    
        ax_tRF = fig.add_subplot(spec[1, i])       
        ax_tRF.plot(t_tRF, tRFs[i], color='black')
        ax_tRF.spines['top'].set_visible(False)
        ax_tRF.spines['right'].set_visible(False)        
        
    if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif not hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'): 
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        nl = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
        xrng = model.nl_xrange
        
        ax_nl.plot(xrng, nl)
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'):

        dims_h = len(model.h_opt)
        t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

        ax_h_opt = fig.add_subplot(spec[nrows, -2])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)    
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        
        xrng = model.nl_xrange

        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
        ax_nl.plot(xrng, nl0)

        if hasattr(model, 'nl_params_opt'):
            nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
            ax_nl.plot(xrng, nl_opt)
        
        ax_nl.set_title('Fitted nonlinearity')
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
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')    
        
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')    
        
    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')
        
    fig.tight_layout()

def plot_stc2d(model, figsize=(8, 4), cmap=plt.cm.bwr):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    dims = model.dims
    
    n_w_stc_pos = model.w_stc_pos.shape[1]
    n_w_stc_neg = model.w_stc_neg.shape[1]
    ncols = np.max([n_w_stc_pos, n_w_stc_neg])
    nrows = 3
    
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    
    if n_w_stc_pos > 0: 
        w_stc_pos = uvec(model.w_stc_pos)
        vmax = np.max([np.abs(w_stc_pos.max()), np.abs(w_stc_pos.min())])
        for i in range(n_w_stc_pos):
            w = w_stc_pos[:, i].reshape(dims)
            ax_w_stc_pos = fig.add_subplot(spec[-2, i])
            ax_w_stc_pos.imshow(w, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax_w_stc_pos.set_xticks([])
            ax_w_stc_pos.set_yticks([])
            if i == 0:
                ax_w_stc_pos.set_ylabel('Pos. Filters')

    if n_w_stc_neg > 0:
        w_stc_neg = uvec(model.w_stc_neg)
        vmax = np.max([np.abs(w_stc_neg.max()), np.abs(w_stc_neg.min())])
        for i in range(n_w_stc_neg):
            w = w_stc_neg[:, i].reshape(dims)
            ax_w_stc_neg = fig.add_subplot(spec[-1, i])
            ax_w_stc_neg.imshow(w, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
            ax_w_stc_neg.set_xticks([])
            ax_w_stc_neg.set_yticks([])
            if i == 0:
                ax_w_stc_neg.set_ylabel('Neg. Filters')
            
    ax_eigen = fig.add_subplot(spec[:2, :])
    xx = np.arange(len(model.w_stc_eigval))
    mask = model.w_stc_eigval_mask
    ax_eigen.axhspan(model.w_stc_min_null, model.w_stc_max_null, alpha=0.5, color='grey', label='null')
    ax_eigen.plot(xx, model.w_stc_eigval, 'k.')
    ax_eigen.plot(xx[mask], model.w_stc_eigval[mask], 'r.')
    ax_eigen.set_title('Eigen Values')
    ax_eigen.legend(frameon=False)
    
    fig.tight_layout()

def plot_nonlinearity(model, others=None):
    
    if not hasattr(model, 'nl_bins'):
        raise ValueError('Nonlinearity is not fitted yet.')
    
    fig, ax = plt.subplots(figsize=(4,3))
    if hasattr(model, 'fnl_nonparametric'):
        ax.plot(model.nl_bins, model.fnl_nonparametric(model.nl_bins), label='nonparametric')
        
    if hasattr(model, 'fnl_nonparametric'):
        ax.plot(model.nl_bins, model.fnl_fitted(model.nl_params, model.nl_bins), label='parametric')
        
    if others is not None:
        for nl in others:
            ax.plot(model.nl_bins, model.fnl(model.nl_bins, nl=nl), label=nl)
     
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Fitted Nonlinearities')
    ax.legend(frameon=False)

def compare_LNP_and_LNLN(lnp, lnln, X_test, y_test, dt=None, shift=None, title=None, response_type='spike', len_time=1, contour=None, figsize=None):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    dims = lnln.dims
    dt = lnln.dt if dt is None else dt
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]

    ws = uvec(lnln.w_opt)
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
    
    fig = plt.figure(figsize=(8, 4))
    
    ncols = num_subunits if num_subunits > 5 else 5 
    ncols += 1 # add lnp
    nrows = 3

    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    axs = []
    ax_sRF_mins= []
    ax_sRF_maxs = []    
   
    # LNP
    w_lnp = uvec(lnp.w_opt).reshape(dims)
    vmax_lnp = np.max([np.abs(w_lnp.max()), np.abs(w_lnp.min())])
    sRF_lnp, tRF_lnp = get_spatial_and_temporal_filters(w_lnp, dims)
    ref = [sRF_lnp[2:, 2:].max(), sRF_lnp[2:, 2:].min()][np.argmax([np.abs(sRF_lnp.max()), np.abs(sRF_lnp.min())])]
    max_coord = np.where(sRF_lnp == ref)
    tRF_lnp = w_lnp[:, max_coord[0], max_coord[1]].flatten()
    tRF_max = np.argmax(tRF_lnp)
    sRF_max = w_lnp[tRF_max]
    tRF_min = np.argmin(tRF_lnp)
    sRF_min = w_lnp[tRF_min]
        
    ax_sRF_min = fig.add_subplot(spec[0, 0])       
    ax_sRF_min.imshow(sRF_min.T, cmap=plt.cm.bwr, vmax=vmax_lnp, vmin=-vmax_lnp)
    ax_sRF_min.set_xticks([])
    ax_sRF_min.set_yticks([])
    ax_sRF_min.set_title(f'LNP')

    ax_sRF_max = fig.add_subplot(spec[1, 0])       
    ax_sRF_max.imshow(sRF_max.T, cmap=plt.cm.bwr, vmax=vmax_lnp, vmin=-vmax_lnp)
    ax_sRF_max.set_xticks([])
    ax_sRF_max.set_yticks([])

    ax_tRF = fig.add_subplot(spec[2, 0])       
    ax_tRF.plot(t_tRF, tRFs[i], color='black')
    ax_tRF.spines['top'].set_visible(False)
    ax_tRF.spines['right'].set_visible(False)
    tRF_max = np.argmax(tRFs[i])
    tRF_min = np.argmin(tRFs[i])
    ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
    ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)

    ax_sRF_min.set_ylabel('Min Frame')
    ax_sRF_max.set_ylabel('Max Frame')   
    
    ax_sRF_mins.append(ax_sRF_min)
    ax_sRF_maxs.append(ax_sRF_max)     

    # LNLN subunits

    for i in range(num_subunits):
        ax_sRF_min = fig.add_subplot(spec[0, i+1])       
        ax_sRF_min.imshow(sRFs_min[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        ax_sRF_min.set_xticks([])
        ax_sRF_min.set_yticks([])
        ax_sRF_min.set_title(f'S{i}')

        ax_sRF_max = fig.add_subplot(spec[1, i+1])       
        ax_sRF_max.imshow(sRFs_max[i].T, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        ax_sRF_max.set_xticks([])
        ax_sRF_max.set_yticks([])
    
        ax_tRF = fig.add_subplot(spec[2, i+1])       
        ax_tRF.plot(t_tRF, tRFs[i], color='black')
        ax_tRF.spines['top'].set_visible(False)
        ax_tRF.spines['right'].set_visible(False)
        tRF_max = np.argmax(tRFs[i])
        tRF_min = np.argmin(tRFs[i])
        ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
        ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
        
        ax_sRF_mins.append(ax_sRF_min)
        ax_sRF_maxs.append(ax_sRF_max)
        
    if contour is not None: # then plot contour

        for i in range(num_subunits+1):
            
            color_min = 'black' if i == 0 else 'lightsteelblue'
            color_max = 'black' if i == 0 else 'lightcoral'

            ax_sRF_mins[i].contour(sRF_min.T, levels=[-contour], colors=[color_min], linestyles=['-'], alpha=1)
            ax_sRF_maxs[i].contour(sRF_max.T, levels=[contour], colors=[color_max], linestyles=['-'], alpha=1)

            for j in range(num_subunits):
                if i-1 != j:
                    color = 'grey'
                    alpha = 0.5
                    style = '--'                    
                else:
                    color = 'black'
                    alpha = 1
                    style = '--'
                ax_sRF_mins[i].contour(sRFs_min[j].T, levels=[-contour], colors=[color], linestyles=[style], alpha=alpha)
                ax_sRF_maxs[i].contour(sRFs_max[j].T, levels=[contour], colors=[color], linestyles=[style], alpha=alpha)
            
    
    for counter, model in enumerate([lnp, lnln]):
        color = 'C3' if counter ==0 else 'C0'
        model_name = 'LNP' if counter ==0 else 'LNLN'
        if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):

            dims_h = len(model.h_opt)
            t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

            ax_h_opt = fig.add_subplot(spec[nrows, -1])
            ax_h_opt.plot(t_hRF, model.h_opt, color=color, label=model_name)
            ax_h_opt.set_title('History Filter')
            ax_h_opt.spines['top'].set_visible(False)
            ax_h_opt.spines['right'].set_visible(False)

            ax_pred = fig.add_subplot(spec[nrows, :-1])

        elif not hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'): 

            ax_nl = fig.add_subplot(spec[nrows, -1])
            nl = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
            xrng = model.nl_xrange

            ax_nl.plot(xrng, nl)
            ax_nl.set_title('Fitted nonlinearity')
            ax_nl.spines['top'].set_visible(False)
            ax_nl.spines['right'].set_visible(False)    

            ax_pred = fig.add_subplot(spec[nrows, :-1])

        elif hasattr(model, 'h_opt') and hasattr(model, 'fnl_fitted'):

            dims_h = len(model.h_opt)
            t_hRF = np.linspace(-(dims_h+1)*dt, -1*dt, dims_h+1)[1:]

            ax_h_opt = fig.add_subplot(spec[nrows, -2])
            ax_h_opt.plot(t_hRF, model.h_opt, color='black')
            ax_h_opt.set_title('History Filter')
            ax_h_opt.spines['top'].set_visible(False)
            ax_h_opt.spines['right'].set_visible(False)    

            ax_nl = fig.add_subplot(spec[nrows, -1])
            xrng = model.nl_xrange
            nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)     
            ax_nl.plot(xrng, nl0)

            if hasattr(model, 'nl_params_opt'):
                nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
                ax_nl.plot(xrng, nl_opt)

            ax_nl.set_title('Fitted nonlinearity')
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
        
        ax_pred.plot(t_pred * dt, y_pred[t_pred], color=color, linewidth=3, label=f'{model_name}={pred_score:.3f}')

    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')    

    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')    
        
    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')

    if hasattr(model, 'h_opt'):
        ax_h_opt.legend(loc="upper left" , frameon=False)
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title is not None:
        fig.suptitle(f'{title}', fontsize=14) 
    
    return fig
    