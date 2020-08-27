import numpy as np
import matplotlib.pyplot as plt

from .utils import get_n_samples, uvec, get_spatial_and_temporal_filters

def plot1d(models, dt=None, shift=None, model_names=None):

    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]

    dt = models[0].dt if dt is None else dt

    fig, ax = plt.subplots(1, 3, figsize=(12,3))

    for i, model in enumerate(models): 
           
        w = model.w_opt
        dims = model.dims
        shift = 0 if shift is None else -shift
        trange = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]
        ax[0].plot(trange, w, color=f'C{i}', label=f'{model_names[i]}')
        
        if hasattr(model, 'h_opt'):
            h = model.h_opt
            dim_t = len(h)
            trange = np.linspace(-(dims[0]+1)*dt, -1*dt, dims[0]+1)[1:]

            ax[1].plot(trange, h, color=f'C{i}')
            ax[1].set_title('Post-spike Filter')
        else:
            ax[1].axis('off')


        if hasattr(model, 'fnl_fitted'):
            
            nl_params = model.nl_params_opt
            nl_xrange = model.nl_xrange
            fnl_fitted = model.fnl_fitted
            ax[2].plot(nl_xrange, fnl_fitted(nl_params, nl_xrange))
            ax[2].set_title('Fitted nonlinearity')
        else:
            ax[2].axis('off')

    for i in range(3):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('Time (s)')

    ax[0].set_title('RF')
    ax[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0., 1.1))

def plot2d(models):
    pass

def plot3d(model, X_test, y_test, dt=None,
        shift=None, model_name=None, response_type='spike'):
        
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift

    w = uvec(model.w_opt.reshape(dims))
    sRF, tRF = get_spatial_and_temporal_filters(w, dims)
    ref = [sRF.max(), sRF.min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
    max_coord = np.where(sRF == ref)
    tRF = w[:,max_coord[0], max_coord[1]].flatten()
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]
    t_hRF = np.linspace(-(dims[0]+1)*dt, -1*dt, dims[0]+1)[1:]
    
    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(ncols=8, nrows=3, figure=fig)
    
    ax_sRF_min = fig.add_subplot(spec[0, 0:2])
    ax_sRF_max = fig.add_subplot(spec[0, 2:4])
    ax_tRF = fig.add_subplot(spec[0, 4:6])
    ax_hRF = fig.add_subplot(spec[0, 6:])

    vmax = np.max([np.abs(sRF.max()), np.abs(sRF.min())])
    tRF_max = np.argmax(tRF)
    sRF_max = w[tRF_max]
    tRF_min = np.argmin(tRF)
    sRF_min = w[tRF_min]

    ax_sRF_max.imshow(sRF_max, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
    ax_sRF_min.imshow(sRF_min, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
    ax_sRF_min.set_title('Spatial (min)')
    ax_sRF_max.set_title('Spatial (max)')

    ax_tRF.plot(t_tRF, tRF, color='black')
    ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
    ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
    ax_tRF.set_title('Temporal (center)')
    
    if hasattr(model, 'h_opt'):
        ax_hRF.plot(t_hRF, model.h_opt, color='black')
        ax_hRF.set_title('post-spike filter')
    else:
        ax_hRF.axis('off')
        
    ax_pred = fig.add_subplot(spec[1, :])

    y_pred = model.predict(X_test, y_test)
    t_pred = np.arange(300)

    pred_score = model.score(X_test, y_test)

    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
    
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'SPL LG={pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')
    
    ax_cost = fig.add_subplot(spec[2, :4])
    ax_metric = fig.add_subplot(spec[2, 4:])
    
    ax_cost.plot(model.cost_train, color='black', label='train')
    ax_cost.plot(model.cost_dev, color='red', label='dev')
    ax_cost.set_title('cost')
    ax_cost.set_ylabel('MSE')
    ax_cost.legend(frameon=False)

    ax_metric.plot(model.metric_train, color='black', label='train')
    ax_metric.plot(model.metric_dev, color='red', label='dev')
    ax_metric.set_title('metric')
    ax_metric.set_ylabel('corrcoef')
    
    for ax in [ax_sRF_min, ax_sRF_max]:
        ax.set_xticks([])
        ax.set_yticks([])
        
    for ax in [ax_tRF, ax_hRF, ax_metric, ax_cost]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(model_name, fontsize=14)

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
            ax[i].imshow(w[i], cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
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