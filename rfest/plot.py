import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .utils import get_n_samples, uvec, get_spatial_and_temporal_filters

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
    figsize = figsize if figsize is not None else (8,  8 * nrows / ncols)
    fig = plt.figure(figsize=figsize)
    
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)     
    
    # plot RF and get stats
    stats = {}
    RF_data = {}
    ax_data = {}
    for i, name in enumerate(model.w_opt): 
            
        if 'stimulus' in name:
            
            RF_data[name] = {

                    "sRFs_min" : [],
                    "sRFs_max" : [],
                    "tRFs" : [],
                    "sRFs_min_cntr": [],
                    "sRFs_max_cntr": [],
            }
            ax_data[name] ={
                    "axes_sRF_min" : [],
                    "axes_sRF_max" : [],
                    "axes_tRF" : [],
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
            
            t_tRF = np.linspace(-(dims[name][0]+shift[name])*dt, -shift[name]*dt, dims[name][0]+1)[1:]
            
            w = uvec(model.w_opt[name])
            vmax = max([w.max(), abs(w.min())])
            n_subunits = w.shape[1]

            for j in range(n_subunits):

                s = w[:, j].reshape(dims[name])
                sRF, tRF = get_spatial_and_temporal_filters(s, model.dims[name])
                ref = [sRF[2:, 2:].max(), sRF[2:, 2:].min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
                max_coord = np.where(sRF == ref)

                tRF = s[:,max_coord[0], max_coord[1]].flatten()
                tRF_max = np.argmax(tRF)
                sRF_max = s[tRF_max]
                tRF_min = np.argmin(tRF)
                sRF_min = s[tRF_min]     
                
                RF_data[name]['sRFs_max'].append(sRF_max)
                RF_data[name]['sRFs_min'].append(sRF_min)
                RF_data[name]['tRFs'].append(tRF)
                
                ax_sRF_min = fig.add_subplot(spec[i * n_subunits  + j, 0])
                ax_data[name]['axes_sRF_min'].append(ax_sRF_min)
                
                ax_sRF_min.imshow(sRF_min.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax , aspect="auto")
                ax_sRF_min.set_xticks([])
                ax_sRF_min.set_yticks([])            
                
                ax_sRF_max = fig.add_subplot(spec[i * n_subunits  + j, 1])
                ax_data[name]['axes_sRF_max'].append(ax_sRF_max)
                ax_sRF_max.imshow(sRF_max.T, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax , aspect="auto")
                ax_sRF_max.set_xticks([])
                ax_sRF_max.set_yticks([])  
                
                ax_tRF = fig.add_subplot(spec[i * n_subunits  + j, 2])
                ax_data[name]['axes_tRF'].append(ax_tRF) 
                ax_tRF.plot(t_tRF, tRF, color='black')

                ax_tRF.axhline(0, color='gray', linestyle='--')
                ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
                ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
                ax_tRF.spines['top'].set_visible(False)
                ax_tRF.spines['right'].set_visible(False)
                ax_tRF.set_yticks([0]) 
                ax_tRF.set_ylim(-vmax-0.01, vmax+0.01)
                
                stats[name]['tRF_time_min'].append(t_tRF[tRF_min])
                stats[name]['tRF_time_max'].append(t_tRF[tRF_max])
                stats[name]['tRF_activation_min'].append(float(tRF[tRF_min]))
                stats[name]['tRF_activation_max'].append(float(tRF[tRF_max]))

                stats[name]['tRF_time_diff'].append(np.abs(t_tRF[tRF_max] - t_tRF[tRF_min]))
                stats[name]['tRF_activation_diff'].append(np.abs(tRF[tRF_max]- tRF[tRF_min]))
                
                if i == 0 and j ==0:
                    ax_sRF_min.set_title('Spatial (min)', fontsize=14)
                    ax_sRF_max.set_title('Spatial (max)', fontsize=14)
                    ax_tRF.set_title('Temporal', fontsize=14)
                    
                if n_subunits > 1:
                    ax_sRF_min.set_ylabel(f'{name} \n Subunits {j}', fontsize=14)
                else:
                    ax_sRF_min.set_ylabel(f'{name}', fontsize=14)
                    
        elif 'history' in name:
            h = uvec(model.w_opt[name])
            t_hRF = np.linspace(-(dims[name][0]+shift[name])*dt, -shift[name]*dt, dims[name][0]+1)[1:]
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

                CS_min = axes_sRF_min[i].contour(sRFs_min[i].T, levels=[-contour], colors=[color_min], linestyles=['-'], linewidths=3, alpha=1)
                cntrs_min = [p.vertices for p in CS_min.collections[0].get_paths()]
                cntrs_size_min = [cv2.contourArea(cntr.astype(np.float32))*pixel_size**2/1000 for cntr in cntrs_min]           

                axes_sRF_min[i].set_xlabel(f'cntr size = {cntrs_size_min[0]:.03f} 10^3 μm^2')

                CS_max = axes_sRF_max[i].contour(sRFs_max[i].T, levels=[ contour],  colors=[color_max], linestyles=['-'], linewidths=3, alpha=1)
                cntrs_max = [p.vertices for p in CS_max.collections[0].get_paths()]
                cntrs_size_max = [cv2.contourArea(cntr.astype(np.float32))*pixel_size**2/1000 for cntr in cntrs_max]           

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

                        axes_sRF_min[i].contour(sRFs_min[j].T, levels=[-contour], colors=['gray'], linestyles=['--'], alpha=0.4)
                        axes_sRF_max[i].contour(sRFs_max[j].T, levels=[ contour], colors=['gray'], linestyles=['--'], alpha=0.4)
                
    if X_test is not None:
        if 'history' in model.w_opt:
            ax_pred = fig.add_subplot(spec[-1, :2])
        else:
            ax_pred = fig.add_subplot(spec[-1, :])
        
        stats[metric], y_pred =  model.score(X_test, y_test, metric, return_prediction=True)
        
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
