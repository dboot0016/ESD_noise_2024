# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure S18a, c, e for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024
# Uses data from 'metrics_model_ep.py'

#%% Load in modules
import xarray as xr 
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import pandas as pd
import test

#%% Functions used in the script
def std(data1):  
    # Determine weighted standard deviation
    latitudes = data1['latitude']
    weights = np.cos(np.deg2rad(latitudes))*xr.ones_like(data1)
    weights = weights / weights.sum()*A_mask_era  
    
    weighted_mean_2 = np.average(data1.fillna(0), weights=weights.fillna(0))
    weighted_variance_2 = np.average((data1.fillna(0) - weighted_mean_2) ** 2, weights=weights.fillna(0))
    weighted_std_2 = np.sqrt(weighted_variance_2)
    
    return weighted_std_2

#%% Load in noise
# Directories where ERA5 and CMIP6 data is + where figures should be saved to
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_cmip = '/Users/Boot0016/Documents/Project_noise_model/CMIP6_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Figures_sub/'
dir_test ='/Users/Boot0016/Documents/Project_noise_model/Test/' 

#%%
# Load in the mask of the Atlantic
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# Load in data
var = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise_era5 = var['__xarray_dataarray_variable__'].compute()*1e3

# Determine statistics of ERA5 noise
noise_mean_e5 = noise_era5.mean('time')
noise_std_e5 = noise_era5.std('time')
noise_skw_e5 = noise_era5.reduce(func=scipy.stats.skew,dim='time')
noise_kur_e5 = noise_era5.reduce(func=scipy.stats.kurtosis,dim='time')

# Determine weighted standard deviation of statistics
std_noise_mean_E5 = std(noise_mean_e5)
std_noise_std_E5 = std(noise_std_e5)
std_noise_skw_E5 = std(noise_skw_e5)
std_noise_kur_E5 = std(noise_kur_e5)

#%% Load in metrics noise
noise_metrics = np.array(pd.read_csv(dir_test+'metrics_noise_ep_noise_models.csv'))
MMM_metrics = np.array(pd.read_csv(dir_cmip+'metrics_noise_MMM_ep_1deg.csv'))

# Select spatial correlation of statistics noise models
noise_corr_mean = noise_metrics[0,1:]
noise_corr_std = noise_metrics[1,1:]
noise_corr_skw = noise_metrics[2,1:]
noise_corr_kur = noise_metrics[3,1:]

# Select weighted standard deviation of statistics noise models
noise_std_mean = noise_metrics[16,1:]
noise_std_std = noise_metrics[17,1:]
noise_std_skw = noise_metrics[18,1:]
noise_std_kur = noise_metrics[19,1:]

#%% Plotting variables
colors = 'red','blue','black'   # Colors used for markers
FS = 20                         # Base font size
quality = 300                   # Quality of figure in dpi

models = np.arange(1,40,1)      # Markers for CMIP6 models
noises = (['a','b','c','d','e','f','g','h','i'])    # Markers for noise models

#%% Plot figures
# !! NOTE: FIRST RUN taylorDiagram.py TO BE ABLE TO PLOT THESE FIGURES !!
# Figure 8a
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_std_E5, fig=fig,
                    label='Reference',srange=(0,1.5))

# Plot noise models
for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2 or noise_i == 4:
        ms = 15*0.9
    else:
        ms = 20*0.9
    dia.add_sample(noise_std_std[noise_i], noise_corr_std[noise_i],
               marker="$"+str(noises[noise_i])+"$", ms=ms, ls='',
               mfc=colors[1], mec=colors[1],
               label='test')
    
# Add RMS contours, and label them
plt.plot([0,0.2],[0,1.5*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.425],[0,1.5*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.73],[0,1.5*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,1.3],[0,1.5*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,3.35],[0,2.5*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,6.9],[0,1.5*std_noise_std_E5],color = '0.5',linestyle = 'dotted')

contours = dia.add_contours(levels=5, colors='0.35') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=16, fmt='%.1f')
dia._ax.set_title('E-P: standard deviation',fontsize=FS)

fig.tight_layout()

plt.savefig(dir_fig+'Figure_S18_a.png', format='png', dpi=quality,bbox_inches='tight')

# Figure 8c
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_skw_E5, fig=fig,
                    label='Reference',srange = (0,2))

# Plot noise models
for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2 or noise_i == 4:
        ms = 15*0.9
    else:
        ms = 20*0.9
    dia.add_sample(noise_std_skw[noise_i], noise_corr_skw[noise_i],
               marker="$"+str(noises[noise_i])+"$", ms=ms, ls='',
               mfc=colors[1], mec=colors[1],
               label='test')
    
# Add RMS contours, and label them
plt.plot([0,0.24],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.515],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.88],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,1.57],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,2.44],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,8.2],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')

contours = dia.add_contours(levels=5, colors='0.35') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=16, fmt='%.1f')
dia._ax.set_title('E-P: skewness',fontsize=FS)

fig.tight_layout()

plt.savefig(dir_fig+'Figure_S18_c.png', format='png', dpi=quality,bbox_inches='tight')

# Figure 8e
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_kur_E5, fig=fig,
                    label='Reference',srange = (0,2))

# Plot noise models
for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2 or noise_i == 4:
        ms = 15*0.9
    else:
        ms = 20*0.9
    dia.add_sample(noise_std_kur[noise_i], noise_corr_kur[noise_i],
               marker="$"+str(noises[noise_i])+"$", ms=ms, ls='',
               mfc=colors[1], mec=colors[1],
               label='test')
    
# Add RMS contours, and label them
plt.plot([0,1.72],[0,2*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,3.7],[0,2*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,6.3],[0,2*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,11.2],[0,2*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,17.5],[0,2*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,59],[0,2*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')

contours = dia.add_contours(levels=5, colors='0.35') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=16, fmt='%.1f')
dia._ax.set_title('E-P: kurtosis',fontsize=FS)

fig.tight_layout()

plt.savefig(dir_fig+'Figure_S18_e.png', format='png', dpi=quality,bbox_inches='tight')

#%% List of CMIP6 models
key_list = np.array(['Generalized Pareto','Johnson SU','Skew-normal','Generalized hyperbolic','Generalized extreme','Gamma','Beta','Normal','NIG'])
model_names = 'ERA5'

# Select model name from key_list
for model_i in range(len(key_list)):
    key = key_list[model_i]
    model = key
    
    model_names = np.append(model_names,model)

total = model_names

#%% Size of markers for legend (smaller size for single digit numbers)
ms_cmip = ([7,7,7,7,7,7,7,7,7,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
ms_noise = ([8*0.8,10*0.8,8*0.8,10*0.8,8*0.8,10*0.8,10*0.8,10*0.8,10*0.8,10*0.8,10*0.8])
        
#%% Construct the legend + save
f = lambda m,c, ms: plt.plot([],[],marker=m, color=c,markersize=ms, ls="none")[0]
handles = ([f("*", 'black',10)])
handles3 = ([f("$"+str(noises[noise_i])+"$", 'blue',ms_noise[noise_i]) for noise_i in range(len(noise_corr_mean))])
[handles.append(l) for l in handles3] 

labels = total
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)

def export_legend(legend, filename="Figure_S18_legend.png", expand=[-5,-5,15,5]):
    fig  = legend.figure
    fig.canvas.draw()
    plt.axis('off')
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(dir_fig+filename, dpi=quality, bbox_inches=bbox)

export_legend(legend)
plt.show()


    
    