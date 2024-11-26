# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure 8b, d, f for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import pandas as pd
import test

#%% Functions used in the script
def std(data1):  
    # Determine weighted standard deviations
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

# Load in the mask of the Atlantic
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# Load in data
var = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise_era5 = var['__xarray_dataarray_variable__'].compute()

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
cmip_metrics = np.array(pd.read_csv(dir_cmip+'metrics_noise_t2m_1deg.csv'))
noise_metrics = np.array(pd.read_csv(dir_era+'metrics_noise_realisations_t2m_new.csv'))
MMM_metrics = np.array(pd.read_csv(dir_cmip+'metrics_noise_MMM_t2m_1deg.csv'))

# Select spatial correlation of statistics CMIP6 models
cmip_corr_mean = cmip_metrics[0,1:]
cmip_corr_std = cmip_metrics[1,1:]
cmip_corr_skw = cmip_metrics[2,1:]
cmip_corr_kur = cmip_metrics[3,1:]

# Select weighted standard deviation of statistics CMIP6 models
cmip_std_mean = cmip_metrics[16,1:]
cmip_std_std = cmip_metrics[17,1:]
cmip_std_skw = cmip_metrics[18,1:]
cmip_std_kur = cmip_metrics[19,1:]

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

# Select spatial correlation of statistics CMIP6 multi model mean
MMM_corr_mean = MMM_metrics[0,1:]
MMM_corr_std = MMM_metrics[1,1:]
MMM_corr_skw = MMM_metrics[2,1:]
MMM_corr_kur = MMM_metrics[3,1:]

# Select weighted standard deviation of statistics CMIP6 multi model mean
MMM_std_mean = MMM_metrics[16,1:]
MMM_std_std = MMM_metrics[17,1:]
MMM_std_skw = MMM_metrics[18,1:]
MMM_std_kur = MMM_metrics[19,1:]

#%% Plotting variables
colors = 'red','blue','black'   # Colors to use
FS = 20                         # Base font size
quality = 300                   # Quality figures in dpi

models = np.arange(1,40,1)      # Markers for the CMIP6 models
noises = (['a','b','c','d'])    # Markers for the noise models

#%% Plot figures
# !! NOTE: FIRST RUN taylorDiagram.py TO BE ABLE TO PLOT THESE FIGURES !!
# Legend is plotted in Figure_8ace.py
# Figure 8b
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_std_E5, fig=fig,
                    label='Reference',srange = (0,1.75))

for MMM_i in range(1):
    dia.add_sample(MMM_std_std[MMM_i], MMM_corr_std[MMM_i],
               marker="s", ms=20, ls='',
               mfc=colors[2], mec=colors[2],
               label='test')
    
for model_i in range(len(cmip_corr_mean)):
    if model_i < 9:
        MS = 15*0.9
    else:
        MS = 20*0.9
    dia.add_sample(cmip_std_std[model_i], cmip_corr_std[model_i],
               marker="$"+str(models[model_i])+"$", ms=MS, ls='',
               mfc=colors[0], mec=colors[0],
               label='test')

for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2:
        ms = 15*0.9
    else:
        ms = 20*0.84
    dia.add_sample(noise_std_std[noise_i], noise_corr_std[noise_i],
               marker="$"+str(noises[noise_i])+"$", ms=ms, ls='',
               mfc=colors[1], mec=colors[1],
               label='test')
    
# Add RMS contours, and label them
plt.plot([0,0.13],[0,1.75*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.28],[0,1.75*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.48],[0,1.75*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.86],[0,1.75*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,1.32],[0,1.75*std_noise_std_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,4.4],[0,1.75*std_noise_std_E5],color = '0.5',linestyle = 'dotted')

contours = dia.add_contours(levels=5, colors='0.35') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=16, fmt='%.1f')
dia._ax.set_title('T$_{2m}$: standard deviation',fontsize=FS)

fig.tight_layout()

plt.savefig(dir_fig+'Figure_8_b.png', format='png', dpi=quality,bbox_inches='tight')

# Figure 8d
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_skw_E5, fig=fig,
                    label='Reference',srange = (0,2))

for MMM_i in range(1):
    dia.add_sample(MMM_std_skw[MMM_i], MMM_corr_skw[MMM_i],
               marker="s", ms=20, ls='',
               mfc=colors[2], mec=colors[2],
               label='test')
    
for model_i in range(len(cmip_corr_mean)):  
    if model_i < 9:
        MS = 15*0.9
    else:
        MS = 20*0.9
    dia.add_sample(cmip_std_skw[model_i], cmip_corr_skw[model_i],
               marker="$"+str(models[model_i])+"$", ms=MS, ls='',
               mfc=colors[0], mec=colors[0],
               label='test')

for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2:
        ms = 15*0.9
    else:
        ms = 20*0.84
    dia.add_sample(noise_std_skw[noise_i], noise_corr_skw[noise_i],
               marker="$"+str(noises[noise_i])+"$", ms=ms, ls='',
               mfc=colors[1], mec=colors[1],
               label='test')

# Add RMS contours, and label them
plt.plot([0,0.095],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.205],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.35],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.62],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.97],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,3.3],[0,2*std_noise_skw_E5],color = '0.5',linestyle = 'dotted')

contours = dia.add_contours(levels=5, colors='0.35') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=16, fmt='%.1f')
dia._ax.set_title('T$_{2m}$: skewness',fontsize=FS)

fig.tight_layout()

plt.savefig(dir_fig+'Figure_8_d.png', format='png', dpi=quality,bbox_inches='tight')

# Figure 8f
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_kur_E5, fig=fig,
                    label='Reference',srange = (0,2.1))
    
for MMM_i in range(1):
    dia.add_sample(MMM_std_kur[MMM_i], MMM_corr_kur[MMM_i],
               marker="s", ms=20, ls='',
               mfc=colors[2], mec=colors[2],
               label='test')

for model_i in range(len(cmip_corr_mean)):
    if model_i < 9:
        MS = 15*0.9
    else:
        MS = 20*0.9
    dia.add_sample(cmip_std_kur[model_i], cmip_corr_kur[model_i],
               marker="$"+str(models[model_i])+"$", ms=MS, ls='',
               mfc=colors[0], mec=colors[0],
               label='test')

for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2:
        ms = 15*0.9
    else:
        ms = 20*0.84
    dia.add_sample(noise_std_kur[noise_i], noise_corr_kur[noise_i],
               marker="$"+str(noises[noise_i])+"$", ms=ms, ls='',
               mfc=colors[1], mec=colors[1],
               label='test')
    
# Add RMS contours, and label them
plt.plot([0,0.42],[0,2.1*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,0.88],[0,2.1*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,1.54],[0,2.1*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,2.7],[0,2.1*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,4.25],[0,2.1*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')
plt.plot([0,14.5],[0,2.1*std_noise_kur_E5],color = '0.5',linestyle = 'dotted')

contours = dia.add_contours(levels=5, colors='0.35') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=16, fmt='%.1f')
dia._ax.set_title('T$_{2m}$: kurtosis',fontsize=FS)

fig.tight_layout()

plt.savefig(dir_fig+'Figure_8_f.png', format='png', dpi=quality,bbox_inches='tight')
