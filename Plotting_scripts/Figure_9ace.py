# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure 8a, c, e and legend for 'Observation based temperature and freshwater flux over the Atlantic Ocean'
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
cmip_metrics = np.array(pd.read_csv(dir_cmip+'metrics_noise_ep_1deg.csv'))
noise_metrics = np.array(pd.read_csv(dir_era+'metrics_noise_realisations_ep_new.csv'))
MMM_metrics = np.array(pd.read_csv(dir_cmip+'metrics_noise_MMM_ep_1deg.csv'))

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
colors = 'red','blue','black'   # Colors used for markers
FS = 20                         # Base font size
quality = 300                   # Quality of figure in dpi

models = np.arange(1,40,1)      # Markers for CMIP6 models
noises = (['a','b','c','d'])    # Markers for noise models

#%% Plot figures
# !! NOTE: FIRST RUN taylorDiagram.py TO BE ABLE TO PLOT THESE FIGURES !!
# Figure 8a
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_std_E5, fig=fig,
                    label='Reference',srange=(0,1.5))

# Plot CMIP6 MMM
for MMM_i in range(1):
    dia.add_sample(MMM_std_std[MMM_i], MMM_corr_std[MMM_i],
               marker="s", ms=20, ls='',
               mfc=colors[2], mec=colors[2],
               label='test')

# Plot CMIP6 models    
for model_i in range(len(cmip_corr_mean)):
    if model_i < 9:
        MS = 15*0.9
    else:
        MS = 20*0.9
    dia.add_sample(cmip_std_std[model_i], cmip_corr_std[model_i],
               marker="$"+str(models[model_i])+"$", ms=MS, ls='',
               mfc=colors[0], mec=colors[0],
               label='test')

# Plot noise models
for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2:
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

plt.savefig(dir_fig+'Figure_8_a.png', format='png', dpi=quality,bbox_inches='tight')

# Figure 8c
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_skw_E5, fig=fig,
                    label='Reference',srange = (0,2))

# Plot CMIP6 MMM
for MMM_i in range(1):
    dia.add_sample(MMM_std_skw[MMM_i], MMM_corr_skw[MMM_i],
               marker="s", ms=20, ls='',
               mfc=colors[2], mec=colors[2],
               label='test')

# Plot CMIP6 models    
for model_i in range(len(cmip_corr_mean)):  
    if model_i < 9:
        MS = 15*0.9
    else:
        MS = 20*0.9
    dia.add_sample(cmip_std_skw[model_i], cmip_corr_skw[model_i],
               marker="$"+str(models[model_i])+"$", ms=MS, ls='',
               mfc=colors[0], mec=colors[0],
               label='test')

# Plot noise models
for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2:
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

plt.savefig(dir_fig+'Figure_8_c.png', format='png', dpi=quality,bbox_inches='tight')

# Figure 8e
fig = plt.figure(figsize=(11,8))
dia = TaylorDiagram(std_noise_kur_E5, fig=fig,
                    label='Reference',srange = (0,2))

# Plot CMIP6 MMM
for MMM_i in range(1):
    dia.add_sample(MMM_std_kur[MMM_i], MMM_corr_kur[MMM_i],
               marker="s", ms=20, ls='',
               mfc=colors[2], mec=colors[2],
               label='test')

# Plot CMIP6 models    
for model_i in range(len(cmip_corr_mean)):
    if model_i < 9:
        MS = 15*0.9
    else:
        MS = 20*0.9
    dia.add_sample(cmip_std_kur[model_i], cmip_corr_kur[model_i],
               marker="$"+str(models[model_i])+"$", ms=MS, ls='',
               mfc=colors[0], mec=colors[0],
               label='test')

# Plot noise models
for noise_i in range(len(noise_corr_mean)):
    if noise_i == 0 or noise_i == 2:
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

plt.savefig(dir_fig+'Figure_8_e.png', format='png', dpi=quality,bbox_inches='tight')

#%% List of CMIP6 models
key_list1 = (['CMIP.AS-RCEC.TaiESM1.historical.Amon.gn','CMIP.AWI.AWI-CM-1-1-MR.historical.Amon.gn','CMIP.AWI.AWI-ESM-1-1-LR.historical.Amon.gn','CMIP.BCC.BCC-CSM2-MR.historical.Amon.gn','CMIP.BCC.BCC-ESM1.historical.Amon.gn','CMIP.CAS.FGOALS-g3.historical.Amon.gn','CMIP.CCCma.CanESM5-CanOE.historical.Amon.gn','CMIP.CAS.CAS-ESM2-0.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-HR4.historical.Amon.gn'])
key_list2 = (['CMIP.CCCma.CanESM5.historical.Amon.gn','CMIP.CCCR-IITM.IITM-ESM.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-SR5.historical.Amon.gn','CMIP.CMCC.CMCC-ESM2.historical.Amon.gn','CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Amon.gn','CMIP.CSIRO.ACCESS-ESM1-5.historical.Amon.gn','CMIP.FIO-QLNM.FIO-ESM-2-0.historical.Amon.gn','CMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.historical.Amon.gn','CMIP.MIROC.MIROC-ES2L.historical.Amon.gn'])
key_list3 = (['CMIP.MIROC.MIROC6.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-LL.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-MM.historical.Amon.gn','CMIP.MOHC.UKESM1-0-LL.historical.Amon.gn','CMIP.MPI-M.MPI-ESM1-2-LR.historical.Amon.gn','CMIP.MRI.MRI-ESM2-0.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-G-CC.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-H.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM-FV2.historical.Amon.gn'])
key_list4 = (['CMIP.NASA-GISS.GISS-E2-2-H.historical.Amon.gn','CMIP.NCAR.CESM2-FV2.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM.historical.Amon.gn','CMIP.NCAR.CESM2.historical.Amon.gn','CMIP.NCC.NorESM2-MM.historical.Amon.gn','CMIP.NIMS-KMA.UKESM1-0-LL.historical.Amon.gn','CMIP.NUIST.NESM3.historical.Amon.gn','CMIP.SNU.SAM0-UNICON.historical.Amon.gn','CMIP.UA.MCM-UA-1-0.historical.Amon.gn'])

key_list = np.concatenate([key_list1,key_list2,key_list3,key_list4])
model_names = 'ERA5'

# Select model name from key_list
for model_i in range(len(cmip_corr_mean)):
    key = key_list[model_i]
    x = [a for a in key.split('.') if a]
    model = ('.'.join(x[2:3]))
    
    model_names = np.append(model_names,model)

#%% Construct list for legend (if single digit, add space before)
A = []

for model_i in range(len((cmip_corr_mean))):
    if model_i < 9:
        A = np.append(' ' + str(models[model_i]),A)
    else:
        A = np.append(str(models[model_i]),A)
        
#%% Other names
B = np.append(model_names,'MMM')
C = np.append(B,'PC (NIG)')
D = np.append(C,'NIG')
E = np.append(D,'PC (1)')
total = np.append(E,'PC (N)')

#%% Size of markers for legend (smaller size for single digit numbers)
ms_cmip = ([7,7,7,7,7,7,7,7,7,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
ms_noise = ([8*0.8,10*0.8,8*0.8,10*0.8])
        
#%% Construct the legend + save
f = lambda m,c, ms: plt.plot([],[],marker=m, color=c,markersize=ms, ls="none")[0]
handles = ([f("*", 'black',10)])
handles1 = ([f("$"+str(A[-(model_i+1)])+"$", 'red',ms_cmip[model_i]) for model_i in range(len(cmip_corr_mean))])
handles2 = ([f("s", 'black',10)])
handles3 = ([f("$"+str(noises[noise_i])+"$", 'blue',ms_noise[noise_i]) for noise_i in range(len(noise_corr_mean))])
[handles.extend(l) for l in (handles1,handles2,handles3)]

labels = total
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)

def export_legend(legend, filename="Figure_8_legend.png", expand=[-5,-5,15,5]):
    fig  = legend.figure
    fig.canvas.draw()
    plt.axis('off')
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(dir_fig+filename, dpi=quality, bbox_inches=bbox)

export_legend(legend)
plt.show()


    
    
