# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure 6, 7, A3 and A4 for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import cmocean as cm
import random
import matplotlib
from scipy.stats import norminvgauss

#%% Functions for plotting 
def aroace():
    # Colormap used for plotting
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ['#1F3554',"#5FAAD7","#FFFFFF","#E7C601","#E38D00"])

def plot_stat_2d(data,stat,model,vmin,vmax,cmap):
    # Function to plot the statistics
    # Input is:
        # the data that needs to be plotted (data)
        # which statistic is plotted (stat)
        # what and from what source is plotted (model)
        # minimum and maximum of color scaling (vmin, vmax)
        # and colormap (cmap)
    
    # Select extent axes and projection
    ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
    ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())
    
    # Plot data
    im=plt.pcolormesh(lon,lat,data,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,cmap=cmap)
    
    # Set title
    ax.set_title(model,fontsize=FS)
        
    # Colorbar specifics
    cbar=plt.colorbar(im,orientation='horizontal', pad=0.05) 
    cbar.ax.set_xlabel(stat, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.offsetText.set_fontsize(16)
    
    # Set specifics of the plotting background
    ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
    ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
    gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    
#%% Functions for analysis
def weighted_corr(x, y, w):
    """Calculate weighted correlation coefficient."""
    w_mean_x = np.sum(w * x) / np.sum(w)
    w_mean_y = np.sum(w * y) / np.sum(w)
    
    cov_xy = np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w)
    std_x = np.sqrt(np.sum(w * (x - w_mean_x)**2) / np.sum(w))
    std_y = np.sqrt(np.sum(w * (y - w_mean_y)**2) / np.sum(w))
    
    return cov_xy / (std_x * std_y)

def metrics(data1,data2):
    # Function to determine metrics 
    # Make sure data is in right format
    data1_xr = data1*xr.ones_like(noise[0,:,:])
    data2_xr = data2*xr.ones_like(noise[0,:,:])
    
    # Flatten data
    da1_flat = data1_xr.stack(z=('latitude', 'longitude')).values
    da2_flat = data2_xr.stack(z=('latitude', 'longitude')).values
    
    # Determine weighting based on latitudes
    latitudes = data2_xr['latitude']
    weights = np.cos(np.deg2rad(latitudes))*xr.ones_like(data2_xr)
    weights = weights / weights.sum()*A_mask_era  # Normalize weights
    
    weights_flat = weights.stack(z=('latitude', 'longitude')).values
    
    # Mask data
    valid_mask = ~np.isnan(da1_flat) & ~np.isnan(da2_flat)
    da1_flat = da1_flat[valid_mask]
    da2_flat = da2_flat[valid_mask]
    weights_flat = weights_flat[valid_mask]

    # Compute the correlation
    correlation = weighted_corr(da1_flat, da2_flat, weights_flat)
    
    # Calculate weighted variables
    weighted_mean_1 = np.average(data1_xr.fillna(0), weights=weights.fillna(0))
    weighted_variance_1 = np.average((data1_xr.fillna(0) - weighted_mean_1) ** 2, weights=weights.fillna(0))
    weighted_std_1 = np.sqrt(weighted_variance_1)
    
    weighted_mean_2 = np.average(data2_xr.fillna(0), weights=weights.fillna(0))
    weighted_variance_2 = np.average((data2_xr.fillna(0) - weighted_mean_2) ** 2, weights=weights.fillna(0))
    weighted_std_2 = np.sqrt(weighted_variance_2)
    
    sf = weighted_std_1/weighted_std_2
    
    # Calculate Taylor Skill Score
    TSS = (1+correlation)**4/((2**4)*(sf+1/sf)**2)
    
    # Calculate Bias
    bias = np.average(data2_xr.fillna(0)-data1_xr.fillna(0), weights=weights.fillna(0))
    
    # Calculate RSME
    rmse = np.sqrt(np.average((data2_xr.fillna(0)-data1_xr.fillna(0))**2, weights=weights.fillna(0)))
    
    return correlation, TSS, bias, rmse, weighted_std_1, weighted_std_2

#%% Plotting variables
FS = 20         # Base fontsize
quality = 300   # Quality in dp

#%% Directories of data and where figures need to be saved
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Figures_sub/'
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'

#%% Determine plotting grid
target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

#%% Load in mask
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

#%% Load in noise
data = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data['__xarray_dataarray_variable__'].compute()

#%% Determine statistics ERA5 noise
Ae5 = np.mean(noise,axis=0)
Be5 = np.std(noise,axis=0)
Ce5 = scipy.stats.skew(noise,axis=0)
De5 = scipy.stats.kurtosis(noise,axis=0)

#%% Load in PCs and EOFs of ERA5 noise
PCs = np.load(dir_era+'PCs_t2m_era_new.npy')
EOF_re = np.load(dir_era+'EOFs_t2m_era_new.npy')
eof_re = np.swapaxes(EOF_re,1,2)#['EOFs'].compute()

#%% Initialize arrays of models
size_x = 10000 # Number of noise realisations

pc_series_re = np.zeros(np.shape(eof_re)[0]) 
grid_n = np.zeros([size_x,np.shape(eof_re)[1],np.shape(eof_re)[2]]) 
grid_a = np.zeros([size_x,np.shape(eof_re)[1],np.shape(eof_re)[2]]) 

#%% Methods to use and Figure numbers per model (PC (NIG): 6, NIG: 7, PC (1): A3, PC (N): A4)
methods = (['PC (NIG)','NIG','PC (1)','PC (N)'])
fig_nr = (['6','6','6','6','6','6','7','7','7','7','7','7','A3','A3','A3','A3','A3','A3','A4','A4','A4','A4','A4','A4'])

#%% Initialize arrays of metrics
corr_noise_mean = np.zeros((len(methods)))
corr_noise_std = np.zeros((len(methods)))
corr_noise_skw = np.zeros((len(methods)))
corr_noise_kur = np.zeros((len(methods)))

TSS_noise_mean = np.zeros((len(methods)))
TSS_noise_std = np.zeros((len(methods)))
TSS_noise_skw = np.zeros((len(methods)))
TSS_noise_kur = np.zeros((len(methods)))

bias_noise_mean = np.zeros((len(methods)))
bias_noise_std = np.zeros((len(methods)))
bias_noise_skw = np.zeros((len(methods)))
bias_noise_kur = np.zeros((len(methods)))

rmse_noise_mean = np.zeros((len(methods)))
rmse_noise_std = np.zeros((len(methods)))
rmse_noise_skw = np.zeros((len(methods)))
rmse_noise_kur = np.zeros((len(methods)))

std_noise_mean = np.zeros((len(methods)))
std_noise_std = np.zeros((len(methods)))
std_noise_skw = np.zeros((len(methods)))
std_noise_kur = np.zeros((len(methods)))

#%% Initialize parameter arrays for NIG distributions
PARAMS = np.zeros((4,721,521))
Params = np.zeros((4,np.shape(eof_re)[0]))

#%% Construct noise timeseries for each model
for method_i in range(len(methods)):
    method = methods[method_i]
    print(method)
    
    if method == 'PC (1)':
        # Loop over time
        for time_i in range(size_x):
            random_nr = random.randint(0,995) # Determine random integer between 0 and 995
            # Loop over EOFs
            for eof_j in range(np.shape(eof_re)[0]):
                grid_n[time_i,:,:] += PCs[random_nr,eof_j] * eof_re[eof_j,:,:]
                
    elif method == 'NIG':
        params_tot = np.load(dir_era+'params_era_t2m_NIG.npy') # Loadin parameters
        
        # Loop over each grid point
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                # If grid point is NaN (i.e. a land point), noise is also NaN
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((4))
                else:
                    # For first time fit + save parameters
                    params = params_tot[:,lat_i,lon_i] #norminvgauss.fit(noise[:,lat_i,lon_i])
                    A = norminvgauss.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        #np.save(dir_era+'params_era_t2m_NIG.npy',PARAMS)
        
    elif method == 'PC (NIG)':
        # Loop over every EOF
        for eof_j in range(np.shape(eof_re)[0]):
            print(eof_j)
            params = norminvgauss.fit(PCs[:,eof_j])
            Params[:,eof_j] = params 
            
            pc_rvs = norminvgauss.rvs(*params,size=size_x)
            # Loop over time
            for time_i in range(size_x):
                grid_n[time_i,:,:] += pc_rvs[time_i] * eof_re[eof_j,:,:]
            
        np.save(dir_era+'params_era_t2m_PC_NIG.npy',Params)
    
    elif method == 'PC (N)':
        # Loop over time
        for time_i in range(size_x):
            # Loop over EOFs
            for eof_j in range(np.shape(eof_re)[0]):
                random_nr = random.randint(0,995) # Random integer between 0 and 995
                grid_n[time_i,:,:] += PCs[random_nr,eof_j] * eof_re[eof_j,:,:]
    
    # Determine statistics for noise fields
    Am = np.mean(grid_n,axis=0)
    B = np.std(grid_n,axis=0)
    C = scipy.stats.skew(grid_n,axis=0)
    D = scipy.stats.kurtosis(grid_n,axis=0)
    
    # Determine metrics for noise fields
    [corr_noise_mean[method_i],TSS_noise_mean[method_i], bias_noise_mean[method_i], rmse_noise_mean[method_i], std_noise_mean[method_i], std_noise_mean_E5] = metrics(Am,Ae5)
    [corr_noise_std[method_i],TSS_noise_std[method_i], bias_noise_std[method_i], rmse_noise_std[method_i], std_noise_std[method_i], std_noise_std_E5] = metrics(B,Be5)
    [corr_noise_skw[method_i],TSS_noise_skw[method_i], bias_noise_skw[method_i], rmse_noise_skw[method_i], std_noise_skw[method_i], std_noise_skw_E5] = metrics(C,Ce5)
    [corr_noise_kur[method_i],TSS_noise_kur[method_i], bias_noise_kur[method_i], rmse_noise_kur[method_i], std_noise_kur[method_i], std_noise_kur_E5] = metrics(D,De5)

    model = method

#%% Plot figures
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(B,'$\sigma$ (T$_{2m}$) [$^{\circ}$C]','Noise model: ' + str(model) ,0,1.75,cm.cm.deep_r)
    plt.text(3.5e6,8.2e6,str(np.round(corr_noise_std[method_i],2)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str(np.round(rmse_noise_std[method_i],2)),color='white',fontsize = 18, weight='bold')
        
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+0]+'_g_new.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(C,'Skewness (T$_{2m}$) [-]','Noise model: ' + str(model) ,-1.15,1.15,aroace())
    plt.text(3.5e6,8.2e6,str(np.round(corr_noise_skw[method_i],2)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str(np.round(rmse_noise_skw[method_i],2)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+1]+'_h_new.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(D,'Kurtosis (T$_{2m}$) [-]','Noise model: ' + str(model) ,-1.75,1.75,aroace())
    plt.text(3.5e6,8.2e6,str(np.round(corr_noise_kur[method_i],2)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str(np.round(rmse_noise_kur[method_i],2)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+2]+'_i_new.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(Be5-B,'$\Delta$$\sigma$ (T$_{2m}$) [$^{\circ}$C]','Noise model: ' + str(model) ,-3.1,3.1,'RdBu_r')
        
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+3]+'_j_alt_new.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(Be5-B,'$\Delta$$\sigma$ (T$_{2m}$) [$^{\circ}$C]','Noise model: ' + str(model) ,-0.075,0.075,'RdBu_r')
        
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+3]+'_j_alt_new.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(Ce5-C,'$\Delta$Skewness (T$_{2m}$) [-]','Noise model: ' + str(model) ,-1.15,1.15,'RdBu_r')
        
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+4]+'_k_new.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(De5-D,'$\Delta$Kurtosis (T$_{2m}$) [-]','Noise model: ' + str(model) ,-1.75,1.75,'RdBu_r')
        
    plt.savefig(dir_fig+'Figure_'+fig_nr[method_i*6+5]+'_l_new.png', format='png', dpi=quality,bbox_inches='tight')
    
#%% Save metrics (correlation, Taylor skill score, bias, rmse)
data_list_d = (['cdm','cdst','cdsk','cdk','tdm','tdst','tdsk','tdk','bdm','bdst','bdsk','bdk','rdm','rdst','rdsk','rdk','sdm','sdst','sdsk','sdk'])
data_d =np.reshape(np.concatenate([corr_noise_mean,corr_noise_std,corr_noise_skw,corr_noise_kur,TSS_noise_mean,TSS_noise_std,TSS_noise_skw,TSS_noise_kur,bias_noise_mean,bias_noise_std,bias_noise_skw,bias_noise_kur,rmse_noise_mean,rmse_noise_std,rmse_noise_skw,rmse_noise_kur,std_noise_mean,std_noise_std,std_noise_skw,std_noise_kur],axis=0),(20,len(methods)))

ser_d = pd.DataFrame(np.round(data_d,2), index=[data_list_d])
ser_d.to_csv(dir_era+'metrics_noise_realisations_t2m_new.csv', index=True)



