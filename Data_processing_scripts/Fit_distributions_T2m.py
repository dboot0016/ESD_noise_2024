# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for to fit different point wise distributions to ERA5 T2m noise for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
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
from scipy.stats import skewnorm, genhyperbolic,genextreme,gamma,beta,johnsonsu,genpareto,norm, norminvgauss

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
    data1_xr = data1*xr.ones_like(noise[0,:,:])
    data2_xr = data2*xr.ones_like(noise[0,:,:])
    
    da1_flat = data1_xr.stack(z=('latitude', 'longitude')).values
    da2_flat = data2_xr.stack(z=('latitude', 'longitude')).values
    
    latitudes = data2_xr['latitude']
    weights = np.cos(np.deg2rad(latitudes))*xr.ones_like(data2_xr)
    weights = weights / weights.sum()*A_mask_era  # Normalize weights
    
    weights_flat = weights.stack(z=('latitude', 'longitude')).values
    
    valid_mask = ~np.isnan(da1_flat) & ~np.isnan(da2_flat)
    da1_flat = da1_flat[valid_mask]
    da2_flat = da2_flat[valid_mask]
    weights_flat = weights_flat[valid_mask]

    # Compute the correlation
    correlation = weighted_corr(da1_flat, da2_flat, weights_flat)
    
    weighted_mean_1 = np.average(data1_xr.fillna(0), weights=weights.fillna(0))
    weighted_variance_1 = np.average((data1_xr.fillna(0) - weighted_mean_1) ** 2, weights=weights.fillna(0))
    weighted_std_1 = np.sqrt(weighted_variance_1)
    
    weighted_mean_2 = np.average(data2_xr.fillna(0), weights=weights.fillna(0))
    weighted_variance_2 = np.average((data2_xr.fillna(0) - weighted_mean_2) ** 2, weights=weights.fillna(0))
    weighted_std_2 = np.sqrt(weighted_variance_2)
    
    sf = weighted_std_1/weighted_std_2
    
    TSS = (1+correlation)**4/((2**4)*(sf+1/sf)**2)
    
    bias = np.average(data2_xr.fillna(0)-data1_xr.fillna(0), weights=weights.fillna(0))
    
    rmse = np.sqrt(np.average((data2_xr.fillna(0)-data1_xr.fillna(0))**2, weights=weights.fillna(0)))
    
    return correlation, TSS, bias, rmse, weighted_std_1, weighted_std_2

#%%
LW = 4
FS = 20
quality = 300

dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Figures_sub/'
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'

dir_fig_test = '/Users/Boot0016/Documents/Project_noise_model/Figures_test/'
dir_era_test = '/Users/Boot0016/Documents/Project_noise_model/Test/'

target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

data = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data['__xarray_dataarray_variable__'].compute()

Ae5 = np.mean(noise,axis=0)
Be5 = np.std(noise,axis=0)
Ce5 = scipy.stats.skew(noise,axis=0)
De5 = scipy.stats.kurtosis(noise,axis=0)

PCs = np.load(dir_era+'ERA5_PCs_T2m.npy')
EOF_re = xr.open_dataset(dir_era+'ERA5_EOFs_T2m.nc')
eof_re = EOF_re['EOFs'].compute()

size_x = 5000

pc_series_re = np.zeros(310) 
grid_n = np.zeros([size_x,np.shape(eof_re)[1],np.shape(eof_re)[2]]) 
grid_a = np.zeros([size_x,np.shape(eof_re)[1],np.shape(eof_re)[2]]) 

methods = (['genpareto','johnsonsu','skewnorm','genhyperbolic','genextreme','gamma','beta','normal','nig'])

#%%
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

X = ([3,4,3,5,3,3,4,2,4])


#%%
for method_i in range(len(methods)-7):
    method = methods[method_i+7]
    print(method)
    
    x = X[method_i+7]
    PARAMS = np.zeros((x,721,521))
    
    if method == 'skewnorm':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = skewnorm.fit(noise[:,lat_i,lon_i])
                    A = skewnorm.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_skwn.npy',PARAMS)
                
    elif method == 'genhyperbolic':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = genhyperbolic.fit(noise[:,lat_i,lon_i])
                    A = genhyperbolic.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_genh.npy',PARAMS)
        
    elif method == 'genextreme':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = genextreme.fit(noise[:,lat_i,lon_i])
                    A = genextreme.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_gene.npy',PARAMS)
    
    elif method == 'gamma':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = gamma.fit(noise[:,lat_i,lon_i])
                    A = gamma.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_gamma.npy',PARAMS)
    
    elif method == 'beta':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = beta.fit(noise[:,lat_i,lon_i])
                    A = beta.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_beta.npy',PARAMS)
        
    elif method == 'genpareto':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = genpareto.fit(noise[:,lat_i,lon_i])
                    A = genpareto.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_genpareto.npy',PARAMS)
        
    elif method == 'johnsonsu':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = johnsonsu.fit(noise[:,lat_i,lon_i])
                    A = johnsonsu.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_johnsonsu.npy',PARAMS)
    
    elif method == 'normal':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = norm.fit(noise[:,lat_i,lon_i])
                    A = norm.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_normal.npy',PARAMS)
        
    elif method == 'nig':
        for lat_i in range(np.shape(eof_re)[1]):
            for lon_i in range(np.shape(eof_re)[2]):
                if np.isnan(noise[0,lat_i,lon_i]):
                    grid_n[:,lat_i,lon_i] = np.nan*np.ones((size_x))
                    PARAMS[:,lat_i,lon_i] = np.nan*np.ones((x))
                else:
                    params = norminvgauss.fit(noise[:,lat_i,lon_i])
                    A = norminvgauss.rvs(*params,size = size_x)
                    grid_n[:,lat_i,lon_i] = A
                    PARAMS[:,lat_i,lon_i] = params
                    
        np.save(dir_era_test+'params_era_t2m_nig_test.npy',PARAMS)
    
        
    Am = np.mean(grid_n,axis=0)
    B = np.std(grid_n,axis=0)
    C = scipy.stats.skew(grid_n,axis=0)
    D = scipy.stats.kurtosis(grid_n,axis=0)
    
    np.save('std_' + method +'_t2m.npy',B)
    np.save('skw_' + method +'_t2m.npy',C)
    np.save('kur_' + method +'_t2m.npy',D)


