# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for determining statistics for Taylor Diagram for different pointwise distributions 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024
# Uses data from 'Fit_distributions_EP.py'

#%% Load in modules
import xarray as xr 
import numpy as np
import scipy.stats
import pandas as pd
import xesmf as xe

#%% Functions used in the scripts
def determine_noise(key):
    # Function to load data and determine noise in the CMIP6 models
    var = xr.open_mfdataset(dir_cmip+key+'_ep_new.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
    P_E = var['e-p'][:,:,:]*A_mask_cmip*conv_int_cmip
    
    PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()
    noise = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()
        
    return P_E, noise
    
def regridder(ds,ds_out,dr):
    # Function to regrid data from 0.25 degree to 1 degree
    regridder = xe.Regridder(ds, ds_out, "nearest_s2d")    
    dr_out = regridder(dr, keep_attrs=True)
    
    return dr_out

def weighted_corr(x, y, w):
    """Calculate weighted correlation coefficient."""
    w_mean_x = np.sum(w * x) / np.sum(w)
    w_mean_y = np.sum(w * y) / np.sum(w)
    
    cov_xy = np.sum(w * (x - w_mean_x) * (y - w_mean_y)) / np.sum(w)
    std_x = np.sqrt(np.sum(w * (x - w_mean_x)**2) / np.sum(w))
    std_y = np.sqrt(np.sum(w * (y - w_mean_y)**2) / np.sum(w))
    
    return cov_xy / (std_x * std_y)

def metrics(data1,data2):
    # Script to determine metrics
    
    # Regrid ERA5 data to a 1 degree grid
    #data2_r = regridder(data2,data1,data2)
    
    # Mask regridded ERA5 data with the CMIP6 land mask
    data2_m = data2#data2_r * A_mask_cmip
    
    # ERA5 resolves islands that are not captured by the CMIP6 land mask.
    # The regridding does not interpolate correctly over those islands.
    # We therefore mask out these islands in the CMIP6 data by multiplying by ERA5/ERA5 
    data1_m = data1 * data2_m/data2_m
    
    # Flatten data
    da1_flat = data1_m.stack(z=('latitude', 'longitude')).values
    da2_flat = data2_m.stack(z=('latitude', 'longitude')).values
    
    # Determine weights based on latitudes
    latitudes = data1['latitude']
    weights = np.cos(np.deg2rad(latitudes))*xr.ones_like(data1_m)
    weights = weights / weights.sum()*A_mask_era  # Normalize weights
    
    weights_flat = weights.stack(z=('latitude', 'longitude')).values
    
    valid_mask = ~np.isnan(da1_flat) & ~np.isnan(da2_flat)
    da1_flat = da1_flat[valid_mask]
    da2_flat = da2_flat[valid_mask]
    weights_flat = weights_flat[valid_mask]

    # Compute the correlation
    correlation = weighted_corr(da1_flat, da2_flat, weights_flat)
    
    # Determine weighted standard deviation
    weighted_mean_1 = np.average(data1_m.fillna(0), weights=weights.fillna(0))
    weighted_variance_1 = np.average((data1_m.fillna(0) - weighted_mean_1) ** 2, weights=weights.fillna(0))
    weighted_std_1 = np.sqrt(weighted_variance_1)
    
    weighted_mean_2 = np.average(data2_m.fillna(0), weights=weights.fillna(0))
    weighted_variance_2 = np.average((data2_m.fillna(0) - weighted_mean_2) ** 2, weights=weights.fillna(0))
    weighted_std_2 = np.sqrt(weighted_variance_2)
    
    # Compute other metrics: Taylor skill score (TSS), the bias, and RMSE
    sf = weighted_std_1/weighted_std_2
    TSS = (1+correlation)**4/((2**4)*(sf+1/sf)**2)
    bias = np.average(data2_m.fillna(0)-data1_m.fillna(0), weights=weights.fillna(0))
    rmse = np.sqrt(np.average((data2_m.fillna(0)-data1.fillna(0))**2, weights=weights.fillna(0)))
    
    return correlation, TSS, bias, rmse, weighted_std_1, weighted_std_2

#%% Masks + directories
# Directories with data
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_cmip = '/Users/Boot0016/Documents/Project_noise_model/CMIP6_data/CMIP6_EP/'
dir_cmip1 = '/Users/Boot0016/Documents/Project_noise_model/CMIP6_data/'

# ERA5 land mask
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# CMIP6 land mask
mask_cmip = xr.open_dataset(dir_cmip1+'cmip6_atlantic_mask_60S_new.nc')
A_mask_cmip = mask_cmip.mask 

# Units CMIP6: kg m-2 s-1 need to be transformed to mm/day
conv_int_cmip = 86400 * 1e3 / 1e3

#%% List of models
methods = (['genpareto','johnsonsu','skewnorm','genhyperbolic','genextreme','gamma','beta','normal','nig'])

#%% ERA5 E - P data
# Load in data and determine noise (for the period 1940 - 2014)
var = xr.open_mfdataset(dir_era+'e_p_1940_2022_monthly_era.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
EP_era5 = -var['e-p'][:900,:,:]*A_mask_era*1e3

PE = EP_era5-EP_era5.rolling(time=60, center=True,min_periods=12).mean().compute()
noise_era5 = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()

# Determine statistics noise
noise_mean_e5 = noise_era5.mean('time').compute()
noise_std_e5 = noise_era5.std('time').compute()
noise_skw_e5 = noise_era5.reduce(func=scipy.stats.skew,dim='time').compute()
noise_kur_e5 = noise_era5.reduce(func=scipy.stats.kurtosis,dim='time').compute()

# Determine statstics data
data_mean_e5 = EP_era5.mean('time')
data_std_e5 = EP_era5.std('time')
data_skw_e5 = EP_era5.reduce(func=scipy.stats.skew,dim='time')
data_kur_e5 = EP_era5.reduce(func=scipy.stats.kurtosis,dim='time')

#%% Construct arrays for the different metrics (size = number of CMIP6 models)
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

#%%
dir_test = '/Users/Boot0016/Documents/Project_noise_model/test/'
for key_i in range(len(methods)):
    # Select model
    method = methods[key_i]
    print(method)

    # Determine statistics noise per model
    noise_std = np.load(dir_test+'std_'+method+'_ep.npy') * 1e3 * xr.ones_like(noise_mean_e5)
    noise_skw = np.load(dir_test+'skw_'+method+'_ep.npy') * xr.ones_like(noise_mean_e5)
    noise_kur = np.load(dir_test+'kur_'+method+'_ep.npy') * xr.ones_like(noise_mean_e5)
    
    # Determine metrics noise per model
    [corr_noise_std[key_i],TSS_noise_std[key_i], bias_noise_std[key_i], rmse_noise_std[key_i], std_noise_std[key_i], std_noise_std_E5] = metrics(noise_std,noise_std_e5)
    [corr_noise_skw[key_i],TSS_noise_skw[key_i], bias_noise_skw[key_i], rmse_noise_skw[key_i], std_noise_skw[key_i], std_noise_skw_E5] = metrics(noise_skw,noise_skw_e5)
    [corr_noise_kur[key_i],TSS_noise_kur[key_i], bias_noise_kur[key_i], rmse_noise_kur[key_i], std_noise_kur[key_i], std_noise_kur_E5] = metrics(noise_kur,noise_kur_e5)
    

#%% Save the metrics
data_list_d = (['cdm','cdst','cdsk','cdk','tdm','tdst','tdsk','tdk','bdm','bdst','bdsk','bdk','rdm','rdst','rdsk','rdk','sdm','sdst','sdsk','sdk'])
data_list_n = (['cnm','cnst','cnsk','cnk','tnm','tnst','tnsk','tnk','bnm','bnst','bnsk','bnk','rnm','rnst','rnsk','rnk','snm','snst','snsk','snk'])

data_n =np.reshape(np.concatenate([corr_noise_mean,corr_noise_std,corr_noise_skw,corr_noise_kur,TSS_noise_mean,TSS_noise_std,TSS_noise_skw,TSS_noise_kur,bias_noise_mean,bias_noise_std,bias_noise_skw,bias_noise_kur,rmse_noise_mean,rmse_noise_std,rmse_noise_skw,rmse_noise_kur,std_noise_mean,std_noise_std,std_noise_skw,std_noise_kur],axis=0),(20,9))

ser_n = pd.DataFrame(np.round(data_n,2), index=[data_list_n])

ser_n.to_csv(dir_test+'metrics_noise_ep_noise_models.csv', index=True)
    
    