# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for determining metrics for CMIP6 models (T2m) for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Geophysical Research Letters' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import scipy.stats
import pandas as pd
import xesmf as xe

#%% Functions used in the scripts
def determine_noise(key):
    # Function to load data and determine noise in the CMIP6 models
    var = xr.open_mfdataset(dir_cmip+key+'_tas_new.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
    T2m = var['tas'][:,:,:]*A_mask_cmip*conv_int_cmip-273 # Mask Atlantic and convert from Kelvin to degrees Celsius
    
    t2m = T2m-T2m.rolling(time=60, center=True,min_periods=12).mean().compute() # Detrend by subtracting a 5 year running mean
    noise = t2m.groupby('time.month')-t2m.groupby('time.month').mean('time').compute()  # Deseasonalize
        
    return T2m, noise
    
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
    data2_r = regridder(data2,data1,data2)
    
    # Mask regridded ERA5 data with the CMIP6 land mask
    data2_m = data2_r * A_mask_cmip
    
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
    weights = weights / weights.sum()*A_mask_cmip  # Normalize weights
    
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
dir_cmip = '/Users/Boot0016/Documents/Project_noise_model/CMIP6_data/'

# ERA5 land mask
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# CMIP6 land mask
mask_cmip = xr.open_dataset(dir_cmip+'cmip6_atlantic_mask_60S_new.nc')
A_mask_cmip = mask_cmip.mask 

# No unit change necessary
conv_int_cmip = 1

#%% List of models
key_list1 = (['CMIP.AS-RCEC.TaiESM1.historical.Amon.gn','CMIP.AWI.AWI-CM-1-1-MR.historical.Amon.gn','CMIP.AWI.AWI-ESM-1-1-LR.historical.Amon.gn','CMIP.BCC.BCC-CSM2-MR.historical.Amon.gn','CMIP.BCC.BCC-ESM1.historical.Amon.gn','CMIP.CAS.FGOALS-g3.historical.Amon.gn','CMIP.CCCma.CanESM5-CanOE.historical.Amon.gn','CMIP.CAS.CAS-ESM2-0.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-HR4.historical.Amon.gn'])
key_list2 = (['CMIP.CCCma.CanESM5.historical.Amon.gn','CMIP.CCCR-IITM.IITM-ESM.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-SR5.historical.Amon.gn','CMIP.CMCC.CMCC-ESM2.historical.Amon.gn','CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Amon.gn','CMIP.CSIRO.ACCESS-ESM1-5.historical.Amon.gn','CMIP.FIO-QLNM.FIO-ESM-2-0.historical.Amon.gn','CMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.historical.Amon.gn','CMIP.MIROC.MIROC-ES2L.historical.Amon.gn'])
key_list3 = (['CMIP.MIROC.MIROC6.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-LL.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-MM.historical.Amon.gn','CMIP.MOHC.UKESM1-0-LL.historical.Amon.gn','CMIP.MPI-M.MPI-ESM1-2-LR.historical.Amon.gn','CMIP.MRI.MRI-ESM2-0.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-G-CC.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-H.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM-FV2.historical.Amon.gn'])
key_list4 = (['CMIP.NASA-GISS.GISS-E2-2-H.historical.Amon.gn','CMIP.NCAR.CESM2-FV2.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM.historical.Amon.gn','CMIP.NCAR.CESM2.historical.Amon.gn','CMIP.NCC.NorESM2-MM.historical.Amon.gn','CMIP.NIMS-KMA.UKESM1-0-LL.historical.Amon.gn','CMIP.NUIST.NESM3.historical.Amon.gn','CMIP.SNU.SAM0-UNICON.historical.Amon.gn','CMIP.UA.MCM-UA-1-0.historical.Amon.gn'])

# Total list
key_list = np.concatenate([key_list1,key_list2,key_list3,key_list4])

#%% ERA5 E - P data
# Load in data and determine noise (for the period 1940 - 2014)
var = xr.open_mfdataset(dir_era+'temp_1940_2022_monthly.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
T2m_era5 = var['t2m'][:900,:,:]*A_mask_era*1-273 # Select same time period in ERA5 as in CMIP6 model, mask out Atlantic and convert from Kelvin to degrees C

T2m = T2m_era5-T2m_era5.rolling(time=60, center=True,min_periods=12).mean().compute()   # Detrend by substracting 5 year running mean
noise_era5 = T2m.groupby('time.month')-T2m.groupby('time.month').mean('time').compute() # Deseasonalize

# Determine statistics noise (ERA5)
noise_mean_e5 = noise_era5.mean('time').compute()                                   # Noise mean
noise_std_e5 = noise_era5.std('time').compute()                                     # Noise standard deviation
noise_skw_e5 = noise_era5.reduce(func=scipy.stats.skew,dim='time').compute()        # Noise skewness
noise_kur_e5 = noise_era5.reduce(func=scipy.stats.kurtosis,dim='time').compute()    # Noise excess kurtosis

# Determine statstics data (ERA5; data == original data without detrending and deseasonalizing)
data_mean_e5 = T2m_era5.mean('time')                                                # Data mean
data_std_e5 = T2m_era5.std('time')                                                  # Data standard deviation
data_skw_e5 = T2m_era5.reduce(func=scipy.stats.skew,dim='time')                     # Data skewness
data_kur_e5 = T2m_era5.reduce(func=scipy.stats.kurtosis,dim='time')                 # Data excess kurtosis

#%% Construct arrays for the different metrics (size = number of CMIP6 models)
# Correlation
corr_noise_mean = np.zeros((key_list.shape))
corr_noise_std = np.zeros((key_list.shape))
corr_noise_skw = np.zeros((key_list.shape))
corr_noise_kur = np.zeros((key_list.shape))

corr_data_mean = np.zeros((key_list.shape))
corr_data_std = np.zeros((key_list.shape))
corr_data_skw = np.zeros((key_list.shape))
corr_data_kur = np.zeros((key_list.shape))

# Taylor Skill Score (TSS); not used
TSS_noise_mean = np.zeros((key_list.shape))
TSS_noise_std = np.zeros((key_list.shape))
TSS_noise_skw = np.zeros((key_list.shape))
TSS_noise_kur = np.zeros((key_list.shape))

TSS_data_mean = np.zeros((key_list.shape))
TSS_data_std = np.zeros((key_list.shape))
TSS_data_skw = np.zeros((key_list.shape))
TSS_data_kur = np.zeros((key_list.shape))

# Bias; not used
bias_noise_mean = np.zeros((key_list.shape))
bias_noise_std = np.zeros((key_list.shape))
bias_noise_skw = np.zeros((key_list.shape))
bias_noise_kur = np.zeros((key_list.shape))

bias_data_mean = np.zeros((key_list.shape))
bias_data_std = np.zeros((key_list.shape))
bias_data_skw = np.zeros((key_list.shape))
bias_data_kur = np.zeros((key_list.shape))

# RMSE
rmse_noise_mean = np.zeros((key_list.shape))
rmse_noise_std = np.zeros((key_list.shape))
rmse_noise_skw = np.zeros((key_list.shape))
rmse_noise_kur = np.zeros((key_list.shape))

rmse_data_mean = np.zeros((key_list.shape))
rmse_data_std = np.zeros((key_list.shape))
rmse_data_skw = np.zeros((key_list.shape))
rmse_data_kur = np.zeros((key_list.shape))

# Standard deviation
std_noise_mean = np.zeros((key_list.shape))
std_noise_std = np.zeros((key_list.shape))
std_noise_skw = np.zeros((key_list.shape))
std_noise_kur = np.zeros((key_list.shape))

std_data_mean = np.zeros((key_list.shape))
std_data_std = np.zeros((key_list.shape))
std_data_skw = np.zeros((key_list.shape))
std_data_kur = np.zeros((key_list.shape))

#%% Determine metrics
for key_i in range(len(key_list)):  # Loop over models
    # Select model
    key = key_list[key_i]         

    # Name of the model
    x = [a for a in key.split('.') if a]
    model = ('.'.join(x[2:3]))
    print(model)
    
    # Load in data and determine noise
    [T2m_cmip6,noise_cmip6] = determine_noise(key)
    
    # Determine statistics noise per model
    noise_mean = noise_cmip6.mean('time').squeeze().compute()
    noise_std = noise_cmip6.std('time').squeeze().compute()
    noise_skw = noise_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze().compute()
    noise_kur = noise_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze().compute()
    
    # Determine statistics data per model
    data_mean = T2m_cmip6.mean('time').squeeze().compute()
    data_std = T2m_cmip6.std('time').squeeze().compute()
    data_skw = T2m_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze().compute()
    data_kur = T2m_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze().compute()
    
    # Determine metrics noise per model
    [corr_noise_mean[key_i],TSS_noise_mean[key_i], bias_noise_mean[key_i], rmse_noise_mean[key_i], std_noise_mean[key_i], std_noise_mean_E5] = metrics(noise_mean,noise_mean_e5)
    [corr_noise_std[key_i],TSS_noise_std[key_i], bias_noise_std[key_i], rmse_noise_std[key_i], std_noise_std[key_i], std_noise_std_E5] = metrics(noise_std,noise_std_e5)
    [corr_noise_skw[key_i],TSS_noise_skw[key_i], bias_noise_skw[key_i], rmse_noise_skw[key_i], std_noise_skw[key_i], std_noise_skw_E5] = metrics(noise_skw,noise_skw_e5)
    [corr_noise_kur[key_i],TSS_noise_kur[key_i], bias_noise_kur[key_i], rmse_noise_kur[key_i], std_noise_kur[key_i], std_noise_kur_E5] = metrics(noise_kur,noise_kur_e5)
    
    # Determine metrics data per model
    [corr_data_mean[key_i],TSS_data_mean[key_i], bias_data_mean[key_i], rmse_data_mean[key_i], std_data_mean[key_i], std_data_mean_E5] = metrics(data_mean,data_mean_e5)
    [corr_data_std[key_i],TSS_data_std[key_i], bias_data_std[key_i], rmse_data_std[key_i], std_data_std[key_i], std_data_std_E5] = metrics(data_std,data_std_e5)
    [corr_data_skw[key_i],TSS_data_skw[key_i], bias_data_skw[key_i], rmse_data_skw[key_i], std_data_skw[key_i], std_data_skw_E5] = metrics(data_skw,data_skw_e5)
    [corr_data_kur[key_i],TSS_data_kur[key_i], bias_data_kur[key_i], rmse_data_kur[key_i], std_data_kur[key_i], std_data_kur_E5] = metrics(data_kur,data_kur_e5)

#%% Save the metrics
data_list_d = (['cdm','cdst','cdsk','cdk','tdm','tdst','tdsk','tdk','bdm','bdst','bdsk','bdk','rdm','rdst','rdsk','rdk','sdm','sdst','sdsk','sdk'])
data_list_n = (['cnm','cnst','cnsk','cnk','tnm','tnst','tnsk','tnk','bnm','bnst','bnsk','bnk','rnm','rnst','rnsk','rnk','snm','snst','snsk','snk'])

data_d =np.reshape(np.concatenate([corr_data_mean,corr_data_std,corr_data_skw,corr_data_kur,TSS_data_mean,TSS_data_std,TSS_data_skw,TSS_data_kur,bias_data_mean,bias_data_std,bias_data_skw,bias_data_kur,rmse_data_mean,rmse_data_std,rmse_data_skw,rmse_data_kur,std_data_mean,std_data_std,std_data_skw,std_data_kur],axis=0),(20,36))
data_n =np.reshape(np.concatenate([corr_noise_mean,corr_noise_std,corr_noise_skw,corr_noise_kur,TSS_noise_mean,TSS_noise_std,TSS_noise_skw,TSS_noise_kur,bias_noise_mean,bias_noise_std,bias_noise_skw,bias_noise_kur,rmse_noise_mean,rmse_noise_std,rmse_noise_skw,rmse_noise_kur,std_noise_mean,std_noise_std,std_noise_skw,std_noise_kur],axis=0),(20,36))

ser_d = pd.DataFrame(np.round(data_d,2), index=[data_list_d])
ser_n = pd.DataFrame(np.round(data_n,2), index=[data_list_n])

ser_d.to_csv(dir_cmip+'metrics_data_t2m_1deg.csv', index=True)
ser_n.to_csv(dir_cmip+'metrics_noise_t2m_1deg.csv', index=True)
    
    