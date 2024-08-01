# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for to construct MMM for CMIP6 E - P 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Geophysical Research Letters' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import scipy.stats

#%% Functions used in scripts
def determine_noise(key):
    # Function to load data and determine noise in the CMIP6 models
    var = xr.open_mfdataset(dir_cmip+key+'_ep_new.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
    P_E = var['e-p'][:,:,:]*A_mask_cmip*conv_int_cmip
    
    PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()
    noise = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()
        
    return P_E, noise

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

# Units CMIP6: kg m-2 s-1 need to be transformed to mm/day
conv_int_cmip = 86400 * 1e3 / 1e3

#%% List of CMIP6 models
key_list1 = (['CMIP.AS-RCEC.TaiESM1.historical.Amon.gn','CMIP.AWI.AWI-CM-1-1-MR.historical.Amon.gn','CMIP.AWI.AWI-ESM-1-1-LR.historical.Amon.gn','CMIP.BCC.BCC-CSM2-MR.historical.Amon.gn','CMIP.BCC.BCC-ESM1.historical.Amon.gn','CMIP.CAS.FGOALS-g3.historical.Amon.gn','CMIP.CCCma.CanESM5-CanOE.historical.Amon.gn','CMIP.CAS.CAS-ESM2-0.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-HR4.historical.Amon.gn'])
key_list2 = (['CMIP.CCCma.CanESM5.historical.Amon.gn','CMIP.CCCR-IITM.IITM-ESM.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-SR5.historical.Amon.gn','CMIP.CMCC.CMCC-ESM2.historical.Amon.gn','CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Amon.gn','CMIP.CSIRO.ACCESS-ESM1-5.historical.Amon.gn','CMIP.FIO-QLNM.FIO-ESM-2-0.historical.Amon.gn','CMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.historical.Amon.gn','CMIP.MIROC.MIROC-ES2L.historical.Amon.gn'])
key_list3 = (['CMIP.MIROC.MIROC6.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-LL.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-MM.historical.Amon.gn','CMIP.MOHC.UKESM1-0-LL.historical.Amon.gn','CMIP.MPI-M.MPI-ESM1-2-LR.historical.Amon.gn','CMIP.MRI.MRI-ESM2-0.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-G-CC.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-H.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM-FV2.historical.Amon.gn'])
key_list4 = (['CMIP.NASA-GISS.GISS-E2-2-H.historical.Amon.gn','CMIP.NCAR.CESM2-FV2.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM.historical.Amon.gn','CMIP.NCAR.CESM2.historical.Amon.gn','CMIP.NCC.NorESM2-MM.historical.Amon.gn','CMIP.NIMS-KMA.UKESM1-0-LL.historical.Amon.gn','CMIP.NUIST.NESM3.historical.Amon.gn','CMIP.SNU.SAM0-UNICON.historical.Amon.gn','CMIP.UA.MCM-UA-1-0.historical.Amon.gn'])

key_list = np.concatenate([key_list1,key_list2,key_list3,key_list4])

#%% Select first model
key_i = 0
key = key_list[key_i]

[EP_cmip6,noise_cmip6] = determine_noise(key)

NM = noise_cmip6.mean('time').squeeze().compute()
NST = noise_cmip6.std('time').squeeze().compute()
NSK = noise_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze().compute()
NK = noise_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze().compute()

DM = EP_cmip6.mean('time').squeeze().compute()
DST = EP_cmip6.std('time').squeeze().compute()
DSK = EP_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze().compute()
DK = EP_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze().compute()

#%% Select second model
key_i = 1
key = key_list[key_i]

[EP_cmip6,noise_cmip6] = determine_noise(key)

noise_mean = noise_cmip6.mean('time').squeeze()
noise_std = noise_cmip6.std('time').squeeze()
noise_skw = noise_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze()
noise_kur = noise_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze()

data_mean = EP_cmip6.mean('time').squeeze()
data_std = EP_cmip6.std('time').squeeze()
data_skw = EP_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze()
data_kur = EP_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze()

#%% Concatenate the first and second model that serves as a base array
DM = xr.concat([DM,data_mean],dim='models')     # Data mean (data == original data without detrending and deseasonalizing)
DST = xr.concat([DST,data_std],dim='models')    # Data standard deviation
DSK = xr.concat([DSK,data_skw],dim='models')    # Data skewnees
DK = xr.concat([DK,data_kur],dim='models')      # Data excess kurtosis

NM = xr.concat([NM,noise_mean],dim='models')    # Noise mean
NST = xr.concat([NST,noise_std],dim='models')   # Noise standard deviation
NSK = xr.concat([NSK,noise_skw],dim='models')   # Noise skewness
NK = xr.concat([NK,noise_kur],dim='models')     # Noise excess kurtosis

#%% Concatenate the statistics of the other models
for key_i in range(len(key_list)-2):    # Loop over rest of models
    # Select model
    key = key_list[key_i+2]
    
    # Select model name
    x = [a for a in key.split('.') if a]
    model = ('.'.join(x[2:3])) 
    print(model)
    
    # Load in data and determine nose
    [EP_cmip6,noise_cmip6] = determine_noise(key)
    
    # Determine statistcs noise per model
    noise_mean = noise_cmip6.mean('time').squeeze().compute()
    noise_std = noise_cmip6.std('time').squeeze().compute()
    noise_skw = noise_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze().compute()
    noise_kur = noise_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze().compute()
    
    # Determine statistics data per model
    data_mean = EP_cmip6.mean('time').squeeze().compute()
    data_std = EP_cmip6.std('time').squeeze().compute()
    data_skw = EP_cmip6.reduce(func=scipy.stats.skew,dim='time').squeeze().compute()
    data_kur = EP_cmip6.reduce(func=scipy.stats.kurtosis,dim='time').squeeze().compute()
    
    # Concantenate the statistics to exisiting dataset
    NM = xr.concat([NM,noise_mean*xr.ones_like(NM[0,:,:])],dim='models')
    NST = xr.concat([NST,noise_std*xr.ones_like(NM[0,:,:])],dim='models')
    NSK = xr.concat([NSK,noise_skw*xr.ones_like(NM[0,:,:])],dim='models')
    NK = xr.concat([NK,noise_kur*xr.ones_like(NM[0,:,:])],dim='models')
    
    DM = xr.concat([DM,data_mean*xr.ones_like(DM[0,:,:])],dim='models')
    DST = xr.concat([DST,data_std*xr.ones_like(DM[0,:,:])],dim='models')
    DSK = xr.concat([DSK,data_skw*xr.ones_like(DM[0,:,:])],dim='models')
    DK = xr.concat([DK,data_kur*xr.ones_like(DM[0,:,:])],dim='models')

#%% Determine multi model mean
nm = NM.mean('models')      # Noise mean
nst = NST.mean('models')    # Noise standard deviation
nsk = NSK.mean('models')    # Noise skewness
nk = NK.mean('models')      # Noise excess kurtosis

dm = DM.mean('models')      # Data mean (data == original data without deseasonalizing and detrending)
dst = DST.mean('models')    # Data standard deviation
dsk = DSK.mean('models')    # Data skewness
dk = DK.mean('models')      # Data excess kurtosis

#%% Save multi model mean
nm.to_netcdf(dir_cmip+'noise_mean_mmm_ep.nc')
nst.to_netcdf(dir_cmip+'noise_std_mmm_ep.nc')
nsk.to_netcdf(dir_cmip+'noise_skw_mmm_ep.nc')
nk.to_netcdf(dir_cmip+'noise_kur_mmm_ep.nc')

dm.to_netcdf(dir_cmip+'data_mean_mmm_ep.nc')
dst.to_netcdf(dir_cmip+'data_std_mmm_ep.nc')
dsk.to_netcdf(dir_cmip+'data_skw_mmm_ep.nc')
dk.to_netcdf(dir_cmip+'data_kur_mmm_ep.nc')