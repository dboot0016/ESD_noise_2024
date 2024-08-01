# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for constructing ERA5 noise for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Geophysical Research Letters' on 01-08-2024

#%% Load in modules
import xarray as xr 

#%%
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# e_p_1940_2022_monthly_era.nc can be constructed from data downloaded from the CDS (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview)
# Download variables evaporation and total precipitation on a monthly frequency from 1940 to 2022
# Sum the two (units are m/day)
var = xr.open_mfdataset(dir_era+'e_p_1940_2022_monthly_era.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1}) 
P_E = -var['e-p'][:,:,:]*A_mask_era*1e3 # Minus to convert P - E to E - P, and convert m/day to mm/day

PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()          # Detrend by removing 5 year moving mean
noise_ep = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute() # Deseasonalize

# e_p_1940_2022_monthly_era.nc can be constructed from data downloaded from the CDS (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview)
# Download variable 2m air temperature on a monthly frequency from 1940 to 2022
# Unit are Kelvin
var = xr.open_mfdataset(dir_era+'temp_1940_2022_monthly.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1}) 
T2m = var['t2m'][:-1,:,:]*A_mask_era-273 # Convert from Kelvin to degrees C

t2m = T2m-T2m.rolling(time=60, center=True,min_periods=12).mean().compute()          # Detrend by removing 5 year moving mean
noise_t2m = t2m.groupby('time.month')-t2m.groupby('time.month').mean('time').compute() # Deseasonalize

#%% Save datasets
noise_ep.to_netcdf(dir_era+'noise_ep_era5.nc')
noise_t2m.to_netcdf(dir_era+'noise_t2m_era5.nc')