# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure A16d, e, i ,j for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import statsmodels.api as sm

#%% Direcotries + load in T2m noise
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()

#%% Define empty arrays
size_x = 997
fitted_samples = np.zeros((997,721,521))

ks_test_fit = np.zeros((2,721,521))
jb_test_fit = np.zeros((4,721,521))

ks_test_sim = np.zeros((2,721,521))
jb_test_sim = np.zeros((4,721,521))

mean_data = np.zeros((100,721,521))
skw_data = np.zeros((100,721,521))
std_data = np.zeros((100,721,521))
kur_data = np.zeros((100,721,521))

n_simulations = 100

#%% Fit AR(1) model and generate samples
for lat_i in range(721):
    print(lat_i)
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            continue
        else:
            model = sm.tsa.ARIMA(np.array(noise[:,lat_i,lon_j].squeeze()), order=(1, 0, 0))  # AR(1) model
            fit = model.fit()
            
            for sim_i in range(n_simulations):
                # Simulate AR(1) process using the estimated parameters
                simulated = fit.get_prediction(start=997, dynamic=False)
                simulated_samples = simulated.predicted_mean + np.random.normal(0, np.sqrt(fit.scale), size=997)

                mean_data[sim_i,lat_i,lon_j] = np.nanmean(simulated_samples,axis=0)
                std_data[sim_i,lat_i,lon_j] = np.nanstd(simulated_samples,axis = 0)
                skw_data[sim_i,lat_i,lon_j] = scipy.stats.skew(simulated_samples,axis=0)
                kur_data[sim_i,lat_i,lon_j] = scipy.stats.kurtosis(simulated_samples,axis=0)  
        
#%% Determine p values
original_skewness = scipy.stats.skew(noise,axis=0)
original_kurtosis = scipy.stats.kurtosis(noise,axis=0)

p_skw = np.zeros((721,521))
p_kur = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            p_skw[lat_i,lon_j] = np.nan 
            p_kur[lat_i,lon_j] = np.nan
        else:
            p_skw[lat_i,lon_j] = np.mean(np.abs(skw_data[:,lat_i,lon_j]) >= np.abs(original_skewness[lat_i,lon_j]))
            p_kur[lat_i,lon_j] = np.mean(np.abs(kur_data[:,lat_i,lon_j]) >= np.abs(original_kurtosis[lat_i,lon_j]))

#%%
np.save('t2m_skw_pvalue_AR1.npy',p_skw)
np.save('t2m_kur_pvalue_AR1.npy',p_kur)

#%%
#p_skw = np.load('t2m_skw_pvalue_AR1.npy')
#p_kur = np.load('t2m_kur_pvalue_AR1.npy')

#%% Plotting variables
target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

#%% Manipulate data for plotting
sign_skw = np.zeros((721,521))
sign_kur = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            sign_skw[lat_i,lon_j] = np.nan
            sign_kur[lat_i,lon_j] = np.nan
        if p_skw[lat_i,lon_j] < 0.05:
            sign_skw[lat_i,lon_j] = 1
        if p_kur[lat_i,lon_j] < 0.05:
            sign_kur[lat_i,lon_j]  =1

A_mask_era = noise[0,:,:]/noise[0,:,:]

#%%
fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

plt.title('T$_{2m}$ (AR (1); skewness)',fontsize=FS)

# Plot data
im=plt.pcolormesh(lon,lat,sign_skw*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')

# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

plt.savefig(dir_fig+'Figure_A16i.png', format='png', dpi=quality,bbox_inches='tight')

fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

plt.title('T$_{2m}$ (AR (1); kurtosis)',fontsize=FS)

# Plot data
im=plt.pcolormesh(lon,lat,sign_kur*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')

# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

plt.savefig(dir_fig+'Figure_A16j.png', format='png', dpi=quality,bbox_inches='tight')

#%% Load E - P noise
data_t2m = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()

#%% Set up empty arrays
size_x = 996
fitted_samples = np.zeros((996,721,521))

ks_test_fit = np.zeros((2,721,521))
jb_test_fit = np.zeros((4,721,521))

ks_test_sim = np.zeros((2,721,521))
jb_test_sim = np.zeros((4,721,521))

mean_data = np.zeros((100,721,521))
skw_data = np.zeros((100,721,521))
std_data = np.zeros((100,721,521))
kur_data = np.zeros((100,721,521))

n_simulations = 100

#%% Fit AR1 model + make simulated samples
for lat_i in range(721):
    print(lat_i)
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            continue
        else:
            model = sm.tsa.ARIMA(np.array(noise[:,lat_i,lon_j].squeeze()), order=(1, 0, 0))  # AR(1) model
            fit = model.fit()
            
            for sim_i in range(n_simulations):
                # Simulate AR(1) process using the estimated parameters
                simulated = fit.get_prediction(start=996, dynamic=False)
                simulated_samples = simulated.predicted_mean + np.random.normal(0, np.sqrt(fit.scale), size=996)

                mean_data[sim_i,lat_i,lon_j] = np.nanmean(simulated_samples,axis=0)
                std_data[sim_i,lat_i,lon_j] = np.nanstd(simulated_samples,axis = 0)
                skw_data[sim_i,lat_i,lon_j] = scipy.stats.skew(simulated_samples,axis=0)
                kur_data[sim_i,lat_i,lon_j] = scipy.stats.kurtosis(simulated_samples,axis=0)  
        
#%% Determine p-values
original_skewness = scipy.stats.skew(noise,axis=0)
original_kurtosis = scipy.stats.kurtosis(noise,axis=0)

p_skw = np.zeros((721,521))
p_kur = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            p_skw[lat_i,lon_j] = np.nan 
            p_kur[lat_i,lon_j] = np.nan
        else:
            p_skw[lat_i,lon_j] = np.mean(np.abs(skw_data[:,lat_i,lon_j]) >= np.abs(original_skewness[lat_i,lon_j]))
            p_kur[lat_i,lon_j] = np.mean(np.abs(kur_data[:,lat_i,lon_j]) >= np.abs(original_kurtosis[lat_i,lon_j]))


#%%
np.save('ep_skw_pvalue_AR1.npy',p_skw)
np.save('ep_kur_pvalue_AR1.npy',p_kur)

#%%
#p_skw = np.load('ep_skw_pvalue_AR1.npy')
#p_kur = np.load('ep_kur_pvalue_AR1.npy')


#%% Manipulate data for plotting
sign_skw = np.zeros((721,521))
sign_kur = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            sign_skw[lat_i,lon_j] = np.nan
            sign_kur[lat_i,lon_j] = np.nan
        if p_skw[lat_i,lon_j] < 0.05:
            sign_skw[lat_i,lon_j] = 1
        if p_kur[lat_i,lon_j] < 0.05:
            sign_kur[lat_i,lon_j]  =1

A_mask_era = noise[0,:,:]/noise[0,:,:]

#%% Plot
fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,sign_skw*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')
plt.title('E - P (AR (1); skewness)',fontsize=FS)

# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

plt.savefig(dir_fig+'Figure_A16d.png', format='png', dpi=quality,bbox_inches='tight')

fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())
plt.title('E - P (AR (1); kurtosis)',fontsize=FS)
# Plot data
im=plt.pcolormesh(lon,lat,sign_kur*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')

# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

plt.savefig(dir_fig+'Figure_A16e.png', format='png', dpi=quality,bbox_inches='tight')
    
    