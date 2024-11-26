# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script to plot Figure A16b, g for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
from scipy import stats

#%%  Directories + plotting variables
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

data_t2m = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()

target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

A_mask_era = noise[0,:,:]/noise[0,:,:]

#%% Perform AD test
ad_stat = np.zeros((721,521))
cv = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        # Perform Anderson-Darling test for normality
        result = stats.anderson(np.array(noise[:,lat_i,lon_j]), dist='norm')
        
        ad_stat[lat_i,lon_j] = result.statistic
        cv[lat_i,lon_j] = result.critical_values[2]

#%% Manipulate data before plotting
A = np.zeros((721,521))
for lat_i in range(721):
    for lon_j in range(521):
        if ad_stat[lat_i,lon_j] > cv[lat_i,lon_j]:
            A[lat_i,lon_j] = 1

#%% Save data
np.save('EP_noise_gauss_AD_test_sign.npy',A)
#A = np.load('EP_noise_gauss_AD_test_sign.npy')

#%% Plot
fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,A*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')

ax.set_title('E - P (AD - test)',fontsize = FS)
# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
plt.savefig(dir_fig+'Figure_A16b.png', format='png', dpi=quality,bbox_inches='tight')

#%% Load in T2m noise
data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()

#%% Perform AD test
ad_stat = np.zeros((721,521))
cv = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        # Perform Anderson-Darling test for normality
        result = stats.anderson(np.array(noise[:,lat_i,lon_j]), dist='norm')
        
        ad_stat[lat_i,lon_j] = result.statistic
        cv[lat_i,lon_j] = result.critical_values[0]

#%% Manipulate data before plotting
A = np.zeros((721,521))
for lat_i in range(721):
    for lon_j in range(521):
        if ad_stat[lat_i,lon_j] > cv[lat_i,lon_j]:
            A[lat_i,lon_j] = 1

#%% Save data
np.save('T2m_noise_gauss_AD_test_sign.npy',A)
#A = np.load('T2m_noise_gauss_AD_test_sign.npy')

#%% Plot
fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,A*A_mask_era,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')
ax.set_title('T$_{2m}$ (AD - test)',fontsize = FS)

# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
plt.savefig(dir_fig+'Figure_A16g.png', format='png', dpi=quality,bbox_inches='tight')


