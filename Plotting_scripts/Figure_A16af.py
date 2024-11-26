# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure A16a, f for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
from scipy.stats import norminvgauss
from scipy.stats import norm

#%% Directories
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'

#%% Load in T2m data
data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()
params = np.load(dir_era+'params_era_t2m_nig.npy')

#%% Determine chi
nbins = 50
chi_nig = np.zeros((721,521))
chi_nor = np.zeros((721,521))
chi_nig_n = np.zeros((721,521))
chi_nor_n = np.zeros((721,521))

for lat_i in range(721):
    print(lat_i)
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            chi_nig[lat_i,lon_j] = np.nan
            chi_nor[lat_i,lon_j] = np.nan
            chi_nig_n[lat_i,lon_j] = np.nan
            chi_nor_n[lat_i,lon_j] = np.nan
        
        else:
            [count,bins] = np.histogram(noise[:,lat_i,lon_j],bins = nbins, density = True)
            bin_mid = np.diff(bins)[0]/2+bins[:-1]
            pdf_nig = norminvgauss.pdf(bin_mid,*params[:,lat_i,lon_j])
            
            gauss_fit = norm.fit(noise[:,lat_i,lon_j])
            pdf_nor = norm.pdf(bin_mid,*gauss_fit)
        
            chi_nig[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nig - count)**2))
            chi_nig_n[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nig - count)**2/pdf_nig**2))
            
            chi_nor[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nor - count)**2))
            chi_nor_n[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nor - count)**2/pdf_nor**2))

#%% Manipulate data before plotting
dchi = np.zeros((721,521))
dchi_n = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        if chi_nig[lat_i,lon_j] < chi_nor[lat_i,lon_j]:
            dchi[lat_i,lon_j] = 1
        if chi_nig_n[lat_i,lon_j] < chi_nor_n[lat_i,lon_j]:
            dchi_n[lat_i,lon_j] = 1
            
#%% Plotting variables
target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

#%% Plot
A_mask_era = noise[0,:,:]/noise[0,:,:]

fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,dchi_n*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')

ax.set_title('T$_{2m}$ ($\chi_n$)',fontsize = FS)
# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
plt.savefig(dir_fig+'Figure_A16f.png', format='png', dpi=quality,bbox_inches='tight')

#%% Directories + plotting variables
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

#%% Load in E - P data
data_t2m = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()
params = np.load(dir_era+'params_era_ep_nig.npy')

#%% Determine chi
nbins = 50
chi_nig = np.zeros((721,521))
chi_nor = np.zeros((721,521))
chi_nig_n = np.zeros((721,521))
chi_nor_n = np.zeros((721,521))

for lat_i in range(721):
    print(lat_i)
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            chi_nig[lat_i,lon_j] = np.nan
            chi_nor[lat_i,lon_j] = np.nan
            chi_nig_n[lat_i,lon_j] = np.nan
            chi_nor_n[lat_i,lon_j] = np.nan
        
        else:
            [count,bins] = np.histogram(noise[:,lat_i,lon_j],bins = nbins, density = True)
            bin_mid = np.diff(bins)[0]/2+bins[:-1]
            pdf_nig = norminvgauss.pdf(bin_mid,*params[:,lat_i,lon_j])
            
            gauss_fit = norm.fit(noise[:,lat_i,lon_j])
            pdf_nor = norm.pdf(bin_mid,*gauss_fit)
        
            chi_nig[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nig - count)**2))
            chi_nig_n[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nig - count)**2/pdf_nig**2))
            
            chi_nor[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nor - count)**2))
            chi_nor_n[lat_i,lon_j] = 1/nbins *(np.sum((pdf_nor - count)**2/pdf_nor**2))

#%% Manipulate data before plotting
ep_dchi = np.zeros((721,521))
ep_dchi_n = np.zeros((721,521))

for lat_i in range(721):
    for lon_j in range(521):
        if chi_nig[lat_i,lon_j] < chi_nor[lat_i,lon_j]:
            ep_dchi[lat_i,lon_j] = 1
        if chi_nig_n[lat_i,lon_j] < chi_nor_n[lat_i,lon_j]:
            ep_dchi_n[lat_i,lon_j] = 1

#%% Plot
A_mask_era = noise[0,:,:]/noise[0,:,:]

fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,ep_dchi_n*A_mask_era ,transform=ccrs.PlateCarree(),vmin=0.99,vmax=1,cmap='Greys_r')

ax.set_title('E - P ($\chi_n$)',fontsize = FS)
# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
plt.savefig(dir_fig+'Figure_A16a.png', format='png', dpi=quality,bbox_inches='tight')
