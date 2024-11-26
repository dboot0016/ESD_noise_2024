# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure A16c, h for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import numpy as np
from scipy.stats import norm, norminvgauss
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
from scipy.special import kv  # Modified Bessel function of the second kind

#%% Define functions
def nig_logpdf(x, alpha, beta, mu, delta):
    gamma = np.sqrt(alpha**2 - beta**2)
    term1 = delta * np.exp(delta * gamma)
    term2 = kv(1, alpha * np.sqrt(delta**2 + (x - mu)**2))
    term3 = np.exp(beta * (x - mu))
    return np.log(term1 * term2 * term3) - np.log(np.pi * gamma)

def genderfaun_nowhite():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#E38D00","#FCD689","#FFF09B","#FAF9CD","#8EDED9","#8CACDE","#9782EC",'#1F3554'])

def genderfaun_nowhite_r():
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1F3554","#9782EC","#8CACDE","#8EDED9","#FAF9CD","#FFF09B","#FCD689",'#E38D00'])

#%% Directories
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'

#%% T2m noise
data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()
params = np.load(dir_era+'params_era_t2m_nig.npy')

#%% Calculate AIC and BIC
k_gaussian = 2                      # Number of parameters Gaussian distribution
k_nig = 4                           # Number of parameters NIG distribution

aic_gaussian = np.zeros((721,521))
bic_gaussian = np.zeros((721,521))
aic_nig = np.zeros((721,521))
bic_nig = np.zeros((721,521))

for lat_i in range(721):
    print(lat_i)
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            aic_gaussian[lat_i,lon_j] = np.nan
            bic_gaussian[lat_i,lon_j] = np.nan
            aic_nig[lat_i,lon_j] = np.nan
            bic_nig[lat_i,lon_j] = np.nan
        
        else:
            mu_gaussian, std_gaussian  = norm.fit(noise[:,lat_i,lon_j])
            log_likelihood_gaussian = np.sum(norm.logpdf(noise[:,lat_i,lon_j], mu_gaussian, std_gaussian))
            
            n = len(noise[:,lat_i,lon_j])  # number of observations
            aic_gaussian[lat_i,lon_j] = 2 * k_gaussian - 2 * log_likelihood_gaussian
            bic_gaussian[lat_i,lon_j] = k_gaussian * np.log(n) - 2 * log_likelihood_gaussian
            
            log_likelihood_nig = np.sum(norminvgauss.logpdf(noise[:,lat_i,lon_j], *params[:,lat_i,lon_j]))
            
            aic_nig[lat_i,lon_j] = 2 * k_nig - 2 * log_likelihood_nig
            bic_nig[lat_i,lon_j] = k_nig * np.log(n) - 2 * log_likelihood_nig

#%% Determine difference AIC and BIC scores
d_aic = aic_gaussian - aic_nig
d_bic = bic_gaussian - bic_nig

#%% Plotting variables
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

A_mask_era = noise[0,:,:]/noise[0,:,:]

target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

#%% 
fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,d_bic*A_mask_era ,transform=ccrs.PlateCarree(),vmin=-0.01,vmax=0.01,cmap='Greys_r')

ax.set_title('T$_{2m}$ ($\Delta$BIC)',fontsize = FS)
# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
plt.savefig(dir_fig+'Figure_A16h.png', format='png', dpi=quality,bbox_inches='tight')

#%% Determine percentage of grid points where NIG is a better fit than a Gaussian distribution (T2m)
count_greater_than_zero = np.sum(d_aic > 0)
count_less_than_zero = np.sum(d_aic < 0)

ratio_aic = count_greater_than_zero/(count_greater_than_zero+count_less_than_zero)*100

count_greater_than_zero = np.sum(d_bic > 0)
count_less_than_zero = np.sum(d_bic < 0)

ratio_bic = count_greater_than_zero/(count_greater_than_zero+count_less_than_zero)*100

#%% Load in E - P noise
data_ep = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise = data_ep['__xarray_dataarray_variable__'].compute()
params = np.load(dir_era+'params_era_ep_nig.npy')

#%% Calculate AIC and BIC
k_gaussian = 2                      # Number of parameters Gaussian distribution
k_nig = 4                           # Number of parameters NIG distribution

aic_gaussian = np.zeros((721,521))
bic_gaussian = np.zeros((721,521))
aic_nig = np.zeros((721,521))
bic_nig = np.zeros((721,521))

for lat_i in range(721):
    print(lat_i)
    for lon_j in range(521):
        if np.isnan(noise[0,lat_i,lon_j]):
            aic_gaussian[lat_i,lon_j] = np.nan
            bic_gaussian[lat_i,lon_j] = np.nan
            aic_nig[lat_i,lon_j] = np.nan
            bic_nig[lat_i,lon_j] = np.nan
        
        else:
            mu_gaussian, std_gaussian  = norm.fit(noise[:,lat_i,lon_j])
            log_likelihood_gaussian = np.sum(norm.logpdf(noise[:,lat_i,lon_j], mu_gaussian, std_gaussian))
            
            n = len(noise[:,lat_i,lon_j])  # number of observations
            aic_gaussian[lat_i,lon_j] = 2 * k_gaussian - 2 * log_likelihood_gaussian
            bic_gaussian[lat_i,lon_j] = k_gaussian * np.log(n) - 2 * log_likelihood_gaussian
            
            log_likelihood_nig = np.sum(norminvgauss.logpdf(noise[:,lat_i,lon_j], *params[:,lat_i,lon_j]))
            
            aic_nig[lat_i,lon_j] = 2 * k_nig - 2 * log_likelihood_nig
            bic_nig[lat_i,lon_j] = k_nig * np.log(n) - 2 * log_likelihood_nig

#%% Determine difference in AIC and BIC values between Gaussian and NIG distributions
d_aic = aic_gaussian - aic_nig
d_bic = bic_gaussian - bic_nig

#%% Plot Figure A16c
fig = plt.figure(figsize=(4, 7))
# Select extent axes and projection
ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())

# Plot data
im=plt.pcolormesh(lon,lat,d_bic*A_mask_era ,transform=ccrs.PlateCarree(),vmin=-0.01,vmax=0.01,cmap='Greys_r')

ax.set_title('E - P ($\Delta$BIC)',fontsize = FS)
# Set specifics of the plotting background
ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
ax.add_feature(cfeature.OCEAN)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
plt.savefig(dir_fig+'Figure_A16c.png', format='png', dpi=quality,bbox_inches='tight')

#%% Determine percentage of grid points where NIG is a better fit than a Gaussian distribution (E - P)
count_greater_than_zero = np.sum(d_aic > 0)
count_less_than_zero = np.sum(d_aic < 0)

ratio_aic = count_greater_than_zero/(count_greater_than_zero+count_less_than_zero)*100

count_greater_than_zero = np.sum(d_bic > 0)
count_less_than_zero = np.sum(d_bic < 0)

ratio_bic = count_greater_than_zero/(count_greater_than_zero+count_less_than_zero)*100




