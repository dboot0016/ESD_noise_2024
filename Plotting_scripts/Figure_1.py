# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure 1 for ' Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm
import matplotlib

#%% Functions for analysis and plotting
def determine_noise_stats(var):
    # Noise is determined by removing a 5 year running mean and subsequently removing the seasonal cycle.
    # Original E - P data is in m/day
    # Original T2m is in K
    # For both the data as the noise statistics are determined
    
    if var == 'e-p':
        # Load in data; important: the original dataset is P minus E explaining the minus sign
        # The Atlantic between 80N and 60S is selected through the mask
        var = xr.open_mfdataset(dir_era+'e_p_1940_2022_monthly_era.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
        P_E = -var['e-p'][:,:,:]*A_mask_era 
        
        # Detrend with running mean of 60 months (at least 12 months necessary)
        PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()
        # Remove seasonal cycle
        noise = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()
        
    elif var == 't2m':
        # Load in data
        # The Atlantic between 80N and 60S is selected through the mask
        var = xr.open_mfdataset(dir_era+'temp_1940_2022_monthly.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
        P_E = var['t2m'][:-1,:,:]*A_mask_era-273

        # Detrend with running mean of 60 months (at least 12 months necessary)
        PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()
        # Remove seasonal cycle
        noise = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()
    
    # Determine the statistics of the data (without detrending and deseasonalizing; not used)
    mean_data = P_E.mean(dim='time')
    std_data = P_E.std(dim='time')
    skw_data = P_E.reduce(func=scipy.stats.skew,dim='time')
    kur_data = P_E.reduce(func=scipy.stats.kurtosis,dim='time')
    
    # Determine the statistics of the noise
    std_noise = noise.std(dim='time')
    skw_noise = noise.reduce(func=scipy.stats.skew,dim='time')
    kur_noise = noise.reduce(func=scipy.stats.kurtosis,dim='time')
    
    return mean_data, std_data, skw_data, kur_data, std_noise, skw_noise, kur_noise

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

#%%
# Directories where data is + where figures should be saved to
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Figures_sub/'

# Load in the mask of the Atlantic
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# Set grid for plotting
target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

# Determine noise and data statistics ERA5
[mean_data_ep,std_data_ep,skw_data_ep,kur_data_ep,std_noise_ep,skw_noise_ep,kur_noise_ep] = determine_noise_stats('e-p')
[mean_data_t2m,std_data_t2m,skw_data_t2m,kur_data_t2m,std_noise_t2m,skw_noise_t2m,kur_noise_t2m] = determine_noise_stats('t2m')

#%% Plot figures
FS = 20 # Base fontsize
quality = 300 # Quality in dpi for figures

#%% Noise E - P
fig = plt.figure(figsize=(4,7))
plot_stat_2d(std_noise_ep*1e3,'$\sigma$ (E-P) [mm/day]','ERA5: noise',0,3.1,cm.cm.deep_r)

plt.savefig(dir_fig+'Figure_1_a.png', format='png', dpi=quality,bbox_inches='tight')

fig = plt.figure(figsize=(4,7))
plot_stat_2d(skw_noise_ep,'Skewness (E-P) [-]','ERA5: noise',-2.1,2.1,aroace())

plt.savefig(dir_fig+'Figure_1_b.png', format='png', dpi=quality,bbox_inches='tight')

fig = plt.figure(figsize=(4,7))
plot_stat_2d(kur_noise_ep,'Kurtosis (E-P) [-]','ERA5: noise',-5.1,5.1,aroace())

plt.savefig(dir_fig+'Figure_1_c.png', format='png', dpi=quality,bbox_inches='tight')

#%% Noise T2m
fig = plt.figure(figsize=(4,7))
plot_stat_2d(std_noise_t2m,'$\sigma$ (T$_{2m}$) [$^{\circ}$C]','ERA5: noise',0,1.75,cm.cm.deep_r)

plt.savefig(dir_fig+'Figure_1_d.png', format='png', dpi=quality,bbox_inches='tight')

fig = plt.figure(figsize=(4,7))
plot_stat_2d(skw_noise_t2m,'Skewness (T$_{2m}$) [-]','ERA5: noise',-1.15,1.15,aroace())

plt.savefig(dir_fig+'Figure_1_e.png', format='png', dpi=quality,bbox_inches='tight')

fig = plt.figure(figsize=(4,7))
plot_stat_2d(kur_noise_t2m,'Kurtosis (T$_{2m}$) [-]','ERA5: noise',-1.75,1.75,aroace())

plt.savefig(dir_fig+'Figure_1_f.png', format='png', dpi=quality,bbox_inches='tight')

    
    