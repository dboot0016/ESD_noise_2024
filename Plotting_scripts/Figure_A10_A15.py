# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure A5 - A10 for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
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
import xesmf as xe
import pandas as pd

#%% Functions used in the script
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
    
    # Determine the statistics of the data
    mean_data = P_E.mean(dim='time')
    std_data = P_E.std(dim='time')
    skw_data = P_E.reduce(func=scipy.stats.skew,dim='time')
    kur_data = P_E.reduce(func=scipy.stats.kurtosis,dim='time')
    
    # Determine the statistics of the noise
    std_noise = noise.std(dim='time')
    skw_noise = noise.reduce(func=scipy.stats.skew,dim='time')
    kur_noise = noise.reduce(func=scipy.stats.kurtosis,dim='time')
    
    return mean_data, std_data, skw_data, kur_data, std_noise, skw_noise, kur_noise

def regridder(ds,ds_out,dr):
    # Function to regrid data used to regrid CMIP6 models from 1 degree to 0.25 degree
    # Regridded data is masked by the land mask of ERA5 data (A_mask_era)
    regridder = xe.Regridder(ds.fillna(0), ds_out, "bilinear")    
    dr_out = regridder(dr.fillna(0), keep_attrs=True) * A_mask_era
    
    return dr_out

def determine_noise_stats_cmip(key,var):
    # Function to load data and determine noise in the CMIP6 models
    if var == 'e-p':
        var = xr.open_mfdataset(dir_cmip+'CMIP6_EP/'+key+'_ep_new.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
        P_E = var['e-p'][:,:,:]*A_mask_cmip*86400
        
        PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()
        noise = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()
        
    elif var == 't2m':
        # Function to load data and determine noise in the CMIP6 models
        var = xr.open_mfdataset(dir_cmip+'CMIP6_T2m/'+key+'_tas_new.nc', chunks={'latitude': 'auto', 'longitude': 'auto', 'time': -1})
        P_E = var['tas'][:,:,:]*A_mask_cmip-273
        
        PE = P_E-P_E.rolling(time=60, center=True,min_periods=12).mean().compute()
        noise = PE.groupby('time.month')-PE.groupby('time.month').mean('time').compute()
        
    # Determine the statistics of the data
    Mean_data = P_E.mean(dim='time')
    Std_data = P_E.std(dim='time')
    Skw_data = P_E.reduce(func=scipy.stats.skew,dim='time')
    Kur_data = P_E.reduce(func=scipy.stats.kurtosis,dim='time')
    
    # Determine the statistics of the noise
    Std_noise = noise.std(dim='time')
    Skw_noise = noise.reduce(func=scipy.stats.skew,dim='time')
    Kur_noise = noise.reduce(func=scipy.stats.kurtosis,dim='time')
    
    mean_data = regridder(Mean_data,mean_data_ep,Mean_data)
    std_data = regridder(Std_data,mean_data_ep,Std_data)
    skw_data = regridder(Skw_data,mean_data_ep,Skw_data)
    kur_data = regridder(Kur_data,mean_data_ep,Kur_data)
    
    std_noise = regridder(Std_noise,mean_data_ep,Std_noise)
    skw_noise = regridder(Skw_noise,mean_data_ep,Skw_noise)
    kur_noise = regridder(Kur_noise,mean_data_ep,Kur_noise)
    
    return mean_data.squeeze(), std_data.squeeze(), skw_data.squeeze(), kur_data.squeeze(), std_noise.squeeze(), skw_noise.squeeze(), kur_noise.squeeze()

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
    plt.pcolormesh(lon,lat,data,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,cmap=cmap)
    
    # Set title
    ax.set_title(model,fontsize=FS)
    
    # Set specifics of the plotting background
    ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
    ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
    gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

#%%
# Directories where ERA5 and CMIP6 data is + where figures should be saved to
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_cmip = '/Users/Boot0016/Documents/Project_noise_model/CMIP6_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Figures_sub/'

# Load in the mask of the Atlantic
mask_era = xr.open_dataset(dir_era+'era5_atlantic_mask_60S.nc')
A_mask_era = mask_era.lsm 

# CMIP6 land mask
mask_cmip = xr.open_dataset(dir_cmip+'cmip6_atlantic_mask_60S_new.nc')
A_mask_cmip = mask_cmip.mask 

# Grid used for plotting
target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

# Determine noise and data statistics ERA5 
[mean_data_ep,std_data_ep,skw_data_ep,kur_data_ep,std_noise_ep,skw_noise_ep,kur_noise_ep] = determine_noise_stats('e-p')
[mean_data_t2m,std_data_t2m,skw_data_t2m,kur_data_t2m,std_noise_t2m,skw_noise_t2m,kur_noise_t2m] = determine_noise_stats('t2m')

# Load in CMIP metrics
CMIP_metrics_ep = np.array(pd.read_csv(dir_cmip+'metrics_noise_ep_1deg.csv'))
CMIP_metrics_t2m = np.array(pd.read_csv(dir_cmip+'metrics_noise_t2m_1deg.csv'))
CMIP_metrics_ep_data = np.array(pd.read_csv(dir_cmip+'metrics_data_ep_1deg.csv'))
CMIP_metrics_t2m_data = np.array(pd.read_csv(dir_cmip+'metrics_data_t2m_1deg.csv'))

# Plotting variables
FS = 20 # Base fontsize
quality = 300 # Quality in dpi for figures

#%% List of models
key_list1 = (['CMIP.AS-RCEC.TaiESM1.historical.Amon.gn','CMIP.AWI.AWI-CM-1-1-MR.historical.Amon.gn','CMIP.AWI.AWI-ESM-1-1-LR.historical.Amon.gn','CMIP.BCC.BCC-CSM2-MR.historical.Amon.gn','CMIP.BCC.BCC-ESM1.historical.Amon.gn','CMIP.CAS.FGOALS-g3.historical.Amon.gn','CMIP.CCCma.CanESM5-CanOE.historical.Amon.gn','CMIP.CAS.CAS-ESM2-0.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-HR4.historical.Amon.gn'])
key_list2 = (['CMIP.CCCma.CanESM5.historical.Amon.gn','CMIP.CCCR-IITM.IITM-ESM.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-SR5.historical.Amon.gn','CMIP.CMCC.CMCC-ESM2.historical.Amon.gn','CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Amon.gn','CMIP.CSIRO.ACCESS-ESM1-5.historical.Amon.gn','CMIP.FIO-QLNM.FIO-ESM-2-0.historical.Amon.gn','CMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.historical.Amon.gn','CMIP.MIROC.MIROC-ES2L.historical.Amon.gn'])
key_list3 = (['CMIP.MIROC.MIROC6.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-LL.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-MM.historical.Amon.gn','CMIP.MOHC.UKESM1-0-LL.historical.Amon.gn','CMIP.MPI-M.MPI-ESM1-2-LR.historical.Amon.gn','CMIP.MRI.MRI-ESM2-0.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-G-CC.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-H.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM-FV2.historical.Amon.gn'])
key_list4 = (['CMIP.NASA-GISS.GISS-E2-2-H.historical.Amon.gn','CMIP.NCAR.CESM2-FV2.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM.historical.Amon.gn','CMIP.NCAR.CESM2.historical.Amon.gn','CMIP.NCC.NorESM2-MM.historical.Amon.gn','CMIP.NIMS-KMA.UKESM1-0-LL.historical.Amon.gn','CMIP.NUIST.NESM3.historical.Amon.gn','CMIP.SNU.SAM0-UNICON.historical.Amon.gn','CMIP.UA.MCM-UA-1-0.historical.Amon.gn'])

key_list = np.concatenate([key_list1,key_list2,key_list3,key_list4])

#%% Loop over models
for key_i in range(1):#len(key_list)):
    key = key_list[key_i]
    
    # Name of the model
    x = [a for a in key.split('.') if a]
    model = ('.'.join(x[2:3]))
    print(model)
    
    # Load in metrics 
    CMIP_corr_std_ep = CMIP_metrics_ep[1,key_i+1]
    CMIP_corr_skw_ep = CMIP_metrics_ep[2,key_i+1]
    CMIP_corr_kur_ep = CMIP_metrics_ep[3,key_i+1]
    
    CMIP_bias_std_ep = CMIP_metrics_ep[9,key_i+1]
    CMIP_bias_skw_ep = CMIP_metrics_ep[10,key_i+1]
    CMIP_bias_kur_ep = CMIP_metrics_ep[11,key_i+1]
    
    CMIP_rmse_std_ep = CMIP_metrics_ep[13,key_i+1]
    CMIP_rmse_skw_ep = CMIP_metrics_ep[14,key_i+1]
    CMIP_rmse_kur_ep = CMIP_metrics_ep[15,key_i+1]
    
    CMIP_corr_std_t2m = CMIP_metrics_t2m[1,key_i+1]
    CMIP_corr_skw_t2m = CMIP_metrics_t2m[2,key_i+1]
    CMIP_corr_kur_t2m = CMIP_metrics_t2m[3,key_i+1]
    
    CMIP_bias_std_t2m = CMIP_metrics_t2m[9,key_i+1]
    CMIP_bias_skw_t2m = CMIP_metrics_t2m[10,key_i+1]
    CMIP_bias_kur_t2m = CMIP_metrics_t2m[11,key_i+1]
    
    CMIP_rmse_std_t2m = CMIP_metrics_t2m[13,key_i+1]
    CMIP_rmse_skw_t2m = CMIP_metrics_t2m[14,key_i+1]
    CMIP_rmse_kur_t2m = CMIP_metrics_t2m[15,key_i+1]
    
    CMIP_corr_mean_ep_data = CMIP_metrics_ep_data[0,key_i+1]
    CMIP_corr_std_ep_data = CMIP_metrics_ep_data[1,key_i+1]
    CMIP_corr_skw_ep_data = CMIP_metrics_ep_data[2,key_i+1]
    CMIP_corr_kur_ep_data = CMIP_metrics_ep_data[3,key_i+1]
    
    CMIP_bias_mean_ep_data = CMIP_metrics_ep_data[8,key_i+1]
    CMIP_bias_std_ep_data = CMIP_metrics_ep_data[9,key_i+1]
    CMIP_bias_skw_ep_data = CMIP_metrics_ep_data[10,key_i+1]
    CMIP_bias_kur_ep_data = CMIP_metrics_ep_data[11,key_i+1]
    
    CMIP_rmse_mean_ep_data = CMIP_metrics_ep_data[12,key_i+1]
    CMIP_rmse_std_ep_data = CMIP_metrics_ep_data[13,key_i+1]
    CMIP_rmse_skw_ep_data = CMIP_metrics_ep_data[14,key_i+1]
    CMIP_rmse_kur_ep_data = CMIP_metrics_ep_data[15,key_i+1]
    
    CMIP_corr_mean_t2m_data = CMIP_metrics_t2m_data[0,key_i+1]
    CMIP_corr_std_t2m_data = CMIP_metrics_t2m_data[1,key_i+1]
    CMIP_corr_skw_t2m_data = CMIP_metrics_t2m_data[2,key_i+1]
    CMIP_corr_kur_t2m_data = CMIP_metrics_t2m_data[3,key_i+1]
    
    CMIP_bias_mean_t2m_data = CMIP_metrics_t2m_data[8,key_i+1]
    CMIP_bias_std_t2m_data = CMIP_metrics_t2m_data[9,key_i+1]
    CMIP_bias_skw_t2m_data = CMIP_metrics_t2m_data[10,key_i+1]
    CMIP_bias_kur_t2m_data = CMIP_metrics_t2m_data[11,key_i+1]
    
    CMIP_rmse_mean_t2m_data = CMIP_metrics_t2m_data[12,key_i+1]
    CMIP_rmse_std_t2m_data = CMIP_metrics_t2m_data[13,key_i+1]
    CMIP_rmse_skw_t2m_data = CMIP_metrics_t2m_data[14,key_i+1]
    CMIP_rmse_kur_t2m_data = CMIP_metrics_t2m_data[15,key_i+1]
    
    # Determine statistics 
    [c6_mean_data_ep,c6_std_data_ep,c6_skw_data_ep,c6_kur_data_ep,c6_std_noise_ep,c6_skw_noise_ep,c6_kur_noise_ep] = determine_noise_stats_cmip(key,'e-p')
    [c6_mean_data_t2m,c6_std_data_t2m,c6_skw_data_t2m,c6_kur_data_t2m,c6_std_noise_t2m,c6_skw_noise_t2m,c6_kur_noise_t2m] = determine_noise_stats_cmip(key,'t2m')

    #%% Plot figures
    
    #%% Noise E - P
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(c6_std_noise_ep,'$\sigma$ (E-P) [mm/day]',model,0,3.1,cm.cm.deep_r)
    plt.text(3.5e6,8.2e6,str((CMIP_corr_std_ep)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str((CMIP_rmse_std_ep)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_A5_'+str(key_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(c6_skw_noise_ep,'Skewness (E-P) [-]',model,-2.1,2.1,aroace())
    plt.text(3.5e6,8.2e6,str((CMIP_corr_skw_ep)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str((CMIP_rmse_skw_ep)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_A6_'+str(key_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(c6_kur_noise_ep,'Kurtosis (E-P) [-]',model,-5.1,5.1,aroace())
    plt.text(3.5e6,8.2e6,str((CMIP_corr_kur_ep)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str((CMIP_rmse_kur_ep)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_A7_'+str(key_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    #%% Noise T2m
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(c6_std_noise_t2m,'$\sigma$ (T$_{air}$) [$^{\circ}$C]',model,0,1.75,cm.cm.deep_r)
    plt.text(3.5e6,8.2e6,str((CMIP_corr_std_t2m)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str((CMIP_rmse_std_t2m)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_A8_'+str(key_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(c6_skw_noise_t2m,'Skewness (T$_{air}$) [-]',model,-1.15,1.15,aroace())
    plt.text(3.5e6,8.2e6,str((CMIP_corr_skw_t2m)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str((CMIP_rmse_skw_t2m)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_A9_'+str(key_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    fig = plt.figure(figsize=(4,7))
    plot_stat_2d(c6_kur_noise_t2m,'Kurtosis (T$_{air}$) [-]',model,-1.75,1.75,aroace())
    plt.text(3.5e6,8.2e6,str((CMIP_corr_kur_t2m)),color='white',fontsize = 18, weight='bold')
    plt.text(3.5e6,6.8e6,str((CMIP_rmse_kur_t2m)),color='white',fontsize = 18, weight='bold')
    
    plt.savefig(dir_fig+'Figure_A10_'+str(key_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    
