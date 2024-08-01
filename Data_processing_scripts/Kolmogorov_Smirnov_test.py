# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Kolmogorov-Smirnov test for NIG noise models for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import numpy as np
import xarray as xr
from scipy.stats import chi2, kstest
from scipy.stats import norminvgauss

#%% Load in data temperature noise
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'    # Directory where data is located
data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')                 # T2m noise
noise_t2m = data_t2m['__xarray_dataarray_variable__'].compute()         # Select data
params_t2m = np.load(dir_era+'params_era_t2m_NIG.npy')                  # Load in NIG parameters

# Select NIG parameters from params_t2m
params_ab_t2m = params_t2m[:2,:,:]
params_loc_t2m = params_t2m[2,:,:]
params_scale_t2m = params_t2m[3,:,:]

#%% 
num_bins = 50                   # Number of bins for test
check_t2m = np.zeros((721,521)) # Array to save whether data passes test (lat x lon = 721 x 521)

#%% Test for temperature noise
for lat_i in range(721):                                    # Loop over latitudes
    for lon_i in range(521):                                # Loop over longitudes
        if np.isnan(noise_t2m[0,lat_i,lon_i]):              # If noise is NaN, i.e. a land point, check_t2m is also NaN
            check_t2m[lat_i,lon_i] = np.nan
        else:   
            PARAMS = params_ab_t2m[:,lat_i,lon_i].squeeze()     # If not select parameters of NIG distribution
            LOC = params_loc_t2m[lat_i,lon_i].squeeze()
            SCALE = params_scale_t2m[lat_i,lon_i].squeeze()
            
            data =  noise_t2m[:,lat_i,lon_i]                    # Select noise
            D, p_value = kstest(data, 'norminvgauss', args=(PARAMS[0],PARAMS[1],LOC,SCALE)) # Perform tet

            # Interpret the result
            alpha = 0.05
            if p_value < alpha:
                print("The fit is not good (reject the null hypothesis).")
                check_t2m[lat_i,lon_i] = 0
            else:
                #print("The fit is good (fail to reject the null hypothesis).")
                check_t2m[lat_i,lon_i] = 1

#%% Load in data freshwater noise
data_ep = xr.open_dataset(dir_era+'noise_ep_era5.nc')           # E-P noise
noise_ep = data_ep['__xarray_dataarray_variable__'].compute()   # Select data
params_ep = np.load(dir_era+'params_era_ep_NIG.npy')            # Load in NIG parameters

# Select NIG parameters from params_ep
params_ab_ep = params_ep[:2,:,:]
params_loc_ep = params_ep[2,:,:]
params_scale_ep = params_ep[3,:,:]

#%% 
num_bins = 50                   # Number of bins for test
check_ep = np.zeros((721,521)) # Array to save whether data passes test (lat x lon = 721 x 521)

#%% Test for frehswater noise
for lat_i in range(721):                                    # Loop over latitudes
    for lon_i in range(521):                                # Loop over longitudes
        if np.isnan(noise_ep[0,lat_i,lon_i]):               # If noise is NaN, i.e. a land point, check_t2m is also NaN
            check_ep[lat_i,lon_i] = np.nan
        else:   
            PARAMS = params_ab_ep[:,lat_i,lon_i].squeeze()  # If not, select NIG parameters
            LOC = params_loc_ep[lat_i,lon_i].squeeze()
            SCALE = params_scale_ep[lat_i,lon_i].squeeze()
            
            data =  noise_ep[:,lat_i,lon_i]                 # Select noise
            D, p_value = kstest(data, 'norminvgauss', args=(PARAMS[0],PARAMS[1],LOC,SCALE)) # Perform test

            # Interpret the result
            alpha = 0.05
            if p_value < alpha:
                print("The fit is not good (reject the null hypothesis).")
                check_ep[lat_i,lon_i] = 0
            else:
                #print("The fit is good (fail to reject the null hypothesis).")
                check_ep[lat_i,lon_i] = 1
                

#%% Save arrays
np.save('check_t2m.npy',check_t2m)
np.save('check_ep.npy',check_ep)