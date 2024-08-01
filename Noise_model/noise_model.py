# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Freshwater and temperature noise models for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import random
from scipy.stats import norminvgauss

#%% Direcotries where data is stored
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'

#%% Load in ERA5 noise
data_ep = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise_ep = data_ep['__xarray_dataarray_variable__'].compute()

data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise_t2m = data_t2m['__xarray_dataarray_variable__'].compute()

#%% Load in EOFs and PCs ERA5 noise
PCs_ep = np.load(dir_era+'PCs_ep_era_new.npy')
EOF_ep = np.load(dir_era+'EOFs_ep_era_new.npy')
eof_ep = np.swapaxes(EOF_ep,1,2)

PCs_t2m = np.load(dir_era+'PCs_t2m_era_new.npy')
EOF_t2m = np.load(dir_era+'EOFs_t2m_era_new.npy')
eof_t2m = np.swapaxes(EOF_ep,1,2)

#%% Initiliaze arrays for noise models
size_x = 1 # Length of noise realizations in months

pc_series_ep = np.zeros(np.shape(eof_ep)[0]) 
noise_ep = np.zeros([size_x,np.shape(eof_ep)[1],np.shape(eof_ep)[2]]) 

pc_series_t2m = np.zeros(np.shape(eof_t2m)[0]) 
noise_t2m = np.zeros([size_x,np.shape(eof_t2m)[1],np.shape(eof_t2m)[2]]) 

#%% Initialize arrays for NIG parameters
PARAMS_ep = np.zeros((4,721,521))
Params_ep = np.zeros((4,np.shape(eof_ep)[0]))

PARAMS_t2m = np.zeros((4,721,521))
Params_t2m = np.zeros((4,np.shape(eof_t2m)[0]))

#%% Choose model and generate a noise field
models = (['PC (NIG)','NIG','PC (1)','PC (N)'])
model_i = 0 # model_i = 0: PC (NIG); model_i = 1: NIG; model_i = 2: PC (1); model_i = 3: PC (N)
model = models[model_i]
print(model)

# Freshwater noise model
if model == 'PC (1)':
    # Loop over time
    for time_i in range(size_x):
        random_nr = random.randint(0,995) # Random integer between 0 and 995
        # Loop over EOFs
        for eof_j in range(np.shape(eof_ep)[0]):
            noise_ep[time_i,:,:] += PCs_ep[random_nr,eof_j] * eof_ep[eof_j,:,:]
            
elif model == 'NIG':
    params_tot = np.load(dir_era+'params_era_ep_NIG.npy') # Load in parameters
    # Loop over grid points
    for lat_i in range(np.shape(eof_ep)[1]):
        for lon_i in range(np.shape(eof_ep)[2]):
            # If ERA5 noise is NaN (i.e. a land point) then noise model is NaN
            if np.isnan(noise_ep[0,lat_i,lon_i]):
                noise_ep[:,lat_i,lon_i] = np.nan*np.ones((size_x))
            else:
                params = params_tot[:,lat_i,lon_i]
                A = norminvgauss.rvs(*params,size = size_x)
                noise_ep[:,lat_i,lon_i] = A            

elif model == 'PC (NIG)':
    # Loop over EOFs
    for eof_j in range(np.shape(eof_ep)[0]):
        print(eof_j)
        params = norminvgauss.fit(PCs_ep[:,eof_j])
        Params_ep[:,eof_j] = params 
        
        pc_rvs = norminvgauss.rvs(*params,size=size_x)
        # Loop over time
        for time_i in range(size_x):
            noise_ep[time_i,:,:] += pc_rvs[time_i] * eof_ep[eof_j,:,:]
        
    np.save(dir_era+'params_era_ep_PC_NIG.npy',Params_ep)
  
elif model == 'PC (N)':
    # Loop over time
    for time_i in range(size_x):
        # Loop over EOFs
        for eof_j in range(np.shape(eof_ep)[0]):
            random_nr = random.randint(0,995) # Random integer between 0 and 995
            noise_ep[time_i,:,:] += PCs_ep[random_nr,eof_j] * eof_ep[eof_j,:,:]
    
    
# Temperature noise model
if model == 'PC (1)':
    # Loop over time
    for time_i in range(size_x):
        random_nr = random.randint(0,995) # Random integer between 0 and 995
        # Loop over EOFs
        for eof_j in range(np.shape(eof_t2m)[0]):
            noise_t2m[time_i,:,:] += PCs_t2m[random_nr,eof_j] * eof_t2m[eof_j,:,:]
            
elif model == 'NIG':
    params_tot = np.load(dir_era+'params_era_t2m_NIG.npy') # Load in parameters
    # Loop over grid points
    for lat_i in range(np.shape(eof_t2m)[1]):
        for lon_i in range(np.shape(eof_t2m)[2]):
            # If ERA5 noise is NaN (i.e. a land point) then noise model is NaN
            if np.isnan(noise_t2m[0,lat_i,lon_i]):
                noise_t2m[:,lat_i,lon_i] = np.nan*np.ones((size_x))
            else:
                params = params_tot[:,lat_i,lon_i]
                A = norminvgauss.rvs(*params,size = size_x)
                noise_t2m[:,lat_i,lon_i] = A            

elif model == 'PC (NIG)':
    # Loop over EOFs
    for eof_j in range(np.shape(eof_t2m)[0]):
        print(eof_j)
        params = norminvgauss.fit(PCs_t2m[:,eof_j])
        Params_t2m[:,eof_j] = params 
        
        pc_rvs = norminvgauss.rvs(*params,size=size_x)
        # Loop over time
        for time_i in range(size_x):
            noise_t2m[time_i,:,:] += pc_rvs[time_i] * eof_t2m[eof_j,:,:]
        
    np.save(dir_era+'params_era_t2m_PC_NIG.npy',Params_t2m)
  
elif model == 'PC (N)':
    # Loop over time
    for time_i in range(size_x):
        # Loop over EOFs
        for eof_j in range(np.shape(eof_t2m)[0]):
            random_nr = random.randint(0,995) # Random integer between 0 and 995
            noise_t2m[time_i,:,:] += PCs_t2m[random_nr,eof_j] * eof_t2m[eof_j,:,:]





