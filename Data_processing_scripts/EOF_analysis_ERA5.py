# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for Figure 7, 8, S5 and S6 for ' A `realistic' atmospheric noise model in comparison to CMIP6 models'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
from eofs.xarray import Eof

#%% Function to determine number of EOFs necessary to explain exp_var of the variance
def pick_pc_t2m(exp_var):
    return np.argmin(np.abs(var_pc_t2m_sum-exp_var))

def pick_pc_ep(exp_var):
    return np.argmin(np.abs(var_pc_ep_sum-exp_var))

varfrac = 0.9 # Amount of variance captured by the EOFs (i.e. 90%)

#%% Directory where data is stored
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'

#%% Load in ERA5 noise
data_ep = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise_ep = data_ep['__xarray_dataarray_variable__'].compute()

data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise_t2m = data_t2m['__xarray_dataarray_variable__'].compute()

#%% Determine weights for EOF analysis
coslat = np.cos(np.deg2rad(noise_ep['latitude'].values))
weights = np.sqrt(coslat)[..., np.newaxis]

#%% Perform PCA 
# T2m
solver_t2m = Eof(noise_t2m,weights=weights)
eof_t2m = solver_t2m.eofsAsCovariance() # EOFs
pc_t2m = solver_t2m.pcs(pcscaling = 1)  # Correpsonding PCs

# E - P
solver_ep = Eof(noise_ep,weights=weights)
eof_ep = solver_ep.eofsAsCovariance() # EOFs
pc_ep = solver_ep.pcs(pcscaling = 1)  # Correpsonding PCs

#%% Determine variance captured per EOF
# T2m
var_pc_t2m = solver_t2m.varianceFraction()
var_pc_t2m_sum = np.cumsum(var_pc_t2m)

# E - P
var_pc_ep = solver_ep.varianceFraction()
var_pc_ep_sum = np.cumsum(var_pc_ep)

#%% Pick number of EOFs necessary to explain 90% of the variance
nr_eofs_t2m = int(pick_pc_t2m(varfrac)) # Number of EOFs needed to have a var. frac. of 90% (T2m)
nr_eofs_ep = int(pick_pc_ep(varfrac)) # Number of EOFs needed to have a var. frac. of 90% (E - P)

#%% Save EOFs and PCs
# T2m
np.save('EOFs_t2m_era_new.npy',eof_t2m[:nr_eofs_t2m])
np.save('PCs_t2m_era_new.npy',pc_t2m[:,:nr_eofs_t2m])

# E - P
np.save('EOFs_ep_era_new.npy',eof_ep[:nr_eofs_ep])
np.save('PCs_ep_era_new.npy',pc_ep[:,:nr_eofs_ep])
