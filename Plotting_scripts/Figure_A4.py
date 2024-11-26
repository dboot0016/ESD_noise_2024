# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script to plot Figure A2 for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import norminvgauss
from scipy.stats import norm

#%% Directories
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'

#%% Load in ERA5 noise
data_t2m = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()

# Determine the statistics of the noise
std_noise = noise.std(dim='time')
skw_noise = noise.reduce(func=scipy.stats.skew,dim='time')
kur_noise = noise.reduce(func=scipy.stats.kurtosis,dim='time')

data = xr.concat([std_noise,skw_noise,kur_noise],dim='moments')
data = data.transpose('latitude','longitude','moments')

#%% Manipulate data
reshaped_data = np.array(data).reshape(-1, 3)

scaler_t2m = StandardScaler()
scaled_data_t2m = scaler_t2m.fit_transform(reshaped_data)

nan_mask = np.isnan(scaled_data_t2m[:,0])
grid_no_nans_t2m = scaled_data_t2m[~nan_mask]
grid_flat_with_nans_t2m = np.empty_like(scaled_data_t2m[:,0])  # Create an empty array of the original shape

#%% Perform k means clustering
kmeans = KMeans(n_clusters=12)  # Choose the number of clusters
clusters_t2m = kmeans.fit_predict(grid_no_nans_t2m)

grid_flat_with_nans_t2m[nan_mask] = np.nan  # Restore NaN positions
grid_flat_with_nans_t2m[~nan_mask] = clusters_t2m  # Insert analyzed data back

# Step 6: Reshape to the original 2D grid
clustered_grid_t2m = grid_flat_with_nans_t2m.reshape(721, 521)

#%% Function used for fitting
def cluster_fit(nr):
    cluster = (clustered_grid_t2m == nr)
    
    noise_cluster = noise*np.where(cluster == 0, np.nan, cluster)
    reshaped = np.array(noise_cluster).reshape(-1, 1)
    
    nan_mask = np.isnan(reshaped)
    cluster_nr_noise = reshaped[~nan_mask]
    
    [hist,bins] = np.histogram(cluster_nr_noise,bins=nbins, density = True)
    
    bin_mid = np.diff(bins)[0]/2+bins[:-1]
    bin_mid2 = np.arange(-10/1e3,10.001/1e3,0.01/1e3)
    
    params_nig = norminvgauss.fit(cluster_nr_noise)
    params_nor = norm.fit(cluster_nr_noise)
    
    pdf_nig = norminvgauss.pdf(bin_mid2,*params_nig)
    pdf_nor = norm.pdf(bin_mid2,*params_nor)

    return hist, pdf_nig, pdf_nor, bin_mid, bin_mid2

#%% Determine histograms
nbins = 50

HIST = np.zeros((12,nbins))
NIG = np.zeros((12,2001))
NOR = np.zeros((12,2001))
BIN = np.zeros((12,nbins))
BIN2 = np.zeros((12,2001))

for cluster_i in range(12):
    print(cluster_i+1)
    [hist1, hist2, hist3, bin_mid, bin_mid2] = cluster_fit(cluster_i)
    HIST[cluster_i,:] = hist1
    NIG[cluster_i,:] = hist2
    NOR[cluster_i,:] = hist3
    BIN[cluster_i,:] = bin_mid
    BIN2[cluster_i,:] = bin_mid2

#%% Plotting variables
LW = 4
FS = 20
quality = 300

W = ([0.2,0.5,0.6,0.15,0.6,0.4,0.5,0.25,0.35,0.4,0.2,0.2])

subf = (['a','b','c','d','e','f','g','h','i','j','k','l'])

#%% Plot
for cluster_i in range(12):
    fig = plt.figure(figsize=(7,5))
    plt.plot(BIN2[cluster_i,:]*1e3,NIG[cluster_i,:],color='tab:red',linewidth = LW)
    plt.plot(BIN2[cluster_i,:]*1e3,NOR[cluster_i,:],color='tab:blue',linewidth = LW)
    plt.bar(BIN[cluster_i,:]*1e3,HIST[cluster_i,:],color='tab:olive',width = W[cluster_i])
    
    plt.xlabel('E - P [mm/day]',fontsize=FS-2)
    plt.ylabel('Density',fontsize=FS-2)
    plt.grid()
    plt.xticks(fontsize=FS-4)
    plt.yticks(fontsize=FS-4)
    plt.title('Cluster ' + str(cluster_i+1),fontsize=FS)
    
    plt.xlim([-7,7])
    
    plt.savefig(dir_fig+'Figure_A2'+subf[cluster_i]+'.png', format='png', dpi=quality,bbox_inches='tight')




