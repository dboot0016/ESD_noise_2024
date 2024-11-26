# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for plotting Figure 2, A1, A2 and A3 for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import xarray as xr 
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import cmocean as cm
from sklearn.metrics import pairwise_distances
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Functions used in script
def aroace():
    # Colormap used for plotting
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ['#1F3554',"#5FAAD7","#FFFFFF","#E7C601","#E38D00"])

def plot_stat_2d_cluster(data,stat,model,vmin,vmax,cmap):
    # Function to plot the statistics
    # Input is:
        # the data that needs to be plotted (data)
        # which statistic is plotted (stat)
        # what and from what source is plotted (model)
        # minimum and maximum of color scaling (vmin, vmax)
        # and colormap (cmap)
    
    fig = plt.figure(figsize=(4, 7))
    # Select extent axes and projection
    ax = fig.add_subplot(1,1,1, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-30, central_latitude=20))
    ax.set_extent((-6e6, 3.5e6, -8.5e6, 1e7), crs=ccrs.LambertAzimuthalEqualArea())
    
    # Plot data
    im=plt.pcolormesh(lon,lat,data,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,cmap=cmap)
    
    # Set title
    ax.set_title(model,fontsize=FS)
        
    # Colorbar specifics
    #cbar=plt.colorbar(im,orientation='horizontal', pad=0.05) 
    #cbar.ax.set_xlabel(stat, fontsize=16)
    #cbar.ax.tick_params(labelsize=16)
    #cbar.ax.xaxis.offsetText.set_fontsize(16)
    
    # Set specifics of the plotting background
    ax.add_feature(cfeature.LAND, zorder=2, edgecolor='black', facecolor='grey', linewidth=.5)
    ax.add_feature(cfeature.OCEAN)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False)
    gl.ylocator = matplotlib.ticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

def calculate_wcss(data, kmeans):
    clusters = kmeans.predict(data)
    wcss = 0
    for cluster in np.unique(clusters):
        cluster_points = data[clusters == cluster]
        wcss += np.sum(np.min(pairwise_distances(cluster_points, [kmeans.cluster_centers_[cluster]]), axis=1))
    return wcss

# Function to calculate the Gap Statistic
def gap_statistic(data, k_range, B=10):
    """
    data: Dataset for clustering.
    k_range: Range of possible cluster counts (e.g., range(1, 10)).
    B: Number of reference data sets to use (default is 10).
    """
    gaps = []
    wcss_data = []

    # Loop over each k value
    for k in k_range:
        # Fit KMeans for actual data
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        wcss_k = calculate_wcss(data, kmeans)
        wcss_data.append(wcss_k)

        # Reference datasets
        wcss_ref = np.zeros(B)
        for i in range(B):
            random_data = np.random.uniform(low=np.min(data, axis=0), high=np.max(data, axis=0), size=data.shape)
            kmeans_ref = KMeans(n_clusters=k)
            kmeans_ref.fit(random_data)
            wcss_ref[i] = calculate_wcss(random_data, kmeans_ref)

        # Gap Statistic calculation
        log_wcss_ref = np.log(wcss_ref)
        log_wcss_k = np.log(wcss_k)
        gap = np.mean(log_wcss_ref) - log_wcss_k
        gaps.append(gap)

    return gaps, wcss_data

#%%
def determine_weighted_average(data,mask):
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()
    
    # Multiply the data by weights, considering masked data (ignoring NaNs automatically)
    weighted_data = data * weights * mask
    
    # Compute the weighted mean, ignoring NaN values
    weighted_mean = weighted_data.mean(dim=['latitude', 'longitude'], skipna=True)
    
    return weighted_mean

#%% Directories + plotting variables
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'
FS = 20
quality = 300

target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-110,20.1,0.25), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(90,-90.1,-0.25), {"units": "degrees_north"}),})

lat = target_grid.latitude
lon = target_grid.longitude

#%% Load in ERA5 noise
data_ep = xr.open_dataset(dir_era+'noise_ep_era5.nc')
noise = data_ep['__xarray_dataarray_variable__'].compute()

# Determine the statistics of the noise
std_noise = noise.std(dim='time')
skw_noise = noise.reduce(func=scipy.stats.skew,dim='time')
kur_noise = noise.reduce(func=scipy.stats.kurtosis,dim='time')

data = xr.concat([std_noise,skw_noise,kur_noise],dim='moments')
data = data.transpose('latitude','longitude','moments')

#%% Reshape date before k means clustering
reshaped_data = np.array(data).reshape(-1, 3)

scaler_ep = StandardScaler()
scaled_data_ep = scaler_ep.fit_transform(reshaped_data)

nan_mask = np.isnan(scaled_data_ep[:,0])
grid_no_nans_ep = scaled_data_ep[~nan_mask]
grid_flat_with_nans_ep = np.empty_like(scaled_data_ep[:,0])  # Create an empty array of the original shape

#%% Determine Silhoutte and Elbow method scores
# Empty list to store silhouette scores
silhouette_scores_ep = []
inertia_ep = []

k_range = range(2, 20)

# Loop through the different values for k
for n_clusters in k_range:
    print(n_clusters)
    # Initialize KMeans with n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(grid_no_nans_ep)
    
    # Calculate the silhouette score for the current number of clusters
    silhouette_avg_ep = silhouette_score(grid_no_nans_ep, cluster_labels)
    silhouette_scores_ep.append(silhouette_avg_ep)
    
    inertia_ep.append(kmeans.inertia_)
    
#%% Plot silhouette and elbow method scores
plt.figure(figsize=(7, 5))
plt.plot(k_range, silhouette_scores_ep, marker='o', linewidth = 2.)
plt.xlabel('Number of Clusters (k)',fontsize=FS-2)
plt.ylabel('Silhouette Score', fontsize=FS-2)
plt.title('E - P: silhouette score',fontsize=FS)
plt.grid()
plt.xticks([2,4,6,8,10,12,14,16,18],fontsize=FS-4)
plt.yticks(fontsize=FS-4)

plt.savefig(dir_fig+'Figure_A1b.png', format='png', dpi=quality,bbox_inches='tight')

# Plotting the elbow method
plt.figure(figsize=(7, 5))
plt.plot(k_range, np.array(inertia_ep)/1e5, marker='o',color='tab:pink',linewidth=2.)
plt.title('E - P: elbow method',fontsize=FS)
plt.xlabel('Number of clusters (k)',fontsize=FS-2)
plt.ylabel('Inertia [10$^{5}$]',fontsize=FS-2)
plt.xticks([2,4,6,8,10,12,14,16,18],fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()
plt.savefig(dir_fig+'Figure_A1a.png', format='png', dpi=quality,bbox_inches='tight')

#%% Calculate the Gap Statistic
gaps, wcss = gap_statistic(grid_no_nans_ep, k_range)

#%% Plot the Gap Statistic
plt.figure(figsize=(7, 5))
plt.plot(k_range, gaps, marker='o',color='tab:purple',linewidth=2.)
plt.xlabel('Number of Clusters (k)',fontsize=FS-2)
plt.ylabel('Gap Statistic',fontsize=FS-2)
plt.title('E - P: gap statistic',fontsize=FS)
plt.xticks([2,4,6,8,10,12,14,16,18],fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()

plt.savefig(dir_fig+'Figure_A1c.png', format='png', dpi=quality,bbox_inches='tight')

#%% Perform k means clustering with 12 clusters
kmeans = KMeans(n_clusters=12)  # Choose the number of clusters
clusters_ep = kmeans.fit_predict(grid_no_nans_ep)

grid_flat_with_nans_ep[nan_mask] = np.nan  # Restore NaN positions
grid_flat_with_nans_ep[~nan_mask] = clusters_ep  # Insert analyzed data back

# Step 6: Reshape to the original 2D grid
clustered_grid_ep = grid_flat_with_nans_ep.reshape(721, 521)

#%% Plot standard deviation, skewness, and kurtosis per cluster on map
weighted_std_ep = np.zeros((int(np.nanmax(clustered_grid_ep)+1),))
weighted_skw_ep = np.zeros((int(np.nanmax(clustered_grid_ep)+1),))
weighted_kur_ep = np.zeros((int(np.nanmax(clustered_grid_ep)+1),))

for cluster_i in range(int(np.nanmax(clustered_grid_ep)+1)):
    print(cluster_i)
    
    indices = np.where(clustered_grid_ep == cluster_i)
    
    mask = np.zeros((721,521))
    mask[indices] = 1
    mask[mask == 0] = np.nan
    
    weighted_std_ep[cluster_i] = determine_weighted_average(std_noise*1e3, mask)
    weighted_skw_ep[cluster_i] = determine_weighted_average(skw_noise, mask)
    weighted_kur_ep[cluster_i] = determine_weighted_average(kur_noise, mask)
    
    plot_stat_2d_cluster(std_noise*mask*1e3,'$\sigma$ [mm/day]','$\sigma$ cluster ' + str(cluster_i+1),0,3.1,cm.cm.deep_r)
    plt.text(3.3e6,8.2e6,str((np.round(weighted_std_ep[cluster_i],2))),color='white',fontsize = 18, weight='bold')
    plt.savefig(dir_fig+'Figure_A2_std_'+str(cluster_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    plot_stat_2d_cluster(skw_noise*mask,'Skewness [-]','S cluster ' + str(cluster_i+1),-2.1,2.1,aroace())
    plt.text(3.3e6,8.2e6,str((np.round(weighted_skw_ep[cluster_i],2))),color='white',fontsize = 18, weight='bold')
    plt.savefig(dir_fig+'Figure_A2_skw_'+str(cluster_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    plot_stat_2d_cluster(kur_noise*mask,'Kurtosis [-]','K cluster ' + str(cluster_i+1),-5.1,5.1,aroace())
    plt.text(3.3e6,8.2e6,str((np.round(weighted_kur_ep[cluster_i],2))),color='white',fontsize = 18, weight='bold')
    plt.savefig(dir_fig+'Figure_A2_kur_'+str(cluster_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')

#%% Plot overview clusters
levels = np.arange(1, 13)
cmap = plt.get_cmap('tab20', 12)  # 10 discrete colors from 'Paired'
norm = mcolors.BoundaryNorm(boundaries=np.arange(0.5, 13.5,1), ncolors=cmap.N, clip=True)

plot_stat_2d_cluster(clustered_grid_ep+1,'Clusters','E - P',0.5,12.5,cmap)
cbar=plt.colorbar(orientation='horizontal', pad=0.05, norm = norm,boundaries=np.arange(0.5,13.5,1), ticks=np.arange(1, 13)) 
cbar.ax.set_xlabel('Clusters', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
cbar.set_ticks(np.arange(1, 13,2))

plt.savefig(dir_fig+'Figure_2a.png', format='png', dpi=quality,bbox_inches='tight')

#%%
num_colors = cmap.N

colors = [cmap(i) for i in range(num_colors)]
hex_colors = [mcolors.rgb2hex(color) for color in colors]

xmin = -5.5
xmax = 1
x = np.arange(xmin,xmax+0.01,0.01)
y = 3/2*x**2

xmnfull = -14
xmax = 1
xf = np.arange(xmnfull,xmax+0.01,0.01)
yf = 3/2*xf**2

fig, ax = plt.subplots(figsize=(7, 5)) 
plt.plot(x,y,c='k',linewidth=1,label='_nolegend_')

for cluster_i in range(int(np.nanmax(clustered_grid_ep)+1)):
    plt.scatter(weighted_skw_ep[cluster_i], weighted_kur_ep[cluster_i],s = weighted_std_ep[cluster_i]*100 , color = hex_colors[cluster_i],zorder=5,label='_nolegend_')
   
plt.xlabel('Skewness [-]',fontsize=FS-2)
plt.ylabel('Kurtosis [-]',fontsize=FS-2)
plt.title('E - P',fontsize=FS)
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()
plt.xlim([xmin,xmax])
plt.ylim([-5,60])

plt.scatter(-20,-2,s = 1*100,c = 'k',label='$\sigma$ = 1.00 mm/day')
plt.scatter(-20,-2,s = 2*100,c = 'k',label='$\sigma$ = 2.00 mm/day')
plt.scatter(-20,-2,s = 3*100,c = 'k',label='$\sigma$ = 3.00 mm/day')
plt.scatter(-20,-2,s = 4*100,c = 'k',label='$\sigma$ = 4.00 mm/day')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=FS-4)


inset_ax = inset_axes(ax, width="30%", height="30%", bbox_to_anchor=(0.57, 0.55, 1.2, 1.2),
                      bbox_transform=ax.transAxes, loc='lower left')

# Create the inset plot
inset_ax.plot(xf,yf,color='k',label='_nolegend_')
inset_ax.scatter(weighted_skw_ep[4],weighted_kur_ep[4], s = 1.9051296516018656*100,color = hex_colors[4],zorder=5,label='_nolegend_')
inset_ax.set_xlim([-12,-10])
inset_ax.set_ylim([150,200])
plt.xticks([-12,-11,-10],fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()


plt.savefig(dir_fig+'Figure_2b.png', format='png', dpi=quality,bbox_inches='tight')


#%% Load in ERA5 noise
data_t2m = xr.open_dataset(dir_era+'noise_t2m_era5.nc')
noise = data_t2m['__xarray_dataarray_variable__'].compute()

# Determine the statistics of the noise
std_noise = noise.std(dim='time')
skw_noise = noise.reduce(func=scipy.stats.skew,dim='time')
kur_noise = noise.reduce(func=scipy.stats.kurtosis,dim='time')

data = xr.concat([std_noise,skw_noise,kur_noise],dim='moments')
data = data.transpose('latitude','longitude','moments')

#%% Manipulate data before k means clustering
reshaped_data = np.array(data).reshape(-1, 3)

scaler_t2m = StandardScaler()
scaled_data_t2m = scaler_t2m.fit_transform(reshaped_data)

nan_mask = np.isnan(scaled_data_t2m[:,0])
grid_no_nans_t2m = scaled_data_t2m[~nan_mask]
grid_flat_with_nans_t2m = np.empty_like(scaled_data_t2m[:,0])  # Create an empty array of the original shape

#%% Determine silhouette and elbow method scores
# Empty list to store silhouette scores
silhouette_scores_t2m = []
inertia_t2m = []

k_range = range(2, 20)

# Loop through the different values for k
for n_clusters in k_range:
    print(n_clusters)
    # Initialize KMeans with n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(grid_no_nans_t2m)
    
    # Calculate the silhouette score for the current number of clusters
    silhouette_avg_t2m = silhouette_score(grid_no_nans_t2m, cluster_labels)
    silhouette_scores_t2m.append(silhouette_avg_t2m)
    
    inertia_t2m.append(kmeans.inertia_)

#%%
plt.figure(figsize=(7, 5))
plt.plot(k_range, silhouette_scores_t2m, marker='o', linewidth = 2.)
plt.xlabel('Number of Clusters (k)',fontsize=FS-2)
plt.ylabel('Silhouette Score', fontsize=FS-2)
plt.title('T$_{2m}$: silhouette score',fontsize=FS)
plt.grid()
plt.xticks([2,4,6,8,10,12,14,16,18],fontsize=FS-4)
plt.yticks(fontsize=FS-4)

plt.savefig(dir_fig+'Figure_A1e.png', format='png', dpi=quality,bbox_inches='tight')

# Plotting the elbow method
plt.figure(figsize=(7, 5))
plt.plot(k_range, np.array(inertia_t2m)/1e5, marker='o',color='tab:pink',linewidth=2.)
plt.title('T$_{2m}$: elbow method',fontsize=FS)
plt.xlabel('Number of clusters (k)',fontsize=FS-2)
plt.ylabel('Inertia [10$^{5}$]',fontsize=FS-2)
plt.xticks([2,4,6,8,10,12,14,16,18],fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()

plt.savefig(dir_fig+'Figure_A1f.png', format='png', dpi=quality,bbox_inches='tight')

#%% Calculate the Gap Statistic
gaps, wcss = gap_statistic(grid_no_nans_t2m, k_range)

#%% Plot the Gap Statistic
plt.figure(figsize=(7, 5))
plt.plot(k_range, gaps, marker='o',color='tab:purple',linewidth=2.)
plt.xlabel('Number of Clusters (k)',fontsize=FS-2)
plt.ylabel('Gap Statistic',fontsize=FS-2)
plt.title('T$_{2m}$: gap statistic',fontsize=FS)
plt.xticks([2,4,6,8,10,12,14,16,18],fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()

plt.savefig(dir_fig+'Figure_A1d.png', format='png', dpi=quality,bbox_inches='tight')

#%% k means clustering with 12 clusters
kmeans = KMeans(n_clusters=12)  # Choose the number of clusters
clusters_t2m = kmeans.fit_predict(grid_no_nans_t2m)

grid_flat_with_nans_t2m[nan_mask] = np.nan  # Restore NaN positions
grid_flat_with_nans_t2m[~nan_mask] = clusters_t2m  # Insert analyzed data back

# Reshape to the original 2D grid
clustered_grid_t2m = grid_flat_with_nans_t2m.reshape(721, 521)

#%% Plot standard deviation, skewness, and kurtosis per cluster on map
weighted_std_t2m = np.zeros((int(np.nanmax(clustered_grid_t2m)+1),))
weighted_skw_t2m = np.zeros((int(np.nanmax(clustered_grid_t2m)+1),))
weighted_kur_t2m = np.zeros((int(np.nanmax(clustered_grid_t2m)+1),))

for cluster_i in range(int(np.nanmax(clustered_grid_t2m)+1)):
    print(cluster_i)
    
    indices = np.where(clustered_grid_t2m == cluster_i)
    
    mask = np.zeros((721,521))
    mask[indices] = 1
    mask[mask == 0] = np.nan
    
    weighted_std_t2m[cluster_i] = determine_weighted_average(std_noise, mask)
    weighted_skw_t2m[cluster_i] = determine_weighted_average(skw_noise, mask)
    weighted_kur_t2m[cluster_i] = determine_weighted_average(kur_noise, mask)
    
    plot_stat_2d_cluster(std_noise*mask,'$\sigma$ [$^{\circ}$C]','$\sigma$ cluster ' + str(cluster_i+1),0,1.75,cm.cm.deep_r)
    plt.text(3.3e6,8.2e6,str((np.round(weighted_std_t2m[cluster_i],2))),color='white',fontsize = 18, weight='bold')
    plt.savefig(dir_fig+'Figure_A3_std_'+str(cluster_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    plot_stat_2d_cluster(skw_noise*mask,'Skewness [-]','S cluster ' + str(cluster_i+1),-1.15,1.15,aroace())
    plt.text(3.3e6,8.2e6,str((np.round(weighted_skw_t2m[cluster_i],2))),color='white',fontsize = 18, weight='bold')
    plt.savefig(dir_fig+'Figure_A3_skw_'+str(cluster_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
    plot_stat_2d_cluster(kur_noise*mask,'Kurtosis [-]','K cluster ' + str(cluster_i+1),-1.75,1.75,aroace())
    plt.text(3.3e6,8.2e6,str((np.round(weighted_kur_t2m[cluster_i],2))),color='white',fontsize = 18, weight='bold')
    plt.savefig(dir_fig+'Figure_A3_kur_'+str(cluster_i+1)+'.png', format='png', dpi=quality,bbox_inches='tight')
    
#%% Plot overview clusters
levels = np.arange(1, 13)
cmap = plt.get_cmap('tab20', 12)  # 10 discrete colors from 'Paired'
norm = mcolors.BoundaryNorm(boundaries=np.arange(0.5, 13.5,1), ncolors=cmap.N, clip=True)

plot_stat_2d_cluster(clustered_grid_t2m+1,'Clusters','T$_{2m}$',0.5,12.5,cmap)
cbar=plt.colorbar(orientation='horizontal', pad=0.05, norm = norm,boundaries=np.arange(0.5,13.5,1), ticks=np.arange(1, 13)) 
cbar.ax.set_xlabel('Clusters', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
cbar.set_ticks(np.arange(1, 13,2))

plt.savefig(dir_fig+'Figure_2c.png', format='png', dpi=quality,bbox_inches='tight')

#%%
num_colors = cmap.N

colors = [cmap(i) for i in range(num_colors)]
hex_colors = [mcolors.rgb2hex(color) for color in colors]

xmin = -0.95
xmax = 0.95
x = np.arange(xmin,xmax+0.01,0.01)
y = 3/2*x**2

plt.figure(figsize=(7, 5))
plt.plot(x,y,c='k',linewidth=1,label ='_nolegend_')

for cluster_i in range(int(np.nanmax(clustered_grid_t2m)+1)):
    plt.scatter(weighted_skw_t2m[cluster_i], weighted_kur_t2m[cluster_i],s = weighted_std_t2m[cluster_i]*200 , color = hex_colors[cluster_i],zorder=5,label ='_nolegend_')
   
plt.xlabel('Skewness [-]',fontsize=FS-2)
plt.ylabel('Kurtosis [-]',fontsize=FS-2)
plt.title('T$_{2m}$',fontsize=FS)
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.grid()
plt.xlim([xmin,xmax])
plt.ylim([-0.2,4.5])

plt.scatter(-2,-2,s = 0.25*200,c = 'k',label='$\sigma$ = 0.25 $^{\circ}$C')
plt.scatter(-2,-2,s = 0.5*200,c = 'k',label='$\sigma$ = 0.50 $^{\circ}$C')
plt.scatter(-2,-2,s = 0.75*200,c = 'k',label='$\sigma$ = 0.75 $^{\circ}$C')
plt.scatter(-2,-2,s = 1*200,c = 'k',label='$\sigma$ = 1.00 $^{\circ}$C')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=FS-4)

plt.savefig(dir_fig+'Figure_2d.png', format='png', dpi=quality,bbox_inches='tight')

#%% Plot legend
MS = ([0.25,0.5,0.75,1.0])
f = lambda m,c, ms: plt.plot([],[],marker=m, color=c,markersize=ms, ls="none")[0]
handles = ([f('o', 'k',MS[model_i]) for model_i in range(4)])


labels = (['$\sigma$ = 0.25 $^{\circ}$C','$\sigma$ = 0.50 $^{\circ}$C','$\sigma$ = 0.75 $^{\circ}$C','$\sigma$ = 1.00 $^{\circ}$C'])
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)

def export_legend(legend, filename="Figure_2_legend.png", expand=[-5,-5,15,5]):
    fig  = legend.figure
    fig.canvas.draw()
    plt.axis('off')
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(dir_fig+filename, dpi=quality, bbox_inches=bbox)

export_legend(legend)
plt.show()


