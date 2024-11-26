# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script to determine and plot distance correlation (Fig. A17) for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import numpy as np
import dcor
import matplotlib.pyplot as plt
import matplotlib

#%% Direcotries where data is stored
dir_era = '/Users/Boot0016/Documents/Project_noise_model/ERA5_data/'

#%% Load in EOFs and PCs ERA5 noise
PCs_ep = np.load(dir_era+'PCs_ep_era_new.npy')
PCs_t2m = np.load(dir_era+'PCs_t2m_era_new.npy')

#%% Determine p-value and distance correlation
dcorr_ep = np.zeros((290,290))
dcorr_t2m = np.zeros((54,54))

p_value_ep = np.zeros((290,290))
p_value_t2m = np.zeros((54,54))

n_permutations=1000

for ep_i in range(290):
    print(ep_i)
    for ep_j in range(290):
        dcorr_ep[ep_i,ep_j] = dcor.distance_correlation(PCs_ep[:,ep_i], PCs_ep[:,ep_j])

        permuted_dcor = np.zeros(n_permutations)
        y = PCs_ep[:,ep_j]

        for perm_i in range(n_permutations):
            np.random.shuffle(y)
            permuted_dcor[perm_i] = dcor.distance_correlation(PCs_ep[:,ep_i],y)
        
        # Calculate p-value
        p_value_ep[ep_i,ep_j] = np.mean(permuted_dcor >= dcorr_ep[ep_i,ep_j])

for t2m_i in range(54):
    print(t2m_i)
    for t2m_j in range(54):
        dcorr_t2m[t2m_i,t2m_j] = dcor.distance_correlation(PCs_t2m[:,t2m_i], PCs_t2m[:,t2m_j])
        
        permuted_dcor = np.zeros(n_permutations)
        y = PCs_t2m[:,t2m_j]

        for perm_i in range(n_permutations):
            np.random.shuffle(y)
            permuted_dcor[perm_i] = dcor.distance_correlation(PCs_t2m[:,t2m_i],y)
        
        # Calculate p-value
        p_value_t2m[t2m_i,t2m_j] = np.mean(permuted_dcor >= dcorr_t2m[t2m_i,t2m_j])

#%%
dcorr_ep[dcorr_ep > 0.9] = np.nan
dcorr_t2m[dcorr_t2m > 0.9] = np.nan

#%% Plotting variables
FS = 20
quality = 300
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Revision/'

def abro():
    # Colormap used for plotting
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ['#4d934d','#75ca91',"#b3e4c7","#FFFFFF","#e695b5","#d9446c",'#d42b53'])

def abro_r():
    # Colormap used for plotting
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ['#d42b53','#d9446c',"#e695b5","#FFFFFF","#b3e4c7","#75ca91",'#4d934d'])

#%% Plot subfigures
fig = plt.figure(figsize=(7,5))
im = plt.pcolormesh(dcorr_ep,vmin=0,vmax=0.12,cmap=abro_r())
plt.xlabel('PC',fontsize=FS-2)
plt.ylabel('PC',fontsize=FS-2)
plt.title('ERA5: E - P', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)

cbar=plt.colorbar(im,orientation='vertical', extend = 'max') 
cbar.ax.set_ylabel('Distance correlation [-]', fontsize=FS-2)
cbar.ax.tick_params(labelsize=FS-4)
cbar.ax.xaxis.offsetText.set_fontsize(FS-4)

plt.savefig(dir_fig+'Figure_A17a.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(5.5,5))
im = plt.pcolormesh(p_value_ep,vmin=0.05,vmax=0.1,cmap='Greys_r')
plt.xlabel('PC',fontsize=FS-2)
plt.ylabel('PC',fontsize=FS-2)
plt.title('ERA5: E - P', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)

plt.savefig(dir_fig+'Figure_A17b.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(7,5))
im = plt.pcolormesh(dcorr_t2m,vmin=0,vmax=0.12,cmap=abro_r())
plt.xlabel('PC',fontsize=FS-2)
plt.ylabel('PC',fontsize=FS-2)
plt.title('ERA5: T$_{2m}$', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)

cbar=plt.colorbar(im,orientation='vertical') 
cbar.ax.set_ylabel('Distance correlation [-]', fontsize=FS-2)
cbar.ax.tick_params(labelsize=FS-4)
cbar.ax.xaxis.offsetText.set_fontsize(FS-4)

plt.savefig(dir_fig+'Figure_A17c.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(5.5,5))
im = plt.pcolormesh(p_value_t2m,vmin=0.05,vmax=0.051,cmap='Greys_r')
plt.xlabel('PC',fontsize=FS-2)
plt.ylabel('PC',fontsize=FS-2)
plt.title('ERA5: T$_{2m}$', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)

plt.savefig(dir_fig+'Figure_A17d.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(7,5))
plt.scatter(PCs_ep[:,28],PCs_ep[:,155],s=20)
plt.xlabel('PC 29',fontsize=FS-2)
plt.ylabel('PC 156',fontsize=FS-2)
plt.title('ERA5: E - P (0.14)', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.xlim([-4.5,4.5])
plt.ylim([-4.5,4.5])

plt.savefig(dir_fig+'Figure_A17e.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(7,5))
plt.scatter(PCs_t2m[:,10],PCs_t2m[:,34],s=20)
plt.xlabel('PC 11',fontsize=FS-2)
plt.ylabel('PC 35',fontsize=FS-2)
plt.title('ERA5: T$_{2m}$ (0.11)', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.xlim([-4.5,4.5])
plt.ylim([-4.5,4.5])

plt.savefig(dir_fig+'Figure_A17g.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(14,5))
plt.plot(np.arange(0,997,1),PCs_t2m[:,10],linewidth = 2.5)
plt.plot(np.arange(0,997,1),PCs_t2m[:,34],linewidth = 2.5)
plt.xlabel('Time [months]',fontsize=FS-2)
plt.ylabel('PC',fontsize=FS-2)
plt.title('ERA5: T$_{2m}$ (0.11)', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.ylim([-4.5,4.5])
plt.xlim([100,300])
plt.legend(['PC 11','PC 35'],fontsize=FS-4)

plt.savefig(dir_fig+'Figure_A17h.png', format='png', dpi=quality,bbox_inches='tight')

#%%
fig = plt.figure(figsize=(14,5))
plt.plot(np.arange(0,996,1),PCs_ep[:,28],linewidth = 2.5)
plt.plot(np.arange(0,996,1),PCs_ep[:,155],linewidth = 2.5)
plt.xlabel('Time [months]',fontsize=FS-2)
plt.ylabel('PC',fontsize=FS-2)
plt.title('ERA5: E - P (0.14)', fontsize=FS)
plt.grid()
plt.xticks(fontsize=FS-4)
plt.yticks(fontsize=FS-4)
plt.ylim([-4.5,4.5])
plt.xlim([100,300])
plt.legend(['PC 29','PC 156'],fontsize=FS-4)

plt.savefig(dir_fig+'Figure_A17f.png', format='png', dpi=quality,bbox_inches='tight')

#%% Save p-values and ditance correlation
np.save('p_t2m.npy',p_value_t2m)
np.save('p_ep.npy',p_value_ep)
np.save('dcorr_t2m.npy',dcorr_t2m)
np.save('dcorr_t2m.npy',dcorr_ep)

#%%


