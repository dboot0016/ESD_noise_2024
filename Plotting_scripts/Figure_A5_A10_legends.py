# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script for legends Figure A5 to A10 for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Import modules
import pylab as pl
import numpy as np
import cmocean as cm
import matplotlib

#%% Make colorscheme
def aroace():
    # Colormap used for plotting
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ['#1F3554',"#5FAAD7","#FFFFFF","#E7C601","#E38D00"])

#%% Directory where legends should be saved
dir_fig = '/Users/Boot0016/Documents/Project_noise_model/Figures_sub/'

#%% Legends for A5 to A10
a = np.array([[0,3.1]])
pl.figure(figsize=(14, 1.5))
img = pl.imshow(a, cmap=cm.cm.deep_r)
pl.gca().set_visible(False)
cbar = pl.colorbar(orientation="horizontal")
cbar.ax.set_xlabel('$\sigma$ (E - P) [mm/day]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
pl.savefig(dir_fig+"Figure_A5_legend.png",dpi = 300,bbox_inches='tight')

a = np.array([[-2.1,2.1]])
pl.figure(figsize=(14, 1.5))
img = pl.imshow(a, cmap=aroace())
pl.gca().set_visible(False)
cbar = pl.colorbar(orientation="horizontal")
cbar.ax.set_xlabel('Skewness (E - P) [-]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
pl.savefig(dir_fig+"Figure_A6_legend.png",dpi = 300,bbox_inches='tight')

a = np.array([[-5.1,5.1]])
pl.figure(figsize=(14, 1.5))
img = pl.imshow(a, cmap=aroace())
pl.gca().set_visible(False)
cbar = pl.colorbar(orientation="horizontal")
cbar.ax.set_xlabel('Kurtosis (E - P) [-]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
pl.savefig(dir_fig+"Figure_A7_legend.png",dpi = 300,bbox_inches='tight')

a = np.array([[0,1.75]])
pl.figure(figsize=(14, 1.5))
img = pl.imshow(a, cmap=cm.cm.deep_r)
pl.gca().set_visible(False)
cbar = pl.colorbar(orientation="horizontal")
cbar.ax.set_xlabel('$\sigma$ (T$_{2m}$) [$^{\circ}$C]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
pl.savefig(dir_fig+"Figure_A8_legend.png",dpi = 300,bbox_inches='tight')

a = np.array([[-1.15,1.15]])
pl.figure(figsize=(14, 1.5))
img = pl.imshow(a, cmap=aroace())
pl.gca().set_visible(False)
cbar = pl.colorbar(orientation="horizontal")
cbar.ax.set_xlabel('Skewness (T$_{2m}$) [-]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
pl.savefig(dir_fig+"Figure_A9_legend.png",dpi = 300,bbox_inches='tight')

a = np.array([[-1.75,1.75]])
pl.figure(figsize=(14, 1.5))
img = pl.imshow(a, cmap=aroace())
pl.gca().set_visible(False)
cbar = pl.colorbar(orientation="horizontal")
cbar.ax.set_xlabel('Kurtosis (T$_{2m}$) [-]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.offsetText.set_fontsize(16)
pl.savefig(dir_fig+"Figure_A10_legend.png",dpi = 300,bbox_inches='tight')


