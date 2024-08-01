# Author: Amber A. Boot (a.a.boot@uu.nl)
# Postdoctoral researcher IMAU, Utrecht University
# Script to download CMIP6 T2m data for 'Observation based temperature and freshwater noise over the Atlantic Ocean'
# Boot, A.A. and Dijkstra, H.A.
# Submitted to 'Earth System Dynamics' on 01-08-2024

#%% Load in modules
import numpy as np
import xarray as xr
import intake
from collections import defaultdict
import xesmf as xe
import gcsfs

#%% List of models
key_list1 = (['CMIP.AS-RCEC.TaiESM1.historical.Amon.gn','CMIP.AWI.AWI-CM-1-1-MR.historical.Amon.gn','CMIP.AWI.AWI-ESM-1-1-LR.historical.Amon.gn','CMIP.BCC.BCC-CSM2-MR.historical.Amon.gn','CMIP.BCC.BCC-ESM1.historical.Amon.gn','CMIP.CAS.FGOALS-g3.historical.Amon.gn','CMIP.CCCma.CanESM5-CanOE.historical.Amon.gn','CMIP.CAS.CAS-ESM2-0.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-HR4.historical.Amon.gn'])
key_list2 = (['CMIP.CCCma.CanESM5.historical.Amon.gn','CMIP.CCCR-IITM.IITM-ESM.historical.Amon.gn','CMIP.CMCC.CMCC-CM2-SR5.historical.Amon.gn','CMIP.CMCC.CMCC-ESM2.historical.Amon.gn','CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Amon.gn','CMIP.CSIRO.ACCESS-ESM1-5.historical.Amon.gn','CMIP.FIO-QLNM.FIO-ESM-2-0.historical.Amon.gn','CMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.historical.Amon.gn','CMIP.MIROC.MIROC-ES2L.historical.Amon.gn'])
key_list3 = (['CMIP.MIROC.MIROC6.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-LL.historical.Amon.gn','CMIP.MOHC.HadGEM3-GC31-MM.historical.Amon.gn','CMIP.MOHC.UKESM1-0-LL.historical.Amon.gn','CMIP.MPI-M.MPI-ESM1-2-LR.historical.Amon.gn','CMIP.MRI.MRI-ESM2-0.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-G-CC.historical.Amon.gn','CMIP.NASA-GISS.GISS-E2-1-H.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM-FV2.historical.Amon.gn'])
key_list4 = (['CMIP.NASA-GISS.GISS-E2-2-H.historical.Amon.gn','CMIP.NCAR.CESM2-FV2.historical.Amon.gn','CMIP.NCAR.CESM2-WACCM.historical.Amon.gn','CMIP.NCAR.CESM2.historical.Amon.gn','CMIP.NCC.NorESM2-MM.historical.Amon.gn','CMIP.NIMS-KMA.UKESM1-0-LL.historical.Amon.gn','CMIP.NUIST.NESM3.historical.Amon.gn','CMIP.SNU.SAM0-UNICON.historical.Amon.gn','CMIP.UA.MCM-UA-1-0.historical.Amon.gn'])

# Total list
key_list = np.concatenate([key_list1,key_list2,key_list3,key_list4])

#%% Masks + directories
# Directories with data
dir_cmip = '/Users/Boot0016/Documents/Project_noise_model/CMIP6_data/'

# CMIP6 land mask
mask_cmip = xr.open_dataset(dir_cmip+'cmip6_atlantic_mask_60S_new.nc')
A_mask_cmip = mask_cmip.mask 

#%%
fs = gcsfs.GCSFileSystem() #list stores, stripp zarr from filename, load 

#%% Intialize target grid, i.e. a rectilinear 1 by 1 degree grid
target_grid = xr.Dataset( #grid to interpolate CMIP6 simulations to
        {   "longitude": (["longitude"], np.arange(-180,180,1), {"units": "degrees_east"}),
            "latitude": (["latitude"], np.arange(-89.5,90,1), {"units": "degrees_north"}),})

#%% Load in data
col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

# Loop over models
for key_i in range(len(key_list)):
    key = key_list[key_i]

    x = [a for a in key.split('.') if a]
    model = ('.'.join(x[2:3]))
    
    print(model)
    
    query = dict(experiment_id=['historical'],
                 table_id='Amon',
                 variable_id=['tas'],
                 grid_label='gn',
                source_id = model)
    

    # subset catalog and get some metrics grouped by 'source_id'
    col_subset = col.search(**query)
    col_subset.df.groupby('source_id')[['experiment_id', 'variable_id', 'table_id']].nunique()
    
    tdict = col_subset.to_dataset_dict(zarr_kwargs={'consolidated': True},
                                       storage_options={'token': 'anon'})
    
    #i = 0
    regridded_tdatasets = defaultdict(dict)
    
    tds = tdict[key]
    tds.attrs["time_concat_key"] = key+'.hist' #add current key information to attributes
    #ds = ds.isel(dcpp_init_year=0,drop=True) #remove this coordinate

    regridder = xe.Regridder(tds,target_grid,'bilinear',ignore_degenerate=True,periodic=True) #create regridder for this dataset
    try:
        regridded_tds = regridder(tds,keep_attrs=True) #apply regridder
    except: #issue with 1 dataset that is chunked along two dimensions, rechunk that
        regridded_tds = regridder(tds.chunk({'time':100,'lat':1000,'lon':1000}),keep_attrs=True)

    regridded_tdatasets[key] = regridded_tds.sel(time = slice('1940-01-01','2015-01-01'))
    
    t2 = regridded_tdatasets[key] 
    
    print('start e-p')
    
    T2m = t2['tas'][0,:,:]*A_mask_cmip
    
    data_t2m = T2m.to_dataset(name = 't2m');
    data_t2m.to_netcdf(str(key)+'_tas_new.nc')
        
        
 
