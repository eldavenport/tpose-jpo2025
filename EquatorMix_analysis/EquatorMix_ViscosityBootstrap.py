# Ellen Davenport April 2025
# This script extracts the viscosity data from Pinkel et al. 2023 and estimates a bootstrap mean and standard error 

import xarray as xr
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.io import loadmat
warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 16

filename = 'Fig4e_extracted_viscosity.mat'
Pinkel_ds = loadmat(filename)

# Load data from the Pinkel et al. 2023 .mat file
visc = Pinkel_ds['vis']
depth = Pinkel_ds['depth'][:-1] # visc is 111 in depth and this axis is 112, depth needs to be reduced by 1
time = Pinkel_ds['time']

# identify time steps with NaN values and drop them prior to bootstrapping
idx = np.isnan(visc[10,:]) 
visc = visc[:,~idx]
time = time[~idx]

# boot strap estimator 
bootmean_by_depth = np.zeros(len(depth))
bootstd_by_depth = np.zeros(len(depth))
for i in range(len(depth)):
    boot_means = []
    for _ in range(10000):
        bootsample = np.random.choice(visc[i,:],size=500, replace=True)
        boot_means.append(np.nanmean(bootsample))
    
    # store the mean and std dev for each depth
    bootmean_by_depth[i] = np.nanmean(boot_means)
    bootstd_by_depth[i] = np.nanstd(boot_means)

# save to a NetCDF file
visc_mean = xr.DataArray(data=bootmean_by_depth,
                       dims=['depth'], 
                       coords=dict(
                        depth=(np.squeeze(depth)*-1),
                    ),
                    attrs=dict(
                        description='Bootstrap mean viscosity from Pinkel et al. 2023',
                        units='m2/s',
                    ),
)
visc_mean.name = 'Mean Viscosity'
visc_std = xr.DataArray(data=bootstd_by_depth,
                       dims=['depth'], 
                       coords=dict(
                        depth=(np.squeeze(depth)*-1),
                    ),
                    attrs=dict(
                        description='Bootstrap std error viscosity from Pinkel et al. 2023',
                        units='m2/s',
                    ),
)
visc_std.name = 'Std Error Viscosity'

data = xr.Dataset({'viscosity':visc_mean,'std_error':visc_std})
data.to_netcdf('Fig4e_extracted_viscosity_bootstrap.nc', mode='w')
