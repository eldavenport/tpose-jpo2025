# Ellen Davenport April 2025
# This script compares zonal velocity in TAO and TPOSE at 140W, 170W, and 110W

import xarray as xr
import numpy as np
import warnings
import matplotlib.pyplot as plt
from fastjmd95 import rho
import matplotlib.gridspec as gs
import cmocean.cm as cmo
from matplotlib.colors import TwoSlopeNorm
from open_tpose import tpose2012

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 16

###---------------------------------------------------------------------TPOSE6 diagnostics ---------------------------------------------------------------------###

latMin = -0.1
latMax = 0.1 
zMin = -280
zMax = 0

prefix = ['diag_state']

ds = tpose2012(prefix)
lats = ds.YC.data
lons = ds.XC.data
depths = ds.Z.data

N = len(ds.time)
ds['time'] = range(0,N,1)
ds['sigma'] = (rho(ds.SALT, ds.THETA, 0)-1000)

# This has to be done because of a big-endian/little-endian incompatibility with TAO and TPOSE data when using interp
ds['XC'] = ds.XC.astype(float)
ds['YC'] = ds.YC.astype(float)
ds['Z'] = ds.Z.astype(float)
ds['XG'] = ds.XG.astype(float)
ds['YG'] = ds.YG.astype(float)

###---------------------------------------------------------------------TAO data-----------------------------------------------------------------------------------###
TAO_file = '/home/edavenport/analysis/adcp_xyzt_dy_TAO2012.cdf' # TAO data on the equator at 5 longitudes
dsTAO = xr.open_dataset(TAO_file)
n = len(dsTAO.time)
dsTAO['time'] = range(0,n,1)
dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.U_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonidx140 = 2
lonidx170 = 1
lonidx110 = -1

U_TAO_140 = U_TAO[:,:,0,lonidx140]
U_TAO_110 = U_TAO[:,:,0,lonidx110]
U_TAO_170 = U_TAO[:,:,0,lonidx170]

depthli = np.argmin(np.abs(depths - zMax))
depthui = np.argmin(np.abs(depths - zMin)) + 1

# sample the TAO data locations from the TPOSE data
U6_140 = ds.UVEL.interp(XG=[220.0],YC=[U_TAO_140.lat],Z=U_TAO_140.depth,time=U_TAO_140.time,method='linear')
U6_110 = ds.UVEL.interp(XG=[250.0],YC=[U_TAO_110.lat],Z=U_TAO_110.depth,time=U_TAO_110.time,method='linear')
U6_170 = ds.UVEL.interp(XG=[190.0],YC=[U_TAO_170.lat],Z=U_TAO_170.depth,time=U_TAO_170.time,method='linear')

# get these on the same grid for plotting
temp = U6_170.values
U6_170 = U_TAO_170.copy(deep=True)
U6_170.values = temp[:,:,0,0]
U6_170 = U6_170 + U_TAO_170 - U_TAO_170 # set the that are Nan in TAO to Nan in TPOSE6
U_170_diff = U6_170 - U_TAO_170

temp = U6_140.values
U6_140 = U_TAO_140.copy(deep=True)
U6_140.values = temp[:,:,0,0]
U6_140 = U6_140 + U_TAO_140 - U_TAO_140 # set the that are Nan in TAO to Nan in TPOSE6
U_140_diff = U6_140 - U_TAO_140

temp = U6_110.values
U6_110 = U_TAO_110.copy(deep=True)
U6_110.values = temp[:,:,0,0]
U6_110 = U6_110 + U_TAO_110 - U_TAO_110 # set the that are Nan in TAO to Nan in TPOSE6
U_110_diff = U6_110 - U_TAO_110

###---------------------------------------------------------------------Plot-----------------------------------------------------------------------------------###

vmin = -1.5
vmax = 1.5
x_axis = np.arange(0,N,61)
x_labels = ['','03/12','05/12','07/12','09/12','11/12']
x_label_dummy = ['','','','','','']

levels = np.arange(vmin,vmax,0.05)
print('plotting')
fig = plt.figure(figsize=(22,12))
grid = gs.GridSpec(17, 24)

ax0 = plt.subplot(grid[0:5,0:7])
ax1 = plt.subplot(grid[6:11,0:7])
(U6_170[:,depthli:depthui].T).plot.contourf(ax=ax1,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax1.set_xlabel('')
ax1.set_xticks(x_axis)
ax1.set_xticklabels(x_label_dummy)
ax1.set_ylabel('Z (m)')
ax1.set_title('Zonal Velocity 170W, TPOSE')
(U_TAO_170[:,depthli:depthui].T).plot.contourf(ax=ax0,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax0.set_xlabel('')
ax0.set_xticks(x_axis)
ax0.set_xticklabels(x_label_dummy)
ax0.set_ylabel('Z (m)')
ax0.set_title('Zonal Velocity 170W, TAO')
ax2 = plt.subplot(grid[12:,0:7])
(U_170_diff[:,depthli:depthui].T).plot(ax=ax2,cmap='bwr',levels=levels,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax2.set_xlabel('Time')
ax2.set_xticks(x_axis)
ax2.set_xticklabels(x_labels)
ax2.set_ylabel('Z (m)')
ax2.set_title('Difference 170W, TPOSE-TAO')

ax3 = plt.subplot(grid[0:5,8:15])
ax4 = plt.subplot(grid[6:11,8:15])
(U6_140[:,depthli:depthui].T).plot.contourf(ax=ax4,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax4.set_xlabel('')
ax4.set_xticks(x_axis)
ax4.set_xticklabels(x_label_dummy)
ax4.set_ylabel('')
ax4.set_title('Zonal Velocity 140W, TPOSE')
(U_TAO_140[:,depthli:depthui].T).plot.contourf(ax=ax3,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax3.set_xlabel('')
ax3.set_xticks(x_axis)
ax3.set_xticklabels(x_label_dummy)
ax3.set_ylabel('')
ax3.set_title('Zonal Velocity 140W, TAO')
ax5 = plt.subplot(grid[12:,8:15])
(U_140_diff[:,depthli:depthui].T).plot(ax=ax5,cmap='bwr',levels=levels,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax5.set_xlabel('Time')
ax5.set_xticks(x_axis)
ax5.set_xticklabels(x_labels)
ax5.set_ylabel('')
ax5.set_title('Difference 140W, TPOSE-TAO')

ax6 = plt.subplot(grid[0:5,16:])
ax7 = plt.subplot(grid[6:11,16:])
(U6_110[:,depthli:depthui].T).plot.contourf(ax=ax7,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax7.set_xlabel('')
ax7.set_xticks(x_axis)
ax7.set_xticklabels(x_label_dummy)
ax7.set_ylabel('')
ax7.set_title('Zonal Velocity 110W, TPOSE')
(U_TAO_110[:,depthli:depthui].T).plot.contourf(ax=ax6,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax6.set_xlabel('')
ax6.set_xticks(x_axis)
ax6.set_xticklabels(x_label_dummy)
ax6.set_ylabel('')
ax6.set_title('Zonal Velocity 110W, TAO')
ax8 = plt.subplot(grid[12:,16:])
(U_110_diff[:,depthli:depthui].T).plot.contourf(ax=ax8,levels=levels,cmap='bwr',cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax8.set_xlabel('Time')
ax8.set_xticks(x_axis)
ax8.set_xticklabels(x_labels)
ax8.set_ylabel('')
ax8.set_title('Difference 110W, TPOSE-TAO')

plt.tight_layout()
image_str = 'TAO_TPOSE6_2012_UVEL.png'
plt.savefig(image_str,format='png',bbox_inches='tight')
plt.close()