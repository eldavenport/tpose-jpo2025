# Ellen Davenport April 2025
# Show Kelvin Wave activity in TPOSE 2012 and wind at 165E TAO mooring

import xarray as xr
import numpy as np
import warnings
import matplotlib.pyplot as plt
import xgcm
from scipy.signal import detrend, sosfiltfilt, butter
import matplotlib.gridspec as gs
from Isosurface import * 
from numba import njit
warnings.filterwarnings("ignore")
from open_tpose import tpose2012

###---------------------------------------------------------------------TPOSE6 Temperature -----------------------------------------------------------------------------------###
plt.rcParams['font.size'] = 17

data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

prefix = 'diag_state'
ds = tpose2012(prefix)

lats = ds.YC.data
lons = ds.XC.data
depths = ds.Z.data

# set up domain 
latMin = -0.1
latMax = 0.1 
lonMin = 160.0
lonMax = 260.0

zMax = 0.0
zMin = -400.0

lonli = np.argmin(np.abs(lons - lonMin)) 
lonui = np.argmin(np.abs(lons - lonMax)) + 1
latli = np.argmin(np.abs(lats - latMin)) 
latui = np.argmin(np.abs(lats - latMax)) + 1

depthli = np.argmin(np.abs(depths - zMax)) 
depthui = np.argmin(np.abs(depths - zMin)) + 1

lonidx160 = np.argmin(np.abs(lons - 165.0)) 

N = len(ds.time)
ds['time'] = range(0,N,1)

###--------------------------------------------------------------------- Filter, find depth of 20 deg Isotherm ----------------------------------------###
StartDay = 1
EndDay = 350
target_theta = 20.0 # 20 degree isotherm

# filter out higher frequency fluctuations (faster than 45 day period)
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/45)*fs #  (1 cycle /45 days) * (1 day/86400 second)
cutoff = np.array(highF)
order = 4
sos = butter(order, cutoff, 'lowpass', fs=fs, output='sos')

# find depth of 20 deg isotherm using isosurface()
T20deg_depth = isosurface(ds.THETA[:,:,latli:latui,lonli:lonui],target_theta,dim='Z') 
T20deg_depth = np.nan_to_num(T20deg_depth) # Nan's indicate the 20degree isotherm was not found (may have surfaced), set to 0
T20deg_depth = sosfiltfilt(sos, T20deg_depth, axis=0)

T20deg_detrend = detrend(T20deg_depth,axis=0) # detrend the depth
T20deg_anom = T20deg_detrend - np.mean(T20deg_detrend,axis=0) # find the anomaly in the isotherm depth

tmp = ds.THETA[:,0,latli:latui,lonli:lonui].copy(deep=True)
tmp.values = T20deg_anom
tmp.name = 'T20deg_anomaly'
T20deg_anomaly = tmp 

# ----------------------------------------------------------- Western Pacific Wind from TAO -----------------------------------------------------------------------------------------

TAO_file = '/data/SO3/edavenport/TAO_misc/2012_TAO_data/zonalWind_0N165E.cdf' # TAO data on the equator at 146E for 2012
dsTAO = xr.open_dataset(TAO_file)
TAO_Wind_165E = np.nan_to_num(dsTAO.uwnd[:,0,0,0].values)

# -----------------------------------------------------------Plotting -----------------------------------------------------------------------------------------
StartDay = 15
EndDay = 160
y_axis = np.array([30,60,90,120,150])
y_labels = ['02/12','03/12','04/12','05/12','06/12']

wwb_start = 78
wwb_end = 105

print('plotting')
fig = plt.figure(figsize=(10,8))
grid = gs.GridSpec(6, 8)

ax0 = plt.subplot(grid[:,0:2])
ax0.plot(TAO_Wind_165E[StartDay:EndDay],range(StartDay,EndDay),color='#526e75',linewidth=2.0)
ax0.plot(TAO_Wind_165E[wwb_start:wwb_end],range(wwb_start,wwb_end),color='m',linewidth=2.0)
ax0.axvline(0,color='k',linewidth=0.75)
ax0.set_ylim(StartDay,EndDay)
ax0.set_yticks(y_axis)
ax0.set_yticklabels(y_labels)
ax0.set_xlabel('(m/s)')
ax0.set_ylabel('')
ax0.set_title('U Wind 165E')

ax1 = plt.subplot(grid[:,2:])
(T20deg_anomaly[StartDay:EndDay,:,:].mean(dim='YC')).plot(ax=ax1,cmap='RdBu_r',cbar_kwargs={'label':'Depth Anomaly (m)'})
ax1.set_xlabel('Longitude (Deg East)')
ax1.set_ylabel('')
ax1.set_yticks(y_axis)
ax1.set_yticklabels(y_labels)
ax1.set_title('20deg Isotherm Depth Anomaly')

plt.tight_layout()
image_str = 'TPOSE6_2012_KelvinWave_Hovmoller_withWind.png'
plt.savefig(image_str,format='png')
plt.close()