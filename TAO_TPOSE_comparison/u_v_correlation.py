# Ellen Davenport April 2025
# This script looks at the correlation between TPOSE and TAO velocity and temperature at 140W, 170W, and 110W  

import xarray as xr
from open_tpose import tpose2012to2016
import numpy as np
import warnings
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from scipy.signal import detrend
plt.rcParams['font.size'] = 16
import numpy.ma as ma
warnings.filterwarnings("ignore")

prefix = ['diag_state']

ds = tpose2012to2016(prefix)

# This has to be done because of a big-endian/little-endian incompatibility with TAO and TPOSE data when using interp
N = len(ds.time)
ds['time'] = range(0,N,1)
ds['XC'] = ds.XC.astype(float)
ds['YC'] = ds.YC.astype(float)
ds['Z'] = ds.Z.astype(float)
ds['XG'] = ds.XG.astype(float)
ds['YG'] = ds.YG.astype(float)

# --------------------------------------------------------------- 140W TAO Data ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N140W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)

dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.u_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonTAO140 = 0

U_TAO_140 = U_TAO[:,:,latidx,lonTAO140]

# sample these locations from the TPOSE data
U6_140 = ds.UVEL.interp(XG=[220.0],YC=[U_TAO_140.lat],Z=U_TAO_140.depth,time=U_TAO_140.time,method='linear')

temp = U6_140.values
U6_140 = U_TAO_140.copy(deep=True)
U6_140.values = temp[:,:,0,0]
U6_140 = U6_140 + U_TAO_140 - U_TAO_140

V_TAO = dsTAO.v_1206.transpose('time','depth','lat','lon')
V_TAO = V_TAO/100 #convert from cm/s to m/s
V_TAO.data[V_TAO.data > 50] = np.nan # change 9999s to nans

V_TAO_140 = V_TAO[:,:,latidx,lonTAO140]

# sample these locations from the TPOSE data
V6_140 = ds.VVEL.interp(XC=[220.0],YG=[V_TAO_140.lat],Z=V_TAO_140.depth,time=V_TAO_140.time,method='linear')

temp = V6_140.values
V6_140 = V_TAO_140.copy(deep=True)
V6_140.values = temp[:,:,0,0]
V6_140 = V6_140 + V_TAO_140 - V_TAO_140

# --------------------------------------------------------------- Temperature ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N140W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
latidx = 0
lonTAO140 = 0

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
Tdepths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_140 = T_TAO[:,:,latidx,lonTAO140]

# sample these locations from the TPOSE data
T6_140 = ds.THETA.interp(XC=[220.0],YC=[T_TAO_140.lat],Z=T_TAO_140.depth,time=T_TAO_140.time,method='linear')

temp = T6_140.values
T6_140 = T_TAO_140.copy(deep=True)
T6_140.values = temp[:,:,0,0]
T6_140 = T6_140 + T_TAO_140 - T_TAO_140

zMax = -35
zMin = -250
Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1
Tdepthli = np.argmin(np.abs(Tdepths - zMax))
Tdepthui = np.argmin(np.abs(Tdepths - zMin)) + 1

# ----------------------------------------------------------- 170W TAO Data ------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N170W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.u_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonTAO170 = 0

U_TAO_170 = U_TAO[:,:,latidx,lonTAO170]

# sample these locations from the TPOSE data
U6_170 = ds.UVEL.interp(XG=[190.0],YC=[U_TAO_170.lat],Z=U_TAO_170.depth,time=U_TAO_170.time,method='linear')

temp = U6_170.values
U6_170 = U_TAO_170.copy(deep=True)
U6_170.values = temp[:,:,0,0]
U6_170 = U6_170 + U_TAO_170 - U_TAO_170

V_TAO = dsTAO.v_1206.transpose('time','depth','lat','lon')
V_TAO = V_TAO/100 #convert from cm/s to m/s
V_TAO.data[V_TAO.data > 50] = np.nan # change 9999s to nans

V_TAO_170 = V_TAO[:,:,latidx,lonTAO170]

# sample these locations from the TPOSE data
V6_170 = ds.VVEL.interp(XC=[220.0],YG=[V_TAO_170.lat],Z=V_TAO_170.depth,time=V_TAO_170.time,method='linear')

temp = V6_170.values
V6_170 = V_TAO_170.copy(deep=True)
V6_170.values = temp[:,:,0,0]
V6_170 = V6_170 + V_TAO_170 - V_TAO_170

print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N170W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
latidx = 0
lonTAO170 = 0

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
Tdepths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_170 = T_TAO[:,:,latidx,lonTAO170]

# sample these locations from the TPOSE data
T6_170 = ds.THETA.interp(XC=[190.0],YC=[T_TAO_170.lat],Z=T_TAO_170.depth,time=T_TAO_170.time,method='linear')

temp = T6_170.values
T6_170 = T_TAO_170.copy(deep=True)
T6_170.values = temp[:,:,0,0]
T6_170 = T6_170 + T_TAO_170 - T_TAO_170

# --------------------------------------------------------------- 110W TAO Data ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N110W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.u_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonTAO110 = 0

U_TAO_110 = U_TAO[:,:,latidx,lonTAO110]

# sample these locations from the TPOSE data
U6_110 = ds.UVEL.interp(XG=[250.0],YC=[U_TAO_110.lat],Z=U_TAO_110.depth,time=U_TAO_110.time,method='linear')

temp = U6_110.values
U6_110 = U_TAO_110.copy(deep=True)
U6_110.values = temp[:,:,0,0]
U6_110 = U6_110 + U_TAO_110 - U_TAO_110

V_TAO = dsTAO.v_1206.transpose('time','depth','lat','lon')
V_TAO = V_TAO/100 #convert from cm/s to m/s
V_TAO.data[V_TAO.data > 50] = np.nan # change 9999s to nans

V_TAO_110 = V_TAO[:,:,latidx,lonTAO110]

# sample these locations from the TPOSE data
V6_110 = ds.VVEL.interp(XC=[220.0],YG=[V_TAO_110.lat],Z=V_TAO_110.depth,time=V_TAO_110.time,method='linear')

temp = V6_110.values
V6_110 = V_TAO_110.copy(deep=True)
V6_110.values = temp[:,:,0,0]
V6_110 = V6_110 + V_TAO_110 - V_TAO_110

print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N110W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
latidx = 0
lonTAO110 = 0

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
Tdepths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_110 = T_TAO[:,:,latidx,lonTAO110]

# sample these locations from the TPOSE data
T6_110 = ds.THETA.interp(XC=[250.0],YC=[T_TAO_110.lat],Z=T_TAO_110.depth,time=T_TAO_110.time,method='linear')

temp = T6_110.values
T6_110 = T_TAO_110.copy(deep=True)
T6_110.values = temp[:,:,0,0]
T6_110 = T6_110 + T_TAO_110 - T_TAO_110

# --------------------------------------------------------------- get anomalies and correlations  ------------------------------------------------------------------------------
# crop the time series to the depths we are interested in 
U6_140 = U6_140[:,Udepthli:Udepthui]
U_TAO_140 = U_TAO_140[:,Udepthli:Udepthui]

V6_140 = V6_140[:,Udepthli:Udepthui]
V_TAO_140 = V_TAO_140[:,Udepthli:Udepthui]
depths = depths[Udepthli:Udepthui]

# Detrend at every depth - this can be handle nans only one line at a time (xarray doesn't allow 2D boolean indexing) -- this is fast because the dataset is relatively small
for z in range(len(depths)):
    signal = V_TAO_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    V_TAO_140[:,z] = signal
    signal = U_TAO_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    U_TAO_140[:,z] = signal
    signal = V6_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    V6_140[:,z] = signal
    signal = U6_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    U6_140[:,z] = signal

for z in range(len(Tdepths)):
    signal = T_TAO_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    T_TAO_140[:,z] = signal

Vprime_anom = V_TAO_140 - np.nanmean(V_TAO_140,axis=0)
Uprime_anom = U_TAO_140 - np.nanmean(U_TAO_140,axis=0)
Vprime_TP_anom = V6_140 - np.nanmean(V6_140,axis=0)
Uprime_TP_anom = U6_140 - np.nanmean(U6_140,axis=0)

# store in data arrays for easier manipulation and plotting
temp = U6_140.copy(deep=True)
temp.values = Uprime_anom
Uprime_anom = temp

temp = V6_140.copy(deep=True)
temp.values = Vprime_anom
Vprime_anom = temp

temp = U6_140.copy(deep=True)
temp.values = Uprime_TP_anom
Uprime_TP_anom = temp

temp = V6_140.copy(deep=True)
temp.values = Vprime_TP_anom
Vprime_TP_anom = temp

# take correlations at all three locations between TAO and TPOSE 
# 140W
U_corr = xr.corr(U6_140, U_TAO_140, dim="time") # correlation in time of the zonal velocity at every depth
V_corr = xr.corr(V6_140, V_TAO_140, dim="time") # correlation in time of the meridional velocity at every depth
T_corr = xr.corr(T6_140, T_TAO_140, dim="time") # correlation in time of the temperature at every depth

# 170W
U_corr_170 = xr.corr(U6_170, U_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_170 = xr.corr(V6_170, V_TAO_170, dim="time") # correlation in time of the meridional velocity at every depth
T_corr_170 = xr.corr(T6_170, T_TAO_170, dim="time") # correlation in time of the temperature at every depth

# 110W
U_corr_110 = xr.corr(U6_110, U_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_110 = xr.corr(V6_110, V_TAO_110, dim="time") # correlation in time of the meridional velocity at every depth
T_corr_110 = xr.corr(T6_110, T_TAO_110, dim="time") # correlation in time of the temperature at every depth

# --------------------------------------------------------------- plotting correlations U, V, T  -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15,7),ncols=3)
ax[0].plot(U_corr_170,U_TAO_170.depth,linewidth=2.0,label='u')
ax[0].plot(V_corr_170,V_TAO_170.depth,linewidth=2.0,label='v')
ax[0].plot(T_corr_170,T_TAO_170.depth,linewidth=2.0,label='T')
ax[0].set_xlabel('Correlation')
ax[0].set_ylabel('Depth (m)')
ax[0].set_title('0N,170W')
ax[0].legend()
ax[0].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[0].set_xlim(-0.5,1)
ax[0].set_ylim(-250,0)

ax[1].plot(U_corr,U_TAO_140.depth,linewidth=2.0,label='u')
ax[1].plot(V_corr,V_TAO_140.depth,linewidth=2.0,label='v')
ax[1].plot(T_corr,T_TAO_140.depth,linewidth=2.0,label='T')
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Depth (m)')
ax[1].set_title('0N,140W')
ax[1].legend()
ax[1].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[1].set_xlim(-0.5,1)
ax[1].set_ylim(-250,0)

ax[2].plot(U_corr_110,U_TAO_110.depth,linewidth=2.0,label='u')
ax[2].plot(V_corr_110,V_TAO_110.depth,linewidth=2.0,label='v')
ax[2].plot(T_corr_110,T_TAO_110.depth,linewidth=2.0,label='T')
ax[2].set_xlabel('Correlation')
ax[2].set_ylabel('Depth (m)')
ax[2].set_title('0N,110W')
ax[2].legend()
ax[2].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[2].set_xlim(-0.5,1)
ax[2].set_ylim(-250,0)

plt.tight_layout()
plt.savefig('V_U_T_correlation_by_depth.png',format='png')

# --------------------------------------------------------------- compute u'v', u'u', v'v'  ---------------------------------------------------------------------------------
# 140W
uv_flux_TP = Uprime_TP_anom*Vprime_TP_anom
uv_flux_TAO = Uprime_anom*Vprime_anom
uu_TP = Uprime_TP_anom*Uprime_TP_anom
uu_TAO = Uprime_anom*Uprime_anom
vv_TP = Vprime_TP_anom*Vprime_TP_anom
vv_TAO = Vprime_anom*Vprime_anom

# correlations of the velocity fluctuation terms at 140W 
Flux_corr = xr.corr(uv_flux_TP, uv_flux_TAO, dim="time") # correlation in time of the zonal velocity at every depth
uu_corr = xr.corr(uu_TP, uu_TAO, dim="time") # correlation in time of the zonal velocity at every depth
vv_corr = xr.corr(vv_TP, vv_TAO, dim="time") # correlation in time of the zonal velocity at every depth

# 170W
uv_flux_TP_170 = U6_170*V6_170
uv_flux_TAO_170 = U_TAO_170*V_TAO_170
uu_TP_170 = U6_170*U6_170
uu_TAO_170 = U_TAO_170*U_TAO_170
vv_TP_170 = V6_170*V6_170
vv_TAO_170 = V_TAO_170*V_TAO_170
Flux_corr_170 = xr.corr(uv_flux_TP_170, uv_flux_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth
uu_corr_170 = xr.corr(uu_TP_170, uu_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth
vv_corr_170 = xr.corr(vv_TP_170, vv_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth

# 110W
uv_flux_TP_110 = U6_110*V6_110
uv_flux_TAO_110 = U_TAO_110*V_TAO_110
uu_TP_110 = U6_110*U6_110
uu_TAO_110 = U_TAO_110*U_TAO_110
vv_TP_110 = V6_110*V6_110
vv_TAO_110 = V_TAO_110*V_TAO_110
Flux_corr_110 = xr.corr(uv_flux_TP_110, uv_flux_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth
uu_corr_110 = xr.corr(uu_TP_110, uu_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth
vv_corr_110 = xr.corr(vv_TP_110, vv_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth

# --------------------------------------------------------------- plotting  ---------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15,7),ncols=3)
ax[0].plot(Flux_corr_170,U_TAO_170.depth,linewidth=2.0,label='u\'v\'',color='tab:blue')
ax[0].plot(uu_corr_170,U_TAO_170.depth,linewidth=2.0,label='${u\'}^2$',color='tab:blue',linestyle='--')
ax[0].plot(vv_corr_170,U_TAO_170.depth,linewidth=2.0,label='${v\'}^2$',color='tab:blue',linestyle=':')
ax[0].set_xlabel('Correlation')
ax[0].set_ylabel('Depth (m)')
ax[0].set_title('0N,170W')
ax[0].legend()
ax[0].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[0].set_ylim(-250,0)
ax[0].set_xlim(-0.5,1)

ax[1].plot(Flux_corr,U_TAO_140.depth,linewidth=2.0,label='u\'v\'',color='tab:blue')
ax[1].plot(uu_corr,U_TAO_140.depth,linewidth=2.0,label='${u\'}^2$',color='tab:blue',linestyle='--')
ax[1].plot(vv_corr,U_TAO_140.depth,linewidth=2.0,label='${v\'}^2$',color='tab:blue',linestyle=':')
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Depth (m)')
ax[1].set_title('0N,140W')
ax[1].legend()
ax[1].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[1].set_ylim(-250,0)
ax[1].set_xlim(-0.5,1)

ax[2].plot(Flux_corr_110,U_TAO_110.depth,linewidth=2.0,label='u\'v\'',color='tab:blue')
ax[2].plot(uu_corr_110,U_TAO_110.depth,linewidth=2.0,label='${u\'}^2$',color='tab:blue',linestyle='--')
ax[2].plot(vv_corr_110,U_TAO_110.depth,linewidth=2.0,label='${v\'}^2$',color='tab:blue',linestyle=':')
ax[2].set_xlabel('Correlation')
ax[2].set_ylabel('Depth (m)')
ax[2].set_title('0N,110W')
ax[2].legend()
ax[2].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[2].set_ylim(-250,0)
ax[2].set_xlim(-0.5,1)

plt.tight_layout()
plt.savefig('V_U_flux_correlation_by_depth.png',format='png')