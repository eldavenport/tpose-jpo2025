# Ellen Davenport April 2025
# This script estimates the power spectra of TPOSE and TAO meridional and zonal velocity anomalies from 2012 and 2013

import xarray as xr
from open_tpose import tpose2012to2016
import numpy as np
import warnings
import matplotlib.pyplot as plt
import cmocean.cm as cmo
from scipy.signal import detrend, butter, sosfiltfilt

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 16

zMin = -250
zMax = 0

prefix = ['diag_state']

ds = tpose2012to2016(prefix)

N = len(ds.time)
ds['time'] = range(0,N,1)
ds['XC'] = ds.XC.astype(float)
ds['YC'] = ds.YC.astype(float)
ds['Z'] = ds.Z.astype(float)
ds['XG'] = ds.XG.astype(float)
ds['YG'] = ds.YG.astype(float)

# --------------------------------------------------------------- 140W TAO Velocity ---------------------------------------------------------------------------------
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

Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
U6_140 = ds.UVEL.interp(XG=[220.0],YC=[U_TAO_140.lat],Z=U_TAO_140.depth,time=U_TAO_140.time,method='linear')

temp = U6_140.values
U6_140 = U_TAO_140.copy(deep=True)
U6_140.values = temp[:,:,0,0]
U6_140 = U6_140 + U_TAO_140 - U_TAO_140
U_140_diff = U6_140 - U_TAO_140

meanU_TAO_140 = np.nanmean(U_TAO_140,axis=0)
stdU_TAO_140 = np.nanstd(U_TAO_140,axis=0)

meanU_TP6_140 = np.nanmean(U6_140,axis=0)
stdU_TP6_140 = np.nanstd(U6_140,axis=0)

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
V_140_diff = V6_140 - V_TAO_140

meanV_TAO_140 = np.nanmean(V_TAO_140,axis=0)
stdV_TAO_140 = np.nanstd(V_TAO_140,axis=0)

meanV_TP6_140 = np.nanmean(V6_140,axis=0)
stdV_TP6_140 = np.nanstd(V6_140,axis=0)

zMax = -35
zMin = -200
Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1
N = 750

# crop the time series to the depths we are interested in and the first two years - interested in 2012-2013 
U6_140 = U6_140[:N,Udepthli:Udepthui]
U_TAO_140 = U_TAO_140[:N,Udepthli:Udepthui]

V6_140 = V6_140[:N,Udepthli:Udepthui]
V_TAO_140 = V_TAO_140[:N,Udepthli:Udepthui]

# --------------------------------------------------------------- Detrend and Get Anomalies ---------------------------------------------------------------------------------

# get Vprime
V_detrend = detrend(V_TAO_140,axis=0,type='linear') 
V_anom = V_detrend - np.nanmean(V_detrend,axis=0)
Vprime_filt = V_anom

# get Uprime 
U_detrend = detrend(U_TAO_140,axis=0,type='linear') 
U_anom = U_detrend - np.nanmean(U_detrend,axis=0)
Uprime_filt = U_anom

# get Vprime TPOSE 
V_TP_detrend = detrend(V6_140,axis=0,type='linear') 
V_TP_anom = V_TP_detrend - np.nanmean(V_TP_detrend,axis=0)
Vprime_TP_filt = V_TP_anom

# get Uprime TPOSE
U_TP_detrend = detrend(U6_140,axis=0,type='linear') 
U_TP_anom = U_TP_detrend - np.nanmean(U_TP_detrend,axis=0)
Uprime_TP_filt = U_TP_anom

# --------------------------------------------------------------- Window and FFT ---------------------------------------------------------------------------------
window = np.hanning(N)*np.ones([len(U_TAO_140.depth),1])
Vp_windowed = Vprime_filt*window.T

Fsst = np.fft.fft(Vp_windowed,axis=0)
amp1 = abs(Fsst[:N//2+1,:]**2)
amp1[1:-2,:] = 2*amp1[1:-2,:]
amp1[1:] = amp1[1:]/N**2 # estimate power spectrum from FFT
freq_segments = 1/(np.arange(N//2 + 1)/N) # days per cycle

Up_windowed = Uprime_filt*window.T
Fsst = np.fft.fft(Up_windowed,axis=0)
amp2 = abs(Fsst[:N//2+1,:]**2)
amp2[1:-2,:] = 2*amp2[1:-2,:]
amp2[1:] = amp2[1:]/N**2 # estimate power spectrum from FFT

Vp_TP_windowed = Vprime_TP_filt*window.T
Fsst = np.fft.fft(Vp_TP_windowed,axis=0)
amp3 = abs(Fsst[:N//2+1,:]**2)
amp3[1:-2,:] = 2*amp3[1:-2,:]
amp3[1:] = amp3[1:]/N**2 # estimate power spectrum from FFT

Up_TP_windowed = Uprime_TP_filt*window.T
Fsst = np.fft.fft(Up_TP_windowed,axis=0)
amp4 = abs(Fsst[:N//2+1,:]**2)
amp4[1:-2,:] = 2*amp4[1:-2,:]
amp4[1:] = amp4[1:]/N**2 # estimate power spectrum from FFT

label_str = r'$m^2/s^2$ * days/cycle'

# --------------------------------------------------------------- Logistical organization of data ----------------------------------------------------------
# For the sake of plotting put these variables into data arrays by copying an old data array to get the correct structure
temp = V_140_diff[:len(freq_segments)-1,Udepthli:Udepthui].copy(deep=True)
temp.values = amp1[1:,:]
amp1 = temp
amp1['time'] = freq_segments[1:]

temp = V_140_diff[:len(freq_segments)-1,Udepthli:Udepthui].copy(deep=True)
temp.values = amp2[1:,:]
amp2 = temp
amp2['time'] = freq_segments[1:]

temp = V_140_diff[:len(freq_segments)-1,Udepthli:Udepthui].copy(deep=True)
temp.values = amp3[1:,:]
amp3 = temp
amp3['time'] = freq_segments[1:]

temp = V_140_diff[:len(freq_segments)-1,Udepthli:Udepthui].copy(deep=True)
temp.values = amp4[1:,:]
amp4 = temp
amp4['time'] = freq_segments[1:]

# ---------------------------------------------------------------- Plot ---------------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,8))
amp1.plot(x='time',y='depth',ax=ax[0,0],cmap=cmo.haline,vmin=0,vmax=0.8e-3,cbar_kwargs={'label':''})
ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('Depth (m)')
ax[0,0].set_xlim(5,80)
ax[0,0].set_title('V\' TAO')

amp2.plot(x='time',y='depth',ax=ax[0,1],cmap=cmo.haline,vmin=0.0,vmax=1.5e-3,cbar_kwargs={'label':label_str})
ax[0,1].set_xlabel('')
ax[0,1].set_ylabel('')
ax[0,1].set_xlim(5,80)
ax[0,1].set_title('U\' TAO')

amp3.plot(x='time',y='depth',ax=ax[1,0],cmap=cmo.haline,vmin=0,vmax=0.8e-3,cbar_kwargs={'label':''})
ax[1,0].set_xlabel('Days Per Cycle')
ax[1,0].set_ylabel('Depth (m)')
ax[1,0].set_xlim(5,80)
ax[1,0].set_title('V\' TPOSE')

amp4.plot(x='time',y='depth',ax=ax[1,1],cmap=cmo.haline,vmin=0.0,vmax=1.5e-3,cbar_kwargs={'label':label_str})
ax[1,1].set_xlabel('Days Per Cycle')
ax[1,1].set_ylabel('')
ax[1,1].set_xlim(5,80)
ax[1,1].set_title('U\' TPOSE')

plt.tight_layout()
plt.savefig('V_U_power_spectrogram_depthFreq.png',format='png')