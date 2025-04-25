# Ellen Davenport 2025
# This script comparses zonal velocity, meridional velocity, and tempearture between TPOSE 2012-2016 and TAO. 
# It includes estimates of average profiles and RMS Difference
# Comparisons are made on the Equator at 140W, 170W, and 110W

import xarray as xr
from open_tpose import tpose2012to2016
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import TwoSlopeNorm
import cmocean.cm as cmo
from scipy.signal import sosfiltfilt, butter

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 16

zMin = -250
zMax = 0

prefix = ['diag_state']

# Open TPOSE 2012 through 2016
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
# --------------------------------------------------------------- Velocity ---------------------------------------------------------------------------------
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N140W_daily.cdf' 
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

# --------------------------------------------------------------- Temperature ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N140W_daily.cdf'
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
depths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_140 = T_TAO[:,:,latidx,lonTAO140]

Tdepthli = np.argmin(np.abs(depths - zMax))
Tdepthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
T6_140 = ds.THETA.interp(XC=[220.0],YC=[T_TAO_140.lat],Z=T_TAO_140.depth,time=T_TAO_140.time,method='linear')

temp = T6_140.values
T6_140 = T_TAO_140.copy(deep=True)
T6_140.values = temp[:,:,0,0]
T6_140 = T6_140 + T_TAO_140 - T_TAO_140
T_140_diff = T6_140 - T_TAO_140

meanT_TAO_140 = np.nanmean(T_TAO_140,axis=0)
stdT_TAO_140 = np.nanstd(T_TAO_140,axis=0)
meanT_TP6_140 = np.nanmean(T6_140,axis=0)
stdT_TP6_140 = np.nanstd(T6_140,axis=0)

# -----------------------------------------------------------Plotting 140W -----------------------------------------------------------------------------------------

vmin = -1.5
vmax = 1.5
levels = np.arange(vmin,vmax,0.1)
x_axis = np.arange(0,1800,365)
x_labels = ['01/2012','01/2013','01/2014','01/2015','01/2016']

fig = plt.figure(figsize=(27,22))
grid = gs.GridSpec(23, 28)

#zonal velocity
ax0 = plt.subplot(grid[0:6,0:9])
(U6_140[:,Udepthli:Udepthui].T).plot.contourf(ax=ax0,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax0.set_xlabel('')
ax0.set_xticks(x_axis)
ax0.set_xticklabels(x_labels)
ax0.set_ylabel('Z (m)')
ax0.set_title('u TPOSE')
ax0.set_ylim(-249,0)
ax1 = plt.subplot(grid[0:6,10:20])
(U_TAO_140[:,Udepthli:Udepthui].T).plot.contourf(ax=ax1,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax1.set_xlabel('')
ax1.set_xticks(x_axis)
ax1.set_xticklabels(x_labels)
ax1.set_ylabel('')
ax1.set_title('u TAO')
ax1.set_ylim(-249,0)

ax2 = plt.subplot(grid[0:6,21:24])
# grid
ax2.axvline(0.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(0.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(1.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(1.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(-0.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
# curves
ax2.plot(meanU_TP6_140[Udepthli:Udepthui],U_TAO_140.depth[Udepthli:Udepthui],linewidth=1.5,color='#f6b26b',label='TPOSE')
ax2.fill_betweenx(U_TAO_140.depth[Udepthli:Udepthui],meanU_TP6_140[Udepthli:Udepthui]-stdU_TP6_140[Udepthli:Udepthui],meanU_TP6_140[Udepthli:Udepthui]+stdU_TP6_140[Udepthli:Udepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax2.plot(meanU_TAO_140[Udepthli:Udepthui],U_TAO_140.depth[Udepthli:Udepthui],linewidth=1.5,color='#526e75',label='TAO')
ax2.fill_betweenx(U_TAO_140.depth[Udepthli:Udepthui],meanU_TAO_140[Udepthli:Udepthui]-stdU_TAO_140[Udepthli:Udepthui],meanU_TAO_140[Udepthli:Udepthui]+stdU_TAO_140[Udepthli:Udepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax2.legend(loc='lower right')
ax2.set_xlabel('Mean (m/s)')
ax2.set_ylim(-249,0)

# meridional velocity
vmin = -1.0
vmax = 1.0
levels = np.arange(vmin,vmax,0.05)
ax3 = plt.subplot(grid[7:13,0:9])
(V6_140[:,Udepthli:Udepthui].T).plot.contourf(ax=ax3,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax3.set_xlabel('')
ax3.set_xticks(x_axis)
ax3.set_xticklabels(x_labels)
ax3.set_ylabel('Z (m)')
ax3.set_title('v TPOSE')
ax3.set_ylim(-249,0)
ax4 = plt.subplot(grid[7:13,10:20])
(V_TAO_140[:,Udepthli:Udepthui].T).plot.contourf(ax=ax4,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax4.set_xticks(x_axis)
ax4.set_xticklabels(x_labels)
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.set_title('v TAO')
ax4.set_ylim(-249,0)

ax5 = plt.subplot(grid[7:13,21:24])
ax5.axvline(0.25,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(0.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(-0.25,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.plot(meanV_TP6_140[Udepthli:Udepthui],V_TAO_140.depth[Udepthli:Udepthui],linewidth=1.5,color='#f6b26b',label='TPOSE')
ax5.fill_betweenx(V_TAO_140.depth[Udepthli:Udepthui],meanV_TP6_140[Udepthli:Udepthui]-stdV_TP6_140[Udepthli:Udepthui],meanV_TP6_140[Udepthli:Udepthui]+stdV_TP6_140[Udepthli:Udepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax5.plot(meanV_TAO_140[Udepthli:Udepthui],V_TAO_140.depth[Udepthli:Udepthui],linewidth=1.5,color='#526e75',label='TAO')
ax5.fill_betweenx(V_TAO_140.depth[Udepthli:Udepthui],meanV_TAO_140[Udepthli:Udepthui]-stdV_TAO_140[Udepthli:Udepthui],meanV_TAO_140[Udepthli:Udepthui]+stdV_TAO_140[Udepthli:Udepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax5.legend(loc='lower right')
ax5.set_xlabel('Mean (m/s)')
ax5.set_ylim(-249,0)
ax5.set_xlim(-0.35,0.35)

# temperature
vmin = 10
vmax = 30
levels = np.arange(vmin,vmax,0.2)
ax6 = plt.subplot(grid[14:20,0:9])
(T6_140[:,Tdepthli:Tdepthui].T).plot.contourf(ax=ax6,levels=levels,cmap=cmo.thermal,add_colorbar=False)
ax6.set_xticks(x_axis)
ax6.set_xticklabels(x_labels)
ax6.set_xlabel('Time')
ax6.set_ylabel('Z (m)')
ax6.set_title('T TPOSE')
ax6.set_ylim(-249,0)
ax7 = plt.subplot(grid[14:20,10:20])
(T_TAO_140[:,Tdepthli:Tdepthui].T).plot.contourf(ax=ax7,levels=levels,cmap=cmo.thermal,cbar_kwargs={'label':'$deg C$'})
ax7.set_xticks(x_axis)
ax7.set_xticklabels(x_labels)
ax7.set_xlabel('Time')
ax7.set_ylabel('')
ax7.set_title('T TAO')
ax7.set_ylim(-249,0)

ax8 = plt.subplot(grid[14:20,21:24])
ax8.axvline(15,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(20,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(25,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.plot(meanT_TP6_140[Tdepthli:Tdepthui],T_TAO_140.depth[Tdepthli:Tdepthui],linewidth=1.5,color='#f6b26b',label='TPOSE')
ax8.fill_betweenx(T_TAO_140.depth[Tdepthli:Tdepthui],meanT_TP6_140[Tdepthli:Tdepthui]-stdT_TP6_140[Tdepthli:Tdepthui],meanT_TP6_140[Tdepthli:Tdepthui]+stdT_TP6_140[Tdepthli:Tdepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax8.plot(meanT_TAO_140[Tdepthli:Tdepthui],T_TAO_140.depth[Tdepthli:Tdepthui],linewidth=1.5,color='#526e75',label='TAO')
ax8.fill_betweenx(T_TAO_140.depth[Tdepthli:Tdepthui],meanT_TAO_140[Tdepthli:Tdepthui]-stdT_TAO_140[Tdepthli:Tdepthui],meanT_TAO_140[Tdepthli:Tdepthui]+stdT_TAO_140[Tdepthli:Tdepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax8.legend(loc='lower right')
ax8.set_xlabel('Mean (deg C)')
ax8.set_ylim(-249,0)

# ----------------------------------------------------------- RMSE Estimation -----------------------------------------------------------------------------------------
# Filter TPOSE and TAO into low (>100 day), mid (20-100 day), and high (<20 day) frequency components
# set nan to -9999
U6_140 = U6_140.fillna(-9999)
U_TAO_140 = U_TAO_140.fillna(-9999)

V6_140 = V6_140.fillna(-9999)
V_TAO_140 = V_TAO_140.fillna(-9999)

T6_140 = T6_140.fillna(-9999)
T_TAO_140 = T_TAO_140.fillna(-9999)

# lowpass filter
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_140 , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_140, axis=0)

TPOSE_V_low = sosfiltfilt(sos, V6_140 , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_140, axis=0)

TPOSE_T_low = sosfiltfilt(sos, T6_140 , axis=0)
TAO_T_low = sosfiltfilt(sos, T_TAO_140, axis=0)

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_140 , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_140, axis=0)

TPOSE_V_sub20 = sosfiltfilt(sos, V6_140 , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_140, axis=0)

TPOSE_T_sub20 = sosfiltfilt(sos, T6_140 , axis=0)
TAO_T_sub20 = sosfiltfilt(sos, T_TAO_140, axis=0)

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_140 , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_140, axis=0)

TPOSE_V_midFreq = sosfiltfilt(sos, V6_140 , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_140, axis=0)

TPOSE_T_midFreq = sosfiltfilt(sos, T6_140 , axis=0)
TAO_T_midFreq = sosfiltfilt(sos, T_TAO_140, axis=0)

N, _ = TPOSE_U_midFreq.shape
# RMS difference
rmse_U_lowFreq = np.sqrt((((TPOSE_U_low -TAO_U_low)**2).mean(axis=0)))
rmse_V_lowFreq = np.sqrt((((TPOSE_V_low -TAO_V_low)**2).mean(axis=0)))
rmse_T_lowFreq = np.sqrt((((TPOSE_T_low -TAO_T_low)**2).mean(axis=0)))

rmse_U_medFreq = np.sqrt((((TPOSE_U_midFreq-TAO_U_midFreq)**2).mean(axis=0)))
rmse_V_medFreq = np.sqrt((((TPOSE_V_midFreq-TAO_V_midFreq)**2).mean(axis=0)))
rmse_T_medFreq = np.sqrt((((TPOSE_T_midFreq-TAO_T_midFreq)**2).mean(axis=0)))

rmse_U_hiFreq = np.sqrt((((TPOSE_U_sub20 -TAO_U_sub20)**2).mean(axis=0)))
rmse_V_hiFreq = np.sqrt((((TPOSE_V_sub20 -TAO_V_sub20)**2).mean(axis=0)))
rmse_T_hiFreq = np.sqrt((((TPOSE_T_sub20 -TAO_T_sub20)**2).mean(axis=0)))

# Add RMSE difference to plot above 
axR1 = plt.subplot(grid[0:6,25:])
axR1.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axvline(0.2,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axvline(0.3,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.plot(rmse_U_hiFreq[Udepthli+4:Udepthui],U_TAO_140.depth[Udepthli+4:Udepthui],linewidth=1.5,label='<20 days')
axR1.plot(rmse_U_medFreq[Udepthli+4:Udepthui],U_TAO_140.depth[Udepthli+4:Udepthui],linewidth=1.5,label='20-100 days')
axR1.plot(rmse_U_lowFreq[Udepthli+4:Udepthui],U_TAO_140.depth[Udepthli+4:Udepthui],linewidth=1.5,label='100+ days')
axR1.set_xlabel('RMSD (m/s)')
axR1.legend(loc='lower right')
axR1.set_ylim(-249,0)

axR2 = plt.subplot(grid[7:13,25:])
axR2.axvline(0.05,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.15,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.plot(rmse_V_hiFreq[Udepthli+4:Udepthui],V_TAO_140.depth[Udepthli+4:Udepthui],linewidth=1.5,label='<20 days')
axR2.plot(rmse_V_medFreq[Udepthli+4:Udepthui],V_TAO_140.depth[Udepthli+4:Udepthui],linewidth=1.5,label='20-100 days')
axR2.plot(rmse_V_lowFreq[Udepthli+4:Udepthui],V_TAO_140.depth[Udepthli+4:Udepthui],linewidth=1.5,label='100+ days')
axR2.set_xlabel('RMSD (m/s)')
axR2.legend(loc='lower right')
axR2.set_ylim(-249,0)

axR3 = plt.subplot(grid[14:20,25:])
axR3.axvline(0.25,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axvline(0.5,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axvline(0.75,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.plot(rmse_T_hiFreq[Tdepthli:Tdepthui],T_TAO_140.depth[Tdepthli:Tdepthui],linewidth=1.5,label='<20 days')
axR3.plot(rmse_T_medFreq[Tdepthli:Tdepthui],T_TAO_140.depth[Tdepthli:Tdepthui],linewidth=1.5,label='20-100 days')
axR3.plot(rmse_T_lowFreq[Tdepthli:Tdepthui],T_TAO_140.depth[Tdepthli:Tdepthui],linewidth=1.5,label='100+ days')
axR3.set_xlabel('RMSD (deg C)')
axR3.legend(loc='lower right')
axR3.set_ylim(-249,0)

plt.tight_layout()
image_str = 'TAO_TPOSE6_2012to2016_140W.png'
plt.savefig(image_str,format='png',bbox_inches='tight')
plt.close()

del U6_140, V6_140, T6_140, T_TAO_140, U_TAO_140, V_TAO_140

# --------------------------------------------------------------- 170W ---------------------------------------------------------------------------------
# ---------------------------------------------------------------TAO Data ---------------------------------------------------------------------------------
# --------------------------------------------------------------- Velocity ---------------------------------------------------------------------------------

TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N170W_daily.cdf' 
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

Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
U6_170 = ds.UVEL.interp(XG=[190.0],YC=[U_TAO_170.lat],Z=U_TAO_170.depth,time=U_TAO_170.time,method='linear')

temp = U6_170.values
U6_170 = U_TAO_170.copy(deep=True)
U6_170.values = temp[:,:,0,0]
U6_170 = U6_170 + U_TAO_170 - U_TAO_170
U_170_diff = U6_170 - U_TAO_170

meanU_TAO_170 = np.nanmean(U_TAO_170,axis=0)
stdU_TAO_170 = np.nanstd(U_TAO_170,axis=0)

meanU_TP6_170 = np.nanmean(U6_170,axis=0)
stdU_TP6_170 = np.nanstd(U6_170,axis=0)

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
U_170_diff = V6_170 - V_TAO_170

meanV_TAO_170 = np.nanmean(V_TAO_170,axis=0)
stdV_TAO_170 = np.nanstd(V_TAO_170,axis=0)

meanV_TP6_170 = np.nanmean(V6_170,axis=0)
stdV_TP6_170 = np.nanstd(V6_170,axis=0)

# --------------------------------------------------------------- Temperature ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N170W_daily.cdf'  
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
depths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_170 = T_TAO[:,:,latidx,lonTAO170]

Tdepthli = np.argmin(np.abs(depths - zMax))
Tdepthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
T6_170 = ds.THETA.interp(XC=[190.0],YC=[T_TAO_170.lat],Z=T_TAO_170.depth,time=T_TAO_170.time,method='linear')

temp = T6_170.values
T6_170 = T_TAO_170.copy(deep=True)
T6_170.values = temp[:,:,0,0]
T6_170 = T6_170 + T_TAO_170 - T_TAO_170
T_170_diff = T6_170 - T_TAO_170

meanT_TAO_170 = np.nanmean(T_TAO_170,axis=0)
stdT_TAO_170 = np.nanstd(T_TAO_170,axis=0)
meanT_TP6_170 = np.nanmean(T6_170,axis=0)
stdT_TP6_170 = np.nanstd(T6_170,axis=0)

# -----------------------------------------------------------Plotting -----------------------------------------------------------------------------------------

vmin = -1.5
vmax = 1.5
x_axis = np.arange(0,1800,365)
x_labels = ['01/2012','01/2013','01/2014','01/2015','01/2016']
levels = np.arange(vmin,vmax,0.1)

fig = plt.figure(figsize=(27,22))
grid = gs.GridSpec(23, 28)

ax0 = plt.subplot(grid[0:6,0:9])
(U6_170[:,Udepthli:Udepthui].T).plot.contourf(ax=ax0,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax0.set_xticks(x_axis)
ax0.set_xticklabels(x_labels)
ax0.set_xlabel('')
ax0.set_ylabel('Z (m)')
ax0.set_title('Zonal Velocity TPOSE6')
ax0.set_ylim(-249,0)
ax1 = plt.subplot(grid[0:6,10:20])
(U_TAO_170[:,Udepthli:Udepthui].T).plot.contourf(ax=ax1,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax1.set_xticks(x_axis)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('Zonal Velocity TAO')
ax1.set_ylim(-249,0)

ax2 = plt.subplot(grid[0:6,21:24])
ax2.axvline(0.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(0.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(1.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(1.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(-0.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.plot(meanU_TP6_170[Udepthli:Udepthui],U_TAO_170.depth[Udepthli:Udepthui],color='#f6b26b',label='TPOSE')
ax2.fill_betweenx(U_TAO_170.depth[Udepthli:Udepthui],meanU_TP6_170[Udepthli:Udepthui]-stdU_TP6_170[Udepthli:Udepthui],meanU_TP6_170[Udepthli:Udepthui]+stdU_TP6_170[Udepthli:Udepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax2.plot(meanU_TAO_170[Udepthli:Udepthui],U_TAO_170.depth[Udepthli:Udepthui],color='#526e75',label='TAO')
ax2.fill_betweenx(U_TAO_170.depth[Udepthli:Udepthui],meanU_TAO_170[Udepthli:Udepthui]-stdU_TAO_170[Udepthli:Udepthui],meanU_TAO_170[Udepthli:Udepthui]+stdU_TAO_170[Udepthli:Udepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax2.legend(loc='lower right')
ax2.set_xlabel('$m/s$')
ax2.set_ylim(-249,0)

vmin = -1.0
vmax = 1.0
levels = np.arange(vmin,vmax,0.05)

ax3 = plt.subplot(grid[7:13,0:9])
(V6_170[:,Udepthli:Udepthui].T).plot.contourf(ax=ax3,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax3.set_xticks(x_axis)
ax3.set_xticklabels(x_labels)
ax3.set_xlabel('')
ax3.set_ylabel('Z (m)')
ax3.set_title('Meridional Velocity TPOSE')
ax3.set_ylim(-249,0)
ax4 = plt.subplot(grid[7:13,10:20])
(V_TAO_170[:,Udepthli:Udepthui].T).plot.contourf(ax=ax4,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax4.set_xticks(x_axis)
ax4.set_xticklabels(x_labels)
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.set_title('Meridional Velocity TAO')
ax4.set_ylim(-249,0)

ax5 = plt.subplot(grid[7:13,21:24])
ax5.axvline(0.25,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(0.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(-0.25,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.plot(meanV_TP6_170[Udepthli:Udepthui],V_TAO_170.depth[Udepthli:Udepthui],color='#f6b26b',label='TPOSE')
ax5.fill_betweenx(V_TAO_170.depth[Udepthli:Udepthui],meanV_TP6_170[Udepthli:Udepthui]-stdV_TP6_170[Udepthli:Udepthui],meanV_TP6_170[Udepthli:Udepthui]+stdV_TP6_170[Udepthli:Udepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax5.plot(meanV_TAO_170[Udepthli:Udepthui],V_TAO_170.depth[Udepthli:Udepthui],color='#526e75',label='TAO')
ax5.fill_betweenx(V_TAO_170.depth[Udepthli:Udepthui],meanV_TAO_170[Udepthli:Udepthui]-stdV_TAO_170[Udepthli:Udepthui],meanV_TAO_170[Udepthli:Udepthui]+stdV_TAO_170[Udepthli:Udepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax5.legend(loc='lower right')
ax5.set_xlabel('$m/s$')
ax5.set_ylim(-249,0)

vmin = 10
vmax = 30

levels = np.arange(vmin,vmax,0.2)

ax6 = plt.subplot(grid[14:20,0:9])
(T6_170[:,Tdepthli:Tdepthui].T).plot.contourf(ax=ax6,levels=levels,cmap=cmo.thermal,add_colorbar=False)
ax6.set_xticks(x_axis)
ax6.set_xticklabels(x_labels)
ax6.set_xlabel('Time')
ax6.set_ylabel('Z (m)')
ax6.set_title('T TPOSE')
ax6.set_ylim(-249,0)
ax7 = plt.subplot(grid[14:20,10:20])
(T_TAO_170[:,Tdepthli:Tdepthui].T).plot.contourf(ax=ax7,levels=levels,cmap=cmo.thermal,cbar_kwargs={'label':'$deg C$'})
ax7.set_xticks(x_axis)
ax7.set_xticklabels(x_labels)
ax7.set_xlabel('Time')
ax7.set_ylabel('')
ax7.set_title('T TAO')
ax7.set_ylim(-249,0)

ax8 = plt.subplot(grid[14:20,21:24])
ax8.axvline(15,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(20,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(25,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.plot(meanT_TP6_170[Tdepthli:Tdepthui],T_TAO_170.depth[Tdepthli:Tdepthui],color='#f6b26b',label='TPOSE')
ax8.fill_betweenx(T_TAO_170.depth[Tdepthli:Tdepthui],meanT_TP6_170[Tdepthli:Tdepthui]-stdT_TP6_170[Tdepthli:Tdepthui],meanT_TP6_170[Tdepthli:Tdepthui]+stdT_TP6_170[Tdepthli:Tdepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax8.plot(meanT_TAO_170[Tdepthli:Tdepthui],T_TAO_170.depth[Tdepthli:Tdepthui],color='#526e75',label='TAO')
ax8.fill_betweenx(T_TAO_170.depth[Tdepthli:Tdepthui],meanT_TAO_170[Tdepthli:Tdepthui]-stdT_TAO_170[Tdepthli:Tdepthui],meanT_TAO_170[Tdepthli:Tdepthui]+stdT_TAO_170[Tdepthli:Tdepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax8.legend(loc='lower right')
ax8.set_xlabel('$deg C$')
ax8.set_ylim(-249,0)


# ----------------------------------------------------------- RMS Difference -----------------------------------------------------------------------------------------
# Filter TPOSE and TAO into low (>100 day), mid (20-100 day), and high (<20 day) frequency components
# set nan to -9999

U6_170 = U6_170.fillna(-9999)
U_TAO_170 = U_TAO_170.fillna(-9999)

V6_170 = V6_170.fillna(-9999)
V_TAO_170 = V_TAO_170.fillna(-9999)

T6_170 = T6_170.fillna(-9999)
T_TAO_170 = T_TAO_170.fillna(-9999)

# lowpass
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_170 , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_170, axis=0)

TPOSE_V_low = sosfiltfilt(sos, V6_170 , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_170, axis=0)

TPOSE_T_low = sosfiltfilt(sos, T6_170 , axis=0)
TAO_T_low = sosfiltfilt(sos, T_TAO_170, axis=0)

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_170 , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_170, axis=0)

TPOSE_V_sub20 = sosfiltfilt(sos, V6_170 , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_170, axis=0)

TPOSE_T_sub20 = sosfiltfilt(sos, T6_170 , axis=0)
TAO_T_sub20 = sosfiltfilt(sos, T_TAO_170, axis=0)

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_170 , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_170, axis=0)

TPOSE_V_midFreq = sosfiltfilt(sos, V6_170 , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_170, axis=0)

TPOSE_T_midFreq = sosfiltfilt(sos, T6_170 , axis=0)
TAO_T_midFreq = sosfiltfilt(sos, T_TAO_170, axis=0)

N, _ = TPOSE_U_midFreq.shape
# RMS Difference
rmse_U_lowFreq = np.sqrt((((TPOSE_U_low -TAO_U_low)**2).mean(axis=0)))
rmse_V_lowFreq = np.sqrt((((TPOSE_V_low -TAO_V_low)**2).mean(axis=0)))
rmse_T_lowFreq = np.sqrt((((TPOSE_T_low -TAO_T_low)**2).mean(axis=0)))

rmse_U_medFreq = np.sqrt((((TPOSE_U_midFreq-TAO_U_midFreq)**2).mean(axis=0)))
rmse_V_medFreq = np.sqrt((((TPOSE_V_midFreq-TAO_V_midFreq)**2).mean(axis=0)))
rmse_T_medFreq = np.sqrt((((TPOSE_T_midFreq-TAO_T_midFreq)**2).mean(axis=0)))

rmse_U_hiFreq = np.sqrt((((TPOSE_U_sub20 -TAO_U_sub20)**2).mean(axis=0)))
rmse_V_hiFreq = np.sqrt((((TPOSE_V_sub20 -TAO_V_sub20)**2).mean(axis=0)))
rmse_T_hiFreq = np.sqrt((((TPOSE_T_sub20 -TAO_T_sub20)**2).mean(axis=0)))

# Add RMSD to plot above
axR1 = plt.subplot(grid[0:6,25:])
axR1.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axvline(0.2,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axvline(0.3,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.plot(rmse_U_hiFreq[Udepthli+4:Udepthui],U_TAO_170.depth[Udepthli+4:Udepthui],linewidth=1.5,label='<20 days')
axR1.plot(rmse_U_medFreq[Udepthli+4:Udepthui],U_TAO_170.depth[Udepthli+4:Udepthui],linewidth=1.5,label='20-100 days')
axR1.plot(rmse_U_lowFreq[Udepthli+4:Udepthui],U_TAO_170.depth[Udepthli+4:Udepthui],linewidth=1.5,label='100+ days')
axR1.set_xlabel('RMSD (m/s)')
axR1.legend(loc='lower right')
axR1.set_ylim(-250,0)

axR2 = plt.subplot(grid[7:13,25:])
axR2.axvline(0.05,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.15,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.2,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.plot(rmse_V_hiFreq[Udepthli+4:Udepthui],V_TAO_170.depth[Udepthli+4:Udepthui],linewidth=1.5,label='<20 days')
axR2.plot(rmse_V_medFreq[Udepthli+4:Udepthui],V_TAO_170.depth[Udepthli+4:Udepthui],linewidth=1.5,label='20-100 days')
axR2.plot(rmse_V_lowFreq[Udepthli+4:Udepthui],V_TAO_170.depth[Udepthli+4:Udepthui],linewidth=1.5,label='100+ days')
axR2.set_xlabel('RMSD (m/s)')
axR2.legend(loc='lower right')
axR2.set_ylim(-249,0)

axR3 = plt.subplot(grid[14:20,25:])
axR3.axvline(0.25,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axvline(0.5,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axvline(0.75,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.plot(rmse_T_hiFreq[Tdepthli:Tdepthui],T_TAO_170.depth[Tdepthli:Tdepthui],linewidth=1.5,label='<20 days')
axR3.plot(rmse_T_medFreq[Tdepthli:Tdepthui],T_TAO_170.depth[Tdepthli:Tdepthui],linewidth=1.5,label='20-100 days')
axR3.plot(rmse_T_lowFreq[Tdepthli:Tdepthui],T_TAO_170.depth[Tdepthli:Tdepthui],linewidth=1.5,label='100+ days')
axR3.set_xlabel('RMSD (deg C)')
axR3.legend(loc='upper right')
axR3.set_ylim(-249,0)

plt.tight_layout()
image_str = 'TAO_TPOSE6_2012to2016_170W.png'
plt.savefig(image_str,format='png',bbox_inches='tight')
plt.close()

del U6_170, V6_170, T6_170, T_TAO_170, U_TAO_170, V_TAO_170

# --------------------------------------------------------------- 110W ---------------------------------------------------------------------------------
# ---------------------------------------------------------------TAO Data ---------------------------------------------------------------------------------
# --------------------------------------------------------------- Velocity ---------------------------------------------------------------------------------

TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N110W_daily.cdf' 
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

Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
U6_110 = ds.UVEL.interp(XG=[250.0],YC=[U_TAO_110.lat],Z=U_TAO_110.depth,time=U_TAO_110.time,method='linear')

temp = U6_110.values
U6_110 = U_TAO_110.copy(deep=True)
U6_110.values = temp[:,:,0,0]
U6_110 = U6_110 + U_TAO_110 - U_TAO_110
U_110_diff = U6_110 - U_TAO_110

meanU_TAO_110 = np.nanmean(U_TAO_110,axis=0)
stdU_TAO_110 = np.nanstd(U_TAO_110,axis=0)

meanU_TP6_110 = np.nanmean(U6_110,axis=0)
stdU_TP6_110 = np.nanstd(U6_110,axis=0)

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
V_110_diff = V6_110 - V_TAO_110

meanV_TAO_110 = np.nanmean(V_TAO_110,axis=0)
stdV_TAO_110 = np.nanstd(V_TAO_110,axis=0)

meanV_TP6_110 = np.nanmean(V6_110,axis=0)
stdV_TP6_110 = np.nanstd(V6_110,axis=0)

# --------------------------------------------------------------- Temperature ---------------------------------------------------------------------------------
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N110W_daily.cdf' 
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
depths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_110 = T_TAO[:,:,latidx,lonTAO110]

Tdepthli = np.argmin(np.abs(depths - zMax))
Tdepthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
T6_110 = ds.THETA.interp(XC=[250.0],YC=[T_TAO_110.lat],Z=T_TAO_110.depth,time=T_TAO_110.time,method='linear')

temp = T6_110.values
T6_110 = T_TAO_110.copy(deep=True)
T6_110.values = temp[:,:,0,0]
T6_110 = T6_110 + T_TAO_110 - T_TAO_110
T_110_diff = T6_110 - T_TAO_110

meanT_TAO_110 = np.nanmean(T_TAO_110,axis=0)
stdT_TAO_110 = np.nanstd(T_TAO_110,axis=0)
meanT_TP6_110 = np.nanmean(T6_110,axis=0)
stdT_TP6_110 = np.nanstd(T6_110,axis=0)

# -----------------------------------------------------------Plotting -----------------------------------------------------------------------------------------

vmin = -1.5
vmax = 1.5
x_axis = np.arange(0,1800,365)
x_labels = ['01/2012','01/2013','01/2014','01/2015','01/2016']

levels = np.arange(vmin,vmax,0.1)

fig = plt.figure(figsize=(27,22))
grid = gs.GridSpec(23, 28)

ax0 = plt.subplot(grid[0:6,0:9])
(U6_110[:,Udepthli:Udepthui].T).plot.contourf(ax=ax0,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax0.set_xticks(x_axis)
ax0.set_xticklabels(x_labels)
ax0.set_xlabel('')
ax0.set_ylabel('Z (m)')
ax0.set_title('Zonal Velocity TPOSE6')
ax0.set_ylim(-249,0)
ax1 = plt.subplot(grid[0:6,10:20])
(U_TAO_110[:,Udepthli:Udepthui].T).plot.contourf(ax=ax1,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax1.set_xticks(x_axis)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('Zonal Velocity TAO')
ax1.set_ylim(-249,0)

ax2 = plt.subplot(grid[0:6,21:24])
ax2.axvline(0.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(0.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(1.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(1.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axvline(-0.5,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax2.plot(meanU_TP6_110[Udepthli:Udepthui],U_TAO_110.depth[Udepthli:Udepthui],color='#f6b26b',label='TPOSE')
ax2.fill_betweenx(U_TAO_110.depth[Udepthli:Udepthui],meanU_TP6_110[Udepthli:Udepthui]-stdU_TP6_110[Udepthli:Udepthui],meanU_TP6_110[Udepthli:Udepthui]+stdU_TP6_110[Udepthli:Udepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax2.plot(meanU_TAO_110[Udepthli:Udepthui],U_TAO_110.depth[Udepthli:Udepthui],color='#526e75',label='TAO')
ax2.fill_betweenx(U_TAO_110.depth[Udepthli:Udepthui],meanU_TAO_110[Udepthli:Udepthui]-stdU_TAO_110[Udepthli:Udepthui],meanU_TAO_110[Udepthli:Udepthui]+stdU_TAO_110[Udepthli:Udepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax2.legend(loc='lower right')
ax2.set_xlabel('$m/s$')
ax2.set_ylim(-249,0)

vmin = -1.0
vmax = 1.0

levels = np.arange(vmin,vmax,0.05)

ax3 = plt.subplot(grid[7:13,0:9])
(V6_110[:,Udepthli:Udepthui].T).plot.contourf(ax=ax3,levels=levels,cmap=cmo.balance,add_colorbar=False,norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax3.set_xticks(x_axis)
ax3.set_xticklabels(x_labels)
ax3.set_xlabel('')
ax3.set_ylabel('Z (m)')
ax3.set_title('Meridional Velocity TPOSE')
ax3.set_ylim(-249,0)
ax4 = plt.subplot(grid[7:13,10:20])
(V_TAO_110[:,Udepthli:Udepthui].T).plot.contourf(ax=ax4,levels=levels,cmap=cmo.balance,cbar_kwargs={'label':'$m/s$'},norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax))
ax4.set_xticks(x_axis)
ax4.set_xticklabels(x_labels)
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.set_title('Meridional Velocity TAO')
ax4.set_ylim(-249,0)

ax5 = plt.subplot(grid[7:13,21:24])
ax5.axvline(0.2,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(0.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(-0.1,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axvline(-0.2,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax5.plot(meanV_TP6_110[Udepthli:Udepthui],V_TAO_110.depth[Udepthli:Udepthui],color='#f6b26b',label='TPOSE')
ax5.fill_betweenx(V_TAO_110.depth[Udepthli:Udepthui],meanV_TP6_110[Udepthli:Udepthui]-stdV_TP6_110[Udepthli:Udepthui],meanV_TP6_110[Udepthli:Udepthui]+stdV_TP6_110[Udepthli:Udepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax5.plot(meanV_TAO_110[Udepthli:Udepthui],V_TAO_110.depth[Udepthli:Udepthui],color='#526e75',label='TAO')
ax5.fill_betweenx(V_TAO_110.depth[Udepthli:Udepthui],meanV_TAO_110[Udepthli:Udepthui]-stdV_TAO_110[Udepthli:Udepthui],meanV_TAO_110[Udepthli:Udepthui]+stdV_TAO_110[Udepthli:Udepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax5.legend(loc='lower right')
ax5.set_xlabel('$m/s$')
ax5.set_ylim(-249,0)

vmin = 10
vmax = 30
levels = np.arange(vmin,vmax,0.2)

ax6 = plt.subplot(grid[14:20,0:9])
(T6_110[:,Tdepthli:Tdepthui].T).plot.contourf(ax=ax6,levels=levels,cmap=cmo.thermal,add_colorbar=False)
ax6.set_xticks(x_axis)
ax6.set_xticklabels(x_labels)
ax6.set_xlabel('Time')
ax6.set_ylabel('Z (m)')
ax6.set_title('T TPOSE')
ax6.set_ylim(-249,0)
ax7 = plt.subplot(grid[14:20,10:20])
(T_TAO_110[:,Tdepthli:Tdepthui].T).plot.contourf(ax=ax7,levels=levels,cmap=cmo.thermal,cbar_kwargs={'label':'$deg C$'})
ax7.set_xticks(x_axis)
ax7.set_xticklabels(x_labels)
ax7.set_xlabel('Time')
ax7.set_ylabel('')
ax7.set_title('T TAO')
ax7.set_ylim(-249,0)

ax8 = plt.subplot(grid[14:20,21:24])
ax8.axvline(10,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(15,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(20,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axvline(25,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
ax8.plot(meanT_TP6_110[Tdepthli:Tdepthui],T_TAO_110.depth[Tdepthli:Tdepthui],color='#f6b26b',label='TPOSE')
ax8.fill_betweenx(T_TAO_110.depth[Tdepthli:Tdepthui],meanT_TP6_110[Tdepthli:Tdepthui]-stdT_TP6_110[Tdepthli:Tdepthui],meanT_TP6_110[Tdepthli:Tdepthui]+stdT_TP6_110[Tdepthli:Tdepthui],color='#f6b26b',label='_nolegend_',alpha=0.35)
ax8.plot(meanT_TAO_110[Tdepthli:Tdepthui],T_TAO_110.depth[Tdepthli:Tdepthui],color='#526e75',label='TAO')
ax8.fill_betweenx(T_TAO_110.depth[Tdepthli:Tdepthui],meanT_TAO_110[Tdepthli:Tdepthui]-stdT_TAO_110[Tdepthli:Tdepthui],meanT_TAO_110[Tdepthli:Tdepthui]+stdT_TAO_110[Tdepthli:Tdepthui],color='#526e75',label='_nolegend_',alpha=0.35)
ax8.legend(loc='lower right')
ax8.set_xlabel('$deg C$')
ax8.set_ylim(-249,0)

# ----------------------------------------------------------- RMS Difference -----------------------------------------------------------------------------------------
# Filter TPOSE and TAO into low (>100 day), mid (20-100 day), and high (<20 day) frequency components
# set nan to -9999

U6_110 = U6_110.fillna(-9999)
U_TAO_110 = U_TAO_110.fillna(-9999)

V6_110 = V6_110.fillna(-9999)
V_TAO_110 = V_TAO_110.fillna(-9999)

T6_110 = T6_110.fillna(-9999)
T_TAO_110 = T_TAO_110.fillna(-9999)

# lowpass
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)

highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_110 , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_110, axis=0)

TPOSE_V_low = sosfiltfilt(sos, V6_110 , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_110, axis=0)

TPOSE_T_low = sosfiltfilt(sos, T6_110 , axis=0)
TAO_T_low = sosfiltfilt(sos, T_TAO_110, axis=0)

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_110 , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_110, axis=0)

TPOSE_V_sub20 = sosfiltfilt(sos, V6_110 , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_110, axis=0)

TPOSE_T_sub20 = sosfiltfilt(sos, T6_110 , axis=0)
TAO_T_sub20 = sosfiltfilt(sos, T_TAO_110, axis=0)

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_110 , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_110, axis=0)

TPOSE_V_midFreq = sosfiltfilt(sos, V6_110 , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_110, axis=0)

TPOSE_T_midFreq = sosfiltfilt(sos, T6_110 , axis=0)
TAO_T_midFreq = sosfiltfilt(sos, T_TAO_110, axis=0)

N, _ = TPOSE_U_midFreq.shape
# RMS Difference
rmse_U_lowFreq = np.sqrt((((TPOSE_U_low -TAO_U_low)**2).mean(axis=0)))
rmse_V_lowFreq = np.sqrt((((TPOSE_V_low -TAO_V_low)**2).mean(axis=0)))
rmse_T_lowFreq = np.sqrt((((TPOSE_T_low -TAO_T_low)**2).mean(axis=0)))

rmse_U_medFreq = np.sqrt((((TPOSE_U_midFreq-TAO_U_midFreq)**2).mean(axis=0)))
rmse_V_medFreq = np.sqrt((((TPOSE_V_midFreq-TAO_V_midFreq)**2).mean(axis=0)))
rmse_T_medFreq = np.sqrt((((TPOSE_T_midFreq-TAO_T_midFreq)**2).mean(axis=0)))

rmse_U_hiFreq = np.sqrt((((TPOSE_U_sub20 -TAO_U_sub20)**2).mean(axis=0)))
rmse_V_hiFreq = np.sqrt((((TPOSE_V_sub20 -TAO_V_sub20)**2).mean(axis=0)))
rmse_T_hiFreq = np.sqrt((((TPOSE_T_sub20 -TAO_T_sub20)**2).mean(axis=0)))

# Add to figure above
axR1 = plt.subplot(grid[0:6,25:])
axR1.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axvline(0.2,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axvline(0.3,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR1.plot(rmse_U_hiFreq[Udepthli+4:Udepthui],U_TAO_110.depth[Udepthli+4:Udepthui],linewidth=1.5,label='<20 days')
axR1.plot(rmse_U_medFreq[Udepthli+4:Udepthui],U_TAO_110.depth[Udepthli+4:Udepthui],linewidth=1.5,label='20-100 days')
axR1.plot(rmse_U_lowFreq[Udepthli+4:Udepthui],U_TAO_110.depth[Udepthli+4:Udepthui],linewidth=1.5,label='100+ days')
axR1.set_xlabel('RMSD (m/s)')
axR1.legend(loc='lower right')
axR1.set_ylim(-249,0)

axR2 = plt.subplot(grid[7:13,25:])
axR2.axvline(0.05,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.1,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axvline(0.15,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR2.plot(rmse_V_hiFreq[Udepthli+4:Udepthui],V_TAO_110.depth[Udepthli+4:Udepthui],linewidth=1.5,label='<20 days')
axR2.plot(rmse_V_medFreq[Udepthli+4:Udepthui],V_TAO_110.depth[Udepthli+4:Udepthui],linewidth=1.5,label='20-100 days')
axR2.plot(rmse_V_lowFreq[Udepthli+4:Udepthui],V_TAO_110.depth[Udepthli+4:Udepthui],linewidth=1.5,label='100+ days')
axR2.set_xlabel('RMSD (m/s)')
axR2.legend(loc='lower right')
axR2.set_ylim(-249,0)

axR3 = plt.subplot(grid[14:20,25:])
axR3.axvline(0.25,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axvline(0.5,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axvline(0.75,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-50.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-100.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-150.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.axhline(-200.0,color='gray',linewidth=0.3,label='_nolabel_')
axR3.plot(rmse_T_hiFreq[Tdepthli:Tdepthui],T_TAO_110.depth[Tdepthli:Tdepthui],linewidth=1.5,label='<20 days')
axR3.plot(rmse_T_medFreq[Tdepthli:Tdepthui],T_TAO_110.depth[Tdepthli:Tdepthui],linewidth=1.5,label='20-100 days')
axR3.plot(rmse_T_lowFreq[Tdepthli:Tdepthui],T_TAO_110.depth[Tdepthli:Tdepthui],linewidth=1.5,label='100+ days')
axR3.set_xlabel('RMSD (deg C)')
axR3.legend(loc='lower right')
axR3.set_ylim(-249,0)

plt.tight_layout()
image_str = 'TAO_TPOSE6_2012to2016_110W.png'
plt.savefig(image_str,format='png',bbox_inches='tight')
plt.close()
