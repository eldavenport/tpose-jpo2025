# Ellen Davenport April 2025
# This script looks at the correlation between TPOSE and TAO velocity at 140W, 170W, and 110W when filtered into different frequency bands
# Gaps in TAO data are skipped 
# The TPOSE data is filtered into low, mid, and high frequency bands using a butterworth filter

import xarray as xr
from open_tpose import tpose2012to2016
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
plt.rcParams['font.size'] = 16
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

zMax = -35
zMin = -250
Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1
N = 750

# --------------------------------------------------------------- 2012-2013 ---------------------------------------------------------------------------------
# Because of large gaps in TAO 2014 data, it works best to calculate the correlation over 2012 to 2013 and then 2015 to 2016 and average them together 

# crop the time series to the depths we are interested in and the first two years (after that there are some large gaps in TAO data)
U6_140_crop = U6_140[:N,Udepthli:Udepthui]
U_TAO_140_crop = U_TAO_140[:N,Udepthli:Udepthui]
V6_140_crop = V6_140[:N,Udepthli:Udepthui]
V_TAO_140_crop = V_TAO_140[:N,Udepthli:Udepthui]
depths = depths[Udepthli:Udepthui]

# lowpass filter each *time series* which is each row (rows are depth)
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_140_crop , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_140_crop, axis=0)
TPOSE_V_low = sosfiltfilt(sos, V6_140_crop , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_140_crop, axis=0)

# Store in data arrays for coordinates and dimensions
temp = U6_140_crop.copy(deep=True)
temp.values = TAO_U_low
TAO_U_low = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TAO_V_low
TAO_V_low = temp
temp = U6_140_crop.copy(deep=True)
temp.values = TPOSE_U_low
TPOSE_U_low = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TPOSE_V_low
TPOSE_V_low = temp

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_140_crop , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_140_crop, axis=0)
TPOSE_V_sub20 = sosfiltfilt(sos, V6_140_crop , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_140_crop, axis=0)

# Store in data arrays for coordinates and dimensions
temp = U6_140_crop.copy(deep=True)
temp.values = TAO_U_sub20
TAO_U_sub20 = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TAO_V_sub20
TAO_V_sub20 = temp
temp = U6_140_crop.copy(deep=True)
temp.values = TPOSE_U_sub20
TPOSE_U_sub20 = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TPOSE_V_sub20
TPOSE_V_sub20 = temp

# mid frequency filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_140_crop , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_140_crop, axis=0)
TPOSE_V_midFreq = sosfiltfilt(sos, V6_140_crop , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_140_crop, axis=0)

# Store in data arrays for coordinates and dimensions
temp = U6_140_crop.copy(deep=True)
temp.values = TAO_U_midFreq
TAO_U_midFreq = temp

temp = V6_140_crop.copy(deep=True)
temp.values = TAO_V_midFreq
TAO_V_midFreq = temp

temp = U6_140_crop.copy(deep=True)
temp.values = TPOSE_U_midFreq
TPOSE_U_midFreq = temp

temp = V6_140_crop.copy(deep=True)
temp.values = TPOSE_V_midFreq
TPOSE_V_midFreq = temp

# Compute correlation at 140W for zonal and meridional velocity at low, med, high frequency 
U_corr_low = xr.corr(TPOSE_U_low, TAO_U_low, dim="time") # correlation in time of the zonal velocity at low frequency
U_corr_mid = xr.corr(TPOSE_U_midFreq, TAO_U_midFreq, dim="time") # correlation in time of the zonal velocity at mid frequencies
U_corr_high = xr.corr(TPOSE_U_sub20, TAO_U_sub20, dim="time") # correlation in time of the zonal velocity at high frequency

V_corr_low = xr.corr(TPOSE_V_low, TAO_V_low, dim="time") # correlation in time of the meridional velocity at low frequency
V_corr_mid = xr.corr(TPOSE_V_midFreq, TAO_V_midFreq, dim="time") # correlation in time of the meridional velocity at mid frequencies
V_corr_high = xr.corr(TPOSE_V_sub20, TAO_V_sub20, dim="time") # correlation in time of the meridional velocity at high frequencies

# --------------------------------------------------------------- 2015-2016 ---------------------------------------------------------------------------------
N = -500
U6_140_crop = U6_140[N:,Udepthli:Udepthui-2]
U_TAO_140_crop = U_TAO_140[N:,Udepthli:Udepthui-2]
V6_140_crop = V6_140[N:,Udepthli:Udepthui-2]
V_TAO_140_crop = V_TAO_140[N:,Udepthli:Udepthui-2]

# filter each *time series* which is each row (rows are depth)
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_140_crop , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_140_crop, axis=0)
TPOSE_V_low = sosfiltfilt(sos, V6_140_crop , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_140_crop, axis=0)

# store in data arrays
temp = U6_140_crop.copy(deep=True)
temp.values = TAO_U_low
TAO_U_low = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TAO_V_low
TAO_V_low = temp
temp = U6_140_crop.copy(deep=True)
temp.values = TPOSE_U_low
TPOSE_U_low = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TPOSE_V_low
TPOSE_V_low = temp

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_140_crop , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_140_crop, axis=0)
TPOSE_V_sub20 = sosfiltfilt(sos, V6_140_crop , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_140_crop, axis=0)

# store in data arrays
temp = U6_140_crop.copy(deep=True)
temp.values = TAO_U_sub20
TAO_U_sub20 = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TAO_V_sub20
TAO_V_sub20 = temp
temp = U6_140_crop.copy(deep=True)
temp.values = TPOSE_U_sub20
TPOSE_U_sub20 = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TPOSE_V_sub20
TPOSE_V_sub20 = temp

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_140_crop , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_140_crop, axis=0)
TPOSE_V_midFreq = sosfiltfilt(sos, V6_140_crop , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_140_crop, axis=0)

# store in data arrays
temp = U6_140_crop.copy(deep=True)
temp.values = TAO_U_midFreq
TAO_U_midFreq = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TAO_V_midFreq
TAO_V_midFreq = temp
temp = U6_140_crop.copy(deep=True)
temp.values = TPOSE_U_midFreq
TPOSE_U_midFreq = temp
temp = V6_140_crop.copy(deep=True)
temp.values = TPOSE_V_midFreq
TPOSE_V_midFreq = temp

# Compute correlation at 140W for zonal and meridional velocity at low, med, high frequency 
U_corr_low_2 = xr.corr(TPOSE_U_low, TAO_U_low, dim="time") # correlation in time of the zonal velocity at low frequency
U_corr_mid_2 = xr.corr(TPOSE_U_midFreq, TAO_U_midFreq, dim="time") # correlation in time of the zonal velocity at mid frequency
U_corr_high_2 = xr.corr(TPOSE_U_sub20, TAO_U_sub20, dim="time")# correlation in time of the zonal velocity at high frequency

V_corr_low_2 = xr.corr(TPOSE_V_low, TAO_V_low, dim="time") # correlation in time of the meridional velocity at low frequency
V_corr_mid_2 = xr.corr(TPOSE_V_midFreq, TAO_V_midFreq, dim="time") # correlation in time of the meridional velocity at mid frequency
V_corr_high_2 = xr.corr(TPOSE_V_sub20, TAO_V_sub20, dim="time") # correlation in time of the meridional velocity at high frequency

# ----------------------------------------------------------- average over all of the time we have -----------------------------------------------------
U_corr_low_140 = (U_corr_low[:-2] + U_corr_low_2)/2
U_corr_mid_140 = (U_corr_mid[:-2] + U_corr_mid_2)/2
U_corr_high_140 = (U_corr_high[:-2] + U_corr_high_2)/2
V_corr_low_140 = (V_corr_low[:-2] + V_corr_low_2)/2
V_corr_mid_140 = (V_corr_mid[:-2] + V_corr_mid_2)/2
V_corr_high_140 = (V_corr_high[:-2] + V_corr_high_2)/2

# ----------------------------------------------------------- 170W TAO Data ----------------------------------------------------------------------------
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

zMax = -35
zMin = -235
Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1

# --------------------------------------------------------------- First time period ---------------------------------------------------------------------------------
# Because of gaps in TAO data, it works best to calculate the correlation over two time periods (before and after the gap) and average them together 
N = 913
U6_170_crop = U6_170[:N,Udepthli:Udepthui]
U_TAO_170_crop = U_TAO_170[:N,Udepthli:Udepthui]
V6_170_crop = V6_170[:N,Udepthli:Udepthui]
V_TAO_170_crop = V_TAO_170[:N,Udepthli:Udepthui]
depths = depths[Udepthli:Udepthui]

# filter each *time series* which is each row (rows are depth)
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_170_crop , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_170_crop, axis=0)
TPOSE_V_low = sosfiltfilt(sos, V6_170_crop , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_170_crop, axis=0)

# store in data arrays
temp = U6_170_crop.copy(deep=True)
temp.values = TAO_U_low
TAO_U_low = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TAO_V_low
TAO_V_low = temp
temp = U6_170_crop.copy(deep=True)
temp.values = TPOSE_U_low
TPOSE_U_low = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TPOSE_V_low
TPOSE_V_low = temp

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_170_crop , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_170_crop, axis=0)
TPOSE_V_sub20 = sosfiltfilt(sos, V6_170_crop , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_170_crop, axis=0)

# store in data arrays
temp = U6_170_crop.copy(deep=True)
temp.values = TAO_U_sub20
TAO_U_sub20 = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TAO_V_sub20
TAO_V_sub20 = temp
temp = U6_170_crop.copy(deep=True)
temp.values = TPOSE_U_sub20
TPOSE_U_sub20 = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TPOSE_V_sub20
TPOSE_V_sub20 = temp

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_170_crop , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_170_crop, axis=0)
TPOSE_V_midFreq = sosfiltfilt(sos, V6_170_crop , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_170_crop, axis=0)

# store in data arrays
temp = U6_170_crop.copy(deep=True)
temp.values = TAO_U_midFreq
TAO_U_midFreq = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TAO_V_midFreq
TAO_V_midFreq = temp
temp = U6_170_crop.copy(deep=True)
temp.values = TPOSE_U_midFreq
TPOSE_U_midFreq = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TPOSE_V_midFreq
TPOSE_V_midFreq = temp

# correlations
U_corr_low = xr.corr(TPOSE_U_low, TAO_U_low, dim="time") # correlation in time of the zonal velocity at every depth
U_corr_mid = xr.corr(TPOSE_U_midFreq, TAO_U_midFreq, dim="time") # correlation in time of the meridional velocity at every depth
U_corr_high = xr.corr(TPOSE_U_sub20, TAO_U_sub20, dim="time") # correlation in time of the temperature at every depth

V_corr_low = xr.corr(TPOSE_V_low, TAO_V_low, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_mid = xr.corr(TPOSE_V_midFreq, TAO_V_midFreq, dim="time") # correlation in time of the meridional velocity at every depth
V_corr_high = xr.corr(TPOSE_V_sub20, TAO_V_sub20, dim="time") # correlation in time of the temperature at every depth

# --------------------------------------------------------------- Second time period ---------------------------------------------------------------------------------
# Because of gaps in TAO data, it works best to calculate the correlation over two time periods (before and after the gap) and average them together 
N = -750
U6_170_crop = U6_170[N:,Udepthli:Udepthui-2]
U_TAO_170_crop = U_TAO_170[N:,Udepthli:Udepthui-2]
V6_170_crop = V6_170[N:,Udepthli:Udepthui-2]
V_TAO_170_crop = V_TAO_170[N:,Udepthli:Udepthui-2]

# filter each *time series* which is each row (rows are depth)
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_170_crop , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_170_crop, axis=0)
TPOSE_V_low = sosfiltfilt(sos, V6_170_crop , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_170_crop, axis=0)

# store in data arrays
temp = U6_170_crop.copy(deep=True)
temp.values = TAO_U_low
TAO_U_low = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TAO_V_low
TAO_V_low = temp
temp = U6_170_crop.copy(deep=True)
temp.values = TPOSE_U_low
TPOSE_U_low = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TPOSE_V_low
TPOSE_V_low = temp

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_170_crop , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_170_crop, axis=0)
TPOSE_V_sub20 = sosfiltfilt(sos, V6_170_crop , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_170_crop, axis=0)

# store in data arrays
temp = U6_170_crop.copy(deep=True)
temp.values = TAO_U_sub20
TAO_U_sub20 = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TAO_V_sub20
TAO_V_sub20 = temp
temp = U6_170_crop.copy(deep=True)
temp.values = TPOSE_U_sub20
TPOSE_U_sub20 = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TPOSE_V_sub20
TPOSE_V_sub20 = temp

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_170_crop , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_170_crop, axis=0)
TPOSE_V_midFreq = sosfiltfilt(sos, V6_170_crop , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_170_crop, axis=0)

# store in data arrays
temp = U6_170_crop.copy(deep=True)
temp.values = TAO_U_midFreq
TAO_U_midFreq = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TAO_V_midFreq
TAO_V_midFreq = temp
temp = U6_170_crop.copy(deep=True)
temp.values = TPOSE_U_midFreq
TPOSE_U_midFreq = temp
temp = V6_170_crop.copy(deep=True)
temp.values = TPOSE_V_midFreq
TPOSE_V_midFreq = temp

# correlations
U_corr_low_2 = xr.corr(TPOSE_U_low, TAO_U_low, dim="time") # correlation in time of the zonal velocity at every depth
U_corr_mid_2 = xr.corr(TPOSE_U_midFreq, TAO_U_midFreq, dim="time") # correlation in time of the meridional velocity at every depth
U_corr_high_2 = xr.corr(TPOSE_U_sub20, TAO_U_sub20, dim="time") # correlation in time of the temperature at every depth

V_corr_low_2 = xr.corr(TPOSE_V_low, TAO_V_low, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_mid_2 = xr.corr(TPOSE_V_midFreq, TAO_V_midFreq, dim="time") # correlation in time of the meridional velocity at every depth
V_corr_high_2 = xr.corr(TPOSE_V_sub20, TAO_V_sub20, dim="time") # correlation in time of the temperature at every depth

# ----------------------------------------------------------- average over all of the time we have------------------------------------------------

U_corr_low_170 = (U_corr_low[:-2] + U_corr_low_2)/2
U_corr_mid_170 = (U_corr_mid[:-2] + U_corr_mid_2)/2
U_corr_high_170 = (U_corr_high[:-2] + U_corr_high_2)/2

V_corr_low_170 = (V_corr_low[:-2] + V_corr_low_2)/2
V_corr_mid_170 = (V_corr_mid[:-2] + V_corr_mid_2)/2
V_corr_high_170 = (V_corr_high[:-2] + V_corr_high_2)/2

# ----------------------------------------------------------- 110W  TAO DATA ---------------------------------------------------------------------

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

Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1

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

zMax = -35
zMin = -195
Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1
N = -1

# crop the time series to the depths we are interested in and the first two years (after that there are some large gaps in TAO data)
U6_110_crop = U6_110[:N,Udepthli:Udepthui]
U_TAO_110_crop = U_TAO_110[:N,Udepthli:Udepthui]
V6_110_crop = V6_110[:N,Udepthli:Udepthui]
V_TAO_110_crop = V_TAO_110[:N,Udepthli:Udepthui]
depths = depths[Udepthli:Udepthui]

# filter each *time series* which is each row (rows are depth)
fs = 1/86400 # sampling rate is 1 day (86400 seconds per day)
highF = (1/100)*fs
order = 4
sos = butter(order, np.array(highF), 'lowpass', fs=fs, output='sos')

TPOSE_U_low = sosfiltfilt(sos, U6_110_crop , axis=0)
TAO_U_low = sosfiltfilt(sos, U_TAO_110_crop, axis=0)
TPOSE_V_low = sosfiltfilt(sos, V6_110_crop , axis=0)
TAO_V_low = sosfiltfilt(sos, V_TAO_110_crop, axis=0)

# store in data arrays
temp = U6_110_crop.copy(deep=True)
temp.values = TAO_U_low
TAO_U_low = temp
temp = V6_110_crop.copy(deep=True)
temp.values = TAO_V_low
TAO_V_low = temp
temp = U6_110_crop.copy(deep=True)
temp.values = TPOSE_U_low
TPOSE_U_low = temp
temp = V6_110_crop.copy(deep=True)
temp.values = TPOSE_V_low
TPOSE_V_low = temp

# highpass filter
lowF = (1/20)*fs 
cutoff = np.array(lowF)
order = 4
sos = butter(order, cutoff, 'highpass', fs=fs, output='sos')

TPOSE_U_sub20 = sosfiltfilt(sos, U6_110_crop , axis=0)
TAO_U_sub20 = sosfiltfilt(sos, U_TAO_110_crop, axis=0)
TPOSE_V_sub20 = sosfiltfilt(sos, V6_110_crop , axis=0)
TAO_V_sub20 = sosfiltfilt(sos, V_TAO_110_crop, axis=0)

# store in data arrays
temp = U6_110_crop.copy(deep=True)
temp.values = TAO_U_sub20
TAO_U_sub20 = temp
temp = V6_110_crop.copy(deep=True)
temp.values = TAO_V_sub20
TAO_V_sub20 = temp
temp = U6_110_crop.copy(deep=True)
temp.values = TPOSE_U_sub20
TPOSE_U_sub20 = temp
temp = V6_110_crop.copy(deep=True)
temp.values = TPOSE_V_sub20
TPOSE_V_sub20 = temp

# midrange filter
highF = (1/20)*fs 
lowF = (1/100)*fs
order = 4
sos = butter(order, np.array([lowF, highF]), 'bandpass', fs=fs, output='sos')

TPOSE_U_midFreq = sosfiltfilt(sos, U6_110_crop , axis=0)
TAO_U_midFreq = sosfiltfilt(sos, U_TAO_110_crop, axis=0)
TPOSE_V_midFreq = sosfiltfilt(sos, V6_110_crop , axis=0)
TAO_V_midFreq = sosfiltfilt(sos, V_TAO_110_crop, axis=0)

# store in data arrays
temp = U6_110_crop.copy(deep=True)
temp.values = TAO_U_midFreq
TAO_U_midFreq = temp
temp = V6_110_crop.copy(deep=True)
temp.values = TAO_V_midFreq
TAO_V_midFreq = temp
temp = U6_110_crop.copy(deep=True)
temp.values = TPOSE_U_midFreq
TPOSE_U_midFreq = temp
temp = V6_110_crop.copy(deep=True)
temp.values = TPOSE_V_midFreq
TPOSE_V_midFreq = temp

# correlations 
U_corr_low_110 = xr.corr(TPOSE_U_low, TAO_U_low, dim="time") # correlation in time of the zonal velocity at every depth
U_corr_mid_110 = xr.corr(TPOSE_U_midFreq, TAO_U_midFreq, dim="time") # correlation in time of the meridional velocity at every depth
U_corr_high_110 = xr.corr(TPOSE_U_sub20, TAO_U_sub20, dim="time") # correlation in time of the temperature at every depth

V_corr_low_110 = xr.corr(TPOSE_V_low, TAO_V_low, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_mid_110 = xr.corr(TPOSE_V_midFreq, TAO_V_midFreq, dim="time") # correlation in time of the meridional velocity at every depth
V_corr_high_110 = xr.corr(TPOSE_V_sub20, TAO_V_sub20, dim="time") # correlation in time of the temperature at every depth

# ----------------------------------------------------------- Plot  ------------------------------------------------

# plot everything together (u, v at all three longitudes for all frequencies)
fig, ax = plt.subplots(ncols=3,nrows=2,figsize=(15,13))
ax[0,0].plot(U_corr_high_170,U_TAO_170_crop.depth,linewidth=2.0,label='<20 days')
ax[0,0].plot(U_corr_mid_170,U_TAO_170_crop.depth,linewidth=2.0,label='20-100 days')
ax[0,0].plot(U_corr_low_170,U_TAO_170_crop.depth,linewidth=2.0,label='>100 days')
ax[0,0].set_xlabel('Correlation')
ax[0,0].set_ylabel('Depth (m)')
ax[0,0].legend(loc='upper right')
ax[0,0].axvline(0,linewidth=0.75,color='tab:grey')
ax[0,0].set_xlim(-0.1,1)
ax[0,0].set_ylim(-250,0)
ax[0,0].set_title('u 0N, 170W')

ax[0,1].plot(U_corr_high_140,U_TAO_140_crop.depth,linewidth=2.0,label='<20 days')
ax[0,1].plot(U_corr_mid_140,U_TAO_140_crop.depth,linewidth=2.0,label='20-100 days')
ax[0,1].plot(U_corr_low_140,U_TAO_140_crop.depth,linewidth=2.0,label='>100 days')
ax[0,1].set_xlabel('Correlation')
ax[0,1].set_ylabel('Depth (m)')
ax[0,1].legend(loc='upper left')
ax[0,1].axvline(0,linewidth=0.75,color='tab:grey')
ax[0,1].set_xlim(-0.1,1)
ax[0,1].set_ylim(-250,0)
ax[0,1].set_title('u 0N, 140W')

ax[0,2].plot(U_corr_high_110,U_TAO_110_crop.depth,linewidth=2.0,label='<20 days')
ax[0,2].plot(U_corr_mid_110,U_TAO_110_crop.depth,linewidth=2.0,label='20-100 days')
ax[0,2].plot(U_corr_low_110,U_TAO_110_crop.depth,linewidth=2.0,label='>100 days')
ax[0,2].set_xlabel('Correlation')
ax[0,2].set_ylabel('Depth (m)')
ax[0,2].legend(loc='upper right')
ax[0,2].axvline(0,linewidth=0.75,color='tab:grey')
ax[0,2].set_xlim(-0.1,1)
ax[0,2].set_ylim(-250,0)
ax[0,2].set_title('u 0N, 110W')

ax[1,0].plot(V_corr_high_170,V_TAO_170_crop.depth,linewidth=2.0,label='<20 days')
ax[1,0].plot(V_corr_mid_170,V_TAO_170_crop.depth,linewidth=2.0,label='20-100 days')
ax[1,0].plot(V_corr_low_170,V_TAO_170_crop.depth,linewidth=2.0,label='>100 days')
ax[1,0].set_xlabel('Correlation')
ax[1,0].set_ylabel('Depth (m)')
ax[1,0].legend(loc='upper right')
ax[1,0].axvline(0,linewidth=0.75,color='tab:grey')
ax[1,0].set_xlim(-0.5,1)
ax[1,0].set_ylim(-250,0)
ax[1,0].set_title('v 0N, 170W')

ax[1,1].plot(V_corr_high_140,V_TAO_140_crop.depth,linewidth=2.0,label='<20 days')
ax[1,1].plot(V_corr_mid_140,V_TAO_140_crop.depth,linewidth=2.0,label='20-100 days')
ax[1,1].plot(V_corr_low_140,V_TAO_140_crop.depth,linewidth=2.0,label='>100 days')
ax[1,1].set_xlabel('Correlation')
ax[1,1].set_ylabel('Depth (m)')
ax[1,1].legend(loc='upper left')
ax[1,1].axvline(0,linewidth=0.75,color='tab:grey')
ax[1,1].set_xlim(-0.5,1)
ax[1,1].set_ylim(-250,0)
ax[1,1].set_title('v 0N, 140W')

ax[1,2].plot(V_corr_high_110,V_TAO_110_crop.depth,linewidth=2.0,label='<20 days')
ax[1,2].plot(V_corr_mid_110,V_TAO_110_crop.depth,linewidth=2.0,label='20-100 days')
ax[1,2].plot(V_corr_low_110,V_TAO_110_crop.depth,linewidth=2.0,label='>100 days')
ax[1,2].set_xlabel('Correlation')
ax[1,2].set_ylabel('Depth (m)')
ax[1,2].legend(loc='upper right')
ax[1,2].axvline(0,linewidth=0.75,color='tab:grey')
ax[1,2].set_xlim(-0.5,1)
ax[1,2].set_ylim(-250,0)
ax[1,2].set_title('v 0N, 110W')

plt.tight_layout()
plt.savefig('V_U_correlation_filtered.png',format='png')