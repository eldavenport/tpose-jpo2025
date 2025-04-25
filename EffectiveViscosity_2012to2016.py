#Ellen Davenport April 2025
# This script generates Figure 10 for Davenport et al. 2025. 
# The first panel compares average viscosity profiles from TPOSE with independent data sets
# The second panel compares average viscosity profiles from SON 2012-2016 within TPOSE
# All estimates include standard error where possible

import sys
from open_tpose import tpose2012to2016_kpp, tposeOct2012_hourly
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
warnings.filterwarnings("ignore")
import xarray as xr

plt.rcParams['font.size'] = 17
prefix = ['diag_kpp','diag_state']

# open the TPOSE 2012-2016 KPP diagnostics
ds = tpose2012to2016_kpp(prefix)

lon = float(sys.argv[1])
zMin = float(sys.argv[2])
latMin = -0.1
latMax = 0.1

lats = ds.YC.data
lons = ds.XC.data
depths = ds.Z.data

latli = np.argmin(np.abs(lats - latMin))
latui = np.argmin(np.abs(lats - latMax)) + 1
lonidx = np.argmin(np.abs(lons - lon))
zMax = 0
depthli = np.argmin(np.abs(depths - zMax))
depthui = np.argmin(np.abs(depths - zMin)) + 1

N = len(ds.time)
ds['time'] = range(len(ds.time))

# ----------------------------------------------------- Estimate Seasonal Effective Viscosity for SON of each year ------------------------------------------------------
# Model KPP diagnostics begins in May of 2012 
MAMli = 0
MAMui = MAMli + 92
JJAui = MAMui + 92
SONui = JJAui + 92

# Use decorrelation scale estimated for TPOSE viscosity to get a standard error from standard deviation
decorr_scale_days = 5 #days
N_ind = (SONui-JJAui)/decorr_scale_days # number of days in a season / decorrelation scale to get independent samples
std_err_denom = np.sqrt(N_ind)

SON_EddyVisc_2012 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')  
SON_EddyVisc_2012_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom  
SON_EUC_Core_2012 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')

#2013
DJFli = SONui 
DJFui = DJFli + 31 + 31 + 28
MAMui = DJFui + 31 + 30 + 31
JJAui = MAMui + 30 + 31 + 31
SONui = JJAui + 30 + 31 + 30

decorr_scale_days = 10 #days
N_ind = (DJFui)/decorr_scale_days
std_err_denom = np.sqrt(N_ind)

EddyVisc_2012 = ds.KPPviscA[:DJFui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')  
EddyVisc_2012_std = ds.KPPviscA[:DJFui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

decorr_scale_days = 5 #days
N_ind = (SONui-JJAui)/decorr_scale_days
std_err_denom = np.sqrt(N_ind)
 
SON_EddyVisc_2013 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')  
SON_EddyVisc_2013_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom  
SON_EUC_Core_2013 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')

#2014
DJFli = SONui 
DJFui = DJFli + 31 + 31 + 28
MAMui = DJFui + 31 + 30 + 31
JJAui = MAMui + 30 + 31 + 31
SONui = JJAui + 30 + 31 + 30

SON_EddyVisc_2014 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')  
SON_EddyVisc_2014_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom  
SON_EUC_Core_2014 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')

#2015
DJFli = SONui 
DJFui = DJFli + 31 + 31 + 28
MAMui = DJFui + 31 + 30 + 31
JJAui = MAMui + 30 + 31 + 31
SONui = JJAui + 30 + 31 + 30

SON_EddyVisc_2015 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')  
SON_EddyVisc_2015_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom  
SON_EUC_Core_2015 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')

#2016
DJFli = SONui 
DJFui = DJFli + 31 + 31 + 29 # leap year
MAMui = DJFui + 31 + 30 + 31
JJAui = MAMui + 30 + 31 + 31
SONui = JJAui + 30 + 31 + 30

SON_EddyVisc_2016 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')  
SON_EddyVisc_2016_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom  
SON_EUC_Core_2016 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')

# ----------------------------------------------------- Estimate Effective Viscosity for Hourly Model Oct 2012 ------------------------------------------------------

ds_hourly = tposeOct2012_hourly(prefix)

N = len(ds_hourly.time)
ds_hourly['time'] = range(0,N,1)

# ----------------------------------------------------- Include Independent Data Sets ------------------------------------------------------

# Estimated viscosity from Whitt et al. 2022. Discard the values at the very edge of the domain
DWLES_ds = loadmat('Whitt22_LES_profiles.mat')
DWLES_x = DWLES_ds['prof_eddy_visc_Dan'][10:,0]
DWLES_std = DWLES_ds['std_eddy_visc_Dan'][10:,0]
DWLES_y = DWLES_ds['z'][10:,0]

# estimated in matlab using Whitt et al. 2022 output
decorr_scale_DWLES = 930
N_ind_DW = 8774/decorr_scale_DWLES
std_err_denom_DW = np.sqrt(N_ind_DW)
DWLES_std = DWLES_std/std_err_denom_DW

# remove the first day of TPOSE hourly diagnostics
offset = 24
N = N - offset
decorr_scale_hours = 72
N_ind = N/decorr_scale_hours
std_err_denom = np.sqrt(N_ind)

# Viscosity estimate from hourly TPOSE data 
Avg_EddyVisc = ds_hourly.KPPviscA[offset:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') # now this is 1xdepth
Avg_EddyVisc_std = (ds_hourly.KPPviscA[offset:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time'))/std_err_denom # now this is 1xdepth

# Data from Dillon et al. 1989
Dillon_89_x2 = np.array([1.8*10**-3, 6.5*10**-4, 7.0*10 **-4, 6.5*10**-4, 4.5*10**-4, 1.7*10**-4, 8.5*10**-5])
Dillon_89_y2 = [-30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0]

# Data from Dillon et al. 1989
Dillon_89_x1 = np.array([1.8*10**-3, 5.5*10**-4, 3.5*10**-4, 9.0*10**-5, 4.5*10**-5, 2.0*10**-5])
Dillon_89_y1 = [-32.0, -45.0, -55.0, -69.0, -80.0, -92.0]

# Data from Pinkel et al., 2023. Bootstrap method in 'EquatorMix_ViscoistyBootstrap.py' generates the .nc file below from EquatorMix data
filename_pinkel = 'Fig4e_extracted_viscosity_bootstrap.nc'
ds_pinkel = xr.open_dataset(filename_pinkel)
Pinkel_23_x1 = ds_pinkel.viscosity
Pinkel_23_y1 = ds_pinkel.depth
Pinkel_23_std = ds_pinkel.std_error

# Data from Qiao and Weisberg 1996
QW_96_x1 = np.array([46, 40, 35, 30, 20, 17, 10, 8, 4, 4, 10, 20, 40])*10**-4
QW_96_y1 = [0.0, -15.0, -25.0, -32.0, -45.0, -50.0, -67.0, -75.0, -100.0, -125.0, -145.0, -175.0, -200.0]
QW_96_std = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,2,3,4,8])*10**-4

# Data from Bryden and Brady 1985
BB_85_x1 = np.array([1.7*10**-3])
BB_85_y1 = [-75.0]

# -----------------------------------------------------------------PLOTTING----------------------------------------------------------------------------------#
fig, ax = plt.subplots(figsize=(16,12),ncols=2)
ax[0].fill_betweenx(Pinkel_23_y1[6:34],Pinkel_23_x1[6:34]-Pinkel_23_std[6:34],Pinkel_23_x1[6:34]+Pinkel_23_std[6:34],color='b',label='_nolegend_',alpha=0.12)
ax[0].plot(Pinkel_23_x1[6:34],Pinkel_23_y1[6:34],marker='o',linewidth=1.5,color='b',label='P23 (+0.3)')
ax[0].plot(QW_96_x1,QW_96_y1,marker='o',linewidth=2.0,markersize=8,color='r',label='Q96 (+0.4)')
ax[0].fill_betweenx(QW_96_y1,QW_96_x1-QW_96_std,QW_96_x1+QW_96_std,color='r',label='_nolegend_',alpha=0.25)
ax[0].plot(EddyVisc_2012,ds.Z[depthli:depthui],linewidth=1.5, color='#17712c',label='TPOSE 2012 (-0.2)')
ax[0].fill_betweenx(ds.Z[depthli:depthui],EddyVisc_2012-EddyVisc_2012_std,EddyVisc_2012+EddyVisc_2012_std,color='#17712c',label='_nolegend_',alpha=0.25)
ax[0].plot(Avg_EddyVisc,ds_hourly.Z[depthli:depthui],linewidth=1.5,color='#0acd38',label='TPOSE OCT 2012 (+0.3)')
ax[0].fill_betweenx(ds_hourly.Z[depthli:depthui],Avg_EddyVisc-Avg_EddyVisc_std,Avg_EddyVisc+Avg_EddyVisc_std,color='#0acd38',label='_nolegend_',alpha=0.25)
ax[0].plot(DWLES_x,DWLES_y,linewidth=2.0,color='tab:orange',label='W22 (-0.3)')
ax[0].fill_betweenx(DWLES_y,DWLES_x-DWLES_std,DWLES_x+DWLES_std,color='tab:orange',label='_nolegend_',alpha=0.25)
ax[0].plot(Dillon_89_x2,Dillon_89_y2,marker='s',linewidth=1.5,markersize=8,color='k',label='D89 GPWOS (-0.75)')
ax[0].plot(Dillon_89_x1,Dillon_89_y1,marker='o',linewidth=1.5,markersize=8,color='k',label='D89 MC (-0.75)')
ax[0].plot(BB_85_x1,BB_85_y1,marker='*',color='m',label='BB85 (+0.1)',markersize=20)
ax[0].semilogx()
ax[0].set_title('TPOSE and Observations')
ax[0].set_xlabel('$m^2/s$')
ax[0].legend(loc='lower right')
ax[0].set_ylim(ds.Z[depthui-1]-10,0)
ax[0].set_ylabel('Z (m)')
ax[0].set_xlim(5e-6,3.0*10**-2)


ax[1].plot(SON_EddyVisc_2012,ds.Z[depthli:depthui],linewidth=2.5,color='#808b96',label='2012 (+0.3)')
ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2012-SON_EddyVisc_2012_std,SON_EddyVisc_2012+SON_EddyVisc_2012_std,color='#808b96',label='_nolegend_',alpha=0.2)
ax[1].axhline(SON_EUC_Core_2012,linewidth=1.5,color='#808b96',label='_nolabel_')
ax[1].plot(SON_EddyVisc_2013,ds.Z[depthli:depthui],linestyle='--',linewidth=2.5,color='#808b96',label='2013 (-0.2)')
ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2013-SON_EddyVisc_2013_std,SON_EddyVisc_2013+SON_EddyVisc_2013_std,color='#808b96',label='_nolegend_',alpha=0.2)
ax[1].axhline(SON_EUC_Core_2013,linestyle='--',linewidth=1.5,color='#808b96',label='_nolabel_')
ax[1].plot(SON_EddyVisc_2014,ds.Z[depthli:depthui],linewidth=2.5,color='#c0392b',label='2014 (+0.5)')
ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2014-SON_EddyVisc_2014_std,SON_EddyVisc_2014+SON_EddyVisc_2014_std,color='#c0392b',label='_nolegend_',alpha=0.2)
ax[1].axhline(SON_EUC_Core_2014,linewidth=1.5,color='#c0392b',label='_nolabel_')
ax[1].plot(SON_EddyVisc_2015,ds.Z[depthli:depthui],linestyle='--',linewidth=2.5,color='#c0392b',label='2015 (+2.4)')
ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2015-SON_EddyVisc_2015_std,SON_EddyVisc_2015+SON_EddyVisc_2015_std,color='#c0392b',label='_nolegend_',alpha=0.2)
ax[1].axhline(SON_EUC_Core_2015,linestyle='--',linewidth=1.5,color='#c0392b',label='_nolabel_')
ax[1].plot(SON_EddyVisc_2016,ds.Z[depthli:depthui],linestyle='-.',linewidth=2.5,color='#082f99',label='2016 (-0.7)')
ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2016-SON_EddyVisc_2016_std,SON_EddyVisc_2016+SON_EddyVisc_2016_std,color='#082f99',label='_nolegend_',alpha=0.2)
ax[1].axhline(SON_EUC_Core_2016,linestyle='-.',linewidth=1.5,color='#082f99',label='_nolabel_')
ax[1].semilogx()
ax[1].set_ylim(ds.Z[depthui-1]-10,0)
ax[1].set_xlim(5e-6,3.0*10**-2)
ax[1].legend(loc='lower right',framealpha=1.0)
ax[1].set_xlabel('$m^2/s$')
ax[1].set_title('TPOSE SON')

plt.tight_layout()
image_str = 'EddyViscosity_Observations_InteranVar.png'
plt.savefig(image_str,format='png')
plt.close()