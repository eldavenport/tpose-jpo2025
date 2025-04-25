# Ellen Davenport April 2025
# This script computes the seasonal and annual average heat budget terms for the TPOSE 2012-2013 state estimate

import xarray as xr
import xgcm
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sys
from open_tpose import tpose2012to2013
warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 17

# All diagnostics necessary to close the heat budget
prefix = ['diag_heat_budget','diag_state','diag_surf']

zMax = 0
lon = float(sys.argv[1])
lat = float(sys.argv[2])
zMin = float(sys.argv[3])

latMin = -0.1 + lat
latMax = 0.1 + lat

# Open TPOSE 2012 to 2013
ds = tpose2012to2013(prefix)

lats = ds.YC.data
lons = ds.XC.data
depths = ds.Z.data

latli = np.argmin(np.abs(lats - latMin))
latui = np.argmin(np.abs(lats - latMax)) + 1
lonidx = np.argmin(np.abs(lons - lon))
depthli = np.argmin(np.abs(depths - zMax))
depthui = np.argmin(np.abs(depths - zMin)) + 1

N = len(ds.time)
ds['time'] = range(0,N,1)

# ------------------------------------------------------------- heat budget -----------------------------------------------------------------------
grid = xgcm.Grid(ds, periodic=['X','Y']) # create grid object with xgcm
vol = ds.hFacC * ds.rA * ds.drF # cell volume

# compute heat budget terms, some of which need to be differenced and normalized by cell volume
# KPP
KPP_heat = grid.diff(ds.KPPg_TH,'Z',boundary='fill',fill_value=0)/vol

# Advection
ADVx_heat = -grid.diff(ds.ADVx_TH, 'X', boundary='extend')/vol
ADVy_heat = -grid.diff(ds.ADVy_TH, 'Y', boundary='extend')/vol
ADVz_heat = grid.diff(ds.ADVr_TH, 'Z', boundary='fill',fill_value=0)/vol
ADV_heat = ADVx_heat + ADVy_heat + ADVz_heat

# Diffusion
Diffx_diff = -grid.diff(ds.DFxE_TH, 'X', boundary='extend')/vol
Diffy_diff = -grid.diff(ds.DFyE_TH, 'Y', boundary='extend')/vol
Diffz_diff = (grid.diff(ds.DFrE_TH, 'Z', boundary='fill',fill_value=0) + grid.diff(ds.DFrI_TH, 'Z', boundary='fill',fill_value=0))/vol
Total_Diff_heat = Diffx_diff + Diffy_diff

# Combine KPP and vertical diffusion
KPP_heat = KPP_heat + Diffz_diff

rho_0 = 1035
Cp = 3994

# parameters for solar heating curve
R = 0.62
zeta1 = 0.6
zeta2 = 20

z1 = np.array(ds.Zl[0:-2].values)
z2 = np.array(ds.Zl[1:-1].values)
q1 = R*np.exp(z1/zeta1) + (1-R)*np.exp(z1/zeta2)
q2 = R*np.exp(z2/zeta1) + (1-R)*np.exp(z2/zeta2)

# compute the fraction of solar heating at each depth
swfrac = np.tile(q1-q2,(len(ds.XC),len(ds.YC),1)).transpose((2,1,0)) # X,Y,Z --> Z, Y, X

# Subtract oceQsw from TFLUX to get the non-solar component of the surface heat tendency
TFLUX_noSW =  (ds.TFLUX[:,latli:latui,lonidx].values-ds.oceQsw[:,latli:latui,lonidx].values)/(rho_0*Cp*ds.hFacC[0,latli:latui,lonidx].values*ds.drF[0].values)
# compute the solar heat tendency at all depths
Force_heat = (swfrac[np.newaxis,:,latli:latui,lonidx]*ds.oceQsw[:,latli:latui,lonidx].values[:,np.newaxis,:])/(rho_0*Cp*ds.hFacC[0:-2,latli:latui,lonidx].values[np.newaxis,:,:]*ds.drF[0:-2].values[np.newaxis,:,np.newaxis])

# put this into a DataArray
dummy = np.zeros((N,2,latui-latli))
Solar_heat = np.concatenate((Force_heat,dummy),axis=1)
tmp = KPP_heat[:,:,latli:latui,lonidx].copy(deep=True)
tmp.values = Solar_heat
tmp.name = 'Solar'
Solar_heat = tmp 

# combing non-solar and solar heat tendencies (solar piece has to be multiplied by swfrac)
dummy = np.zeros((N,2,latui-latli))
Force_heat = np.concatenate((Force_heat,dummy),axis=1)
Force_heat[:,0,:] += TFLUX_noSW
tmp = KPP_heat[:,:,latli:latui,lonidx].copy(deep=True)
tmp.values = Force_heat
tmp.name = 'Force_heat'
Force_heat = tmp 

# Add boundary heating to KPP tendency at the surface (these boundary conditions have to match)
tmp = Force_heat.copy(deep=True)
tmp.values = KPP_heat[:,:,latli:latui,lonidx].values
tmp[:,0,:] += TFLUX_noSW 
tmp.name = 'KPP_heat'
KPP_heat = tmp

# decorrelation scales -----------------------------------------------------------------------------------------------------------------------
decorr_scale_days = 10 # estimated decorrelation timescale for momentum budget terms at 75m depth
N_ind = (180)/decorr_scale_days # approximate number of days in 2 seasons (we have two estimates of each season, so ~180 days per season)
std_err_denom = np.sqrt(N_ind)

# lump the seasons together -----------------------------------------------------------------------------------------------------------------------
DJFli = 0
DJFui = DJFli + 31 + 31 + 28
MAMui = DJFui + 92
JJAui = MAMui + 92
SONui = JJAui + 92

#2013
DJFli13 = SONui 
DJFui13 = DJFli13 + 31 + 31 + 28
MAMui13 = DJFui13 + 31 + 30 + 31
JJAui13 = MAMui13 + 30 + 31 + 31
SONui13 = JJAui13 + 30 + 31 + 30

# Estimate budget terms for the desired lat/lon/season --------------------------------------------------------------------------------------------
# KPP term - concatenate seasons across the two years
KPP_heat_DJF = xr.concat([KPP_heat[DJFli:DJFui], KPP_heat[DJFli13:DJFui13]], dim='time')
KPP_heat_MAM = xr.concat([KPP_heat[DJFui:MAMui], KPP_heat[DJFui13:MAMui13]], dim='time')
KPP_heat_JJA = xr.concat([KPP_heat[MAMui:JJAui], KPP_heat[MAMui13:JJAui13]], dim='time')
KPP_heat_SON = xr.concat([KPP_heat[JJAui:SONui], KPP_heat[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_heat_KPP = KPP_heat_DJF[:,depthli:depthui].mean(dim='YC').mean(dim='time') 
MAM_heat_KPP = KPP_heat_MAM[:,depthli:depthui].mean(dim='YC').mean(dim='time') # now this is 1xdepth
JJA_heat_KPP = KPP_heat_JJA[:,depthli:depthui].mean(dim='YC').mean(dim='time') # now this is 1xdepth
SON_heat_KPP = KPP_heat_SON[:,depthli:depthui].mean(dim='YC').mean(dim='time') # now this is 1xdepth
Avg_heat_KPP = KPP_heat[:,depthli:depthui].mean(dim='YC').mean(dim='time') # now this is 1xdepth

DJF_KPP_std = KPP_heat_DJF[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom
MAM_KPP_std = KPP_heat_MAM[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom
JJA_KPP_std = KPP_heat_JJA[:,depthli:depthui].mean(dim='YC').std(dim='time') /std_err_denom
SON_KPP_std = KPP_heat_SON[:,depthli:depthui].mean(dim='YC').std(dim='time') /std_err_denom
Avg_KPP_std = KPP_heat[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom

# ADV term - concatenate seasons across the two years
ADV_heat_DJF = xr.concat([ADV_heat[DJFli:DJFui], ADV_heat[DJFli13:DJFui13]], dim='time')
ADV_heat_MAM = xr.concat([ADV_heat[DJFui:MAMui], ADV_heat[DJFui13:MAMui13]], dim='time')
ADV_heat_JJA = xr.concat([ADV_heat[MAMui:JJAui], ADV_heat[MAMui13:JJAui13]], dim='time')
ADV_heat_SON = xr.concat([ADV_heat[JJAui:SONui], ADV_heat[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_heat_ADV = ADV_heat_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') 
MAM_heat_ADV = ADV_heat_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
JJA_heat_ADV = ADV_heat_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
SON_heat_ADV = ADV_heat_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
Avg_heat_ADV = ADV_heat[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth

DJF_adv_std = ADV_heat_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
MAM_adv_std = ADV_heat_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
JJA_adv_std = ADV_heat_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
SON_adv_std = ADV_heat_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
Avg_adv_std = ADV_heat[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

# Solar term - concatenate seasons across the two years
Solar_heat_DJF = xr.concat([Solar_heat[DJFli:DJFui], Solar_heat[DJFli13:DJFui13]], dim='time')
Solar_heat_MAM = xr.concat([Solar_heat[DJFui:MAMui], Solar_heat[DJFui13:MAMui13]], dim='time')
Solar_heat_JJA = xr.concat([Solar_heat[MAMui:JJAui], Solar_heat[MAMui13:JJAui13]], dim='time')
Solar_heat_SON = xr.concat([Solar_heat[JJAui:SONui], Solar_heat[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_solar = Solar_heat_DJF[:,depthli:depthui].mean(dim='YC').mean(dim='time') 
MAM_solar = Solar_heat_MAM[:,depthli:depthui].mean(dim='time').mean(dim='YC') # now this is 1xdepth
JJA_solar = Solar_heat_JJA[:,depthli:depthui].mean(dim='time').mean(dim='YC') # now this is 1xdepth
SON_solar = Solar_heat_SON[:,depthli:depthui].mean(dim='time').mean(dim='YC') # now this is 1xdepth
Avg_solar = Solar_heat[:,depthli:depthui].mean(dim='time').mean(dim='YC') # now this is 1xdepth

DJF_std_solar = Solar_heat_DJF[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom
MAM_std_solar = Solar_heat_MAM[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom
JJA_std_solar = Solar_heat_JJA[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom
SON_std_solar = Solar_heat_SON[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom
Avg_std_solar = Solar_heat[:,depthli:depthui].mean(dim='YC').std(dim='time')/std_err_denom

ds['TOTTTEND'] = ds.TOTTTEND/86400
# Total tendency - concatenate seasons across the two years
TOT_DJF = xr.concat([ds.TOTTTEND[DJFli:DJFui], ds.TOTTTEND[DJFli13:DJFui13]], dim='time')
TOT_MAM = xr.concat([ds.TOTTTEND[DJFui:MAMui], ds.TOTTTEND[DJFui13:MAMui13]], dim='time')
TOT_JJA = xr.concat([ds.TOTTTEND[MAMui:JJAui], ds.TOTTTEND[MAMui13:JJAui13]], dim='time')
TOT_SON = xr.concat([ds.TOTTTEND[JJAui:SONui], ds.TOTTTEND[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_heatTOT = TOT_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') 
MAM_heatTOT = TOT_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
JJA_heatTOT = TOT_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
SON_heatTOT = TOT_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
Avg_heatTOT = ds.TOTTTEND[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth

DJF_std_TOT = TOT_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
MAM_std_TOT = TOT_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
JJA_std_TOT = TOT_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
SON_std_TOT = TOT_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
Avg_std_TOT = ds.TOTTTEND[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

# Estimate EUC core from Zonal Velocity
UVEL_DJF = xr.concat([ds.UVEL[DJFli:DJFui], ds.UVEL[DJFli13:DJFui13]], dim='time')
UVEL_MAM = xr.concat([ds.UVEL[DJFui:MAMui], ds.UVEL[DJFui13:MAMui13]], dim='time')
UVEL_JJA = xr.concat([ds.UVEL[MAMui:JJAui], ds.UVEL[MAMui13:JJAui13]], dim='time')
UVEL_SON = xr.concat([ds.UVEL[JJAui:SONui], ds.UVEL[JJAui13:SONui13]], dim='time')

DJF_EUC_Core = ds.Z[UVEL_DJF[:,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')
MAM_EUC_Core = ds.Z[UVEL_MAM[:,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')
JJA_EUC_Core = ds.Z[UVEL_JJA[:,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')
SON_EUC_Core = ds.Z[UVEL_SON[:,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')
Avg_EUC_Core = ds.Z[ds.UVEL[:,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time')

# ------------------------------------ PLOTTING -----------------------------------------------------------------

xmin = -6.75 * (10**-6)
xmax = 6.75 * (10**-6)
print('plotting')
depth_axis = ds.Z[depthli:depthui]

fig, ax = plt.subplots(figsize=(25,10),ncols=5)
ax[0].plot(DJF_heat_KPP,depth_axis,linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[0].fill_betweenx(depth_axis,DJF_heat_KPP-DJF_KPP_std,DJF_heat_KPP+DJF_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[0].plot(DJF_heat_ADV,depth_axis,linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[0].fill_betweenx(depth_axis,DJF_heat_ADV-DJF_adv_std,DJF_heat_ADV+DJF_adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[0].plot(DJF_solar,depth_axis,linewidth=3.0,color='y',label=r'$G_{solar}$')
ax[0].fill_betweenx(depth_axis,DJF_solar-DJF_std_solar,DJF_solar+DJF_std_solar,color='y',label='_nolegend_',alpha=0.25)
ax[0].plot(DJF_heatTOT,depth_axis,linewidth=3.0,color='k',label=r'$\partial T/\partial t$')
ax[0].fill_betweenx(depth_axis,DJF_heatTOT-DJF_std_TOT,DJF_heatTOT+DJF_std_TOT,color='k',label='_nolegend_',alpha=0.1)
ax[0].axhline(DJF_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[0].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[0].set_xlim(xmin,xmax)
ax[0].set_ylim(ds.Z[depthui-1],0)
ax[0].set_xlabel('$degC/s$')
ax[0].set_ylabel('Z (m)')
ax[0].set_title('DJF')

ax[1].plot(MAM_heat_KPP,depth_axis,linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[1].fill_betweenx(depth_axis,MAM_heat_KPP-MAM_KPP_std,MAM_heat_KPP+MAM_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[1].plot(MAM_heat_ADV,depth_axis,linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[1].fill_betweenx(depth_axis,MAM_heat_ADV-MAM_adv_std,MAM_heat_ADV+MAM_adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[1].plot(MAM_solar,depth_axis,linewidth=3.0,color='y',label=r'$G_{solar}$')
ax[1].fill_betweenx(depth_axis,MAM_solar-MAM_std_solar,MAM_solar+MAM_std_solar,color='y',label='_nolegend_',alpha=0.25)
ax[1].plot(MAM_heatTOT,depth_axis,linewidth=3.0,color='k',label=r'$\partial T/\partial t$')
ax[1].fill_betweenx(depth_axis,MAM_heatTOT-MAM_std_TOT,MAM_heatTOT+MAM_std_TOT,color='k',label='_nolegend_',alpha=0.1)
ax[1].axhline(MAM_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[1].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[1].set_xlim(xmin,xmax)
ax[1].set_ylim(ds.Z[depthui-1],0)
ax[1].set_xlabel('$degC/s$')
ax[1].set_ylabel('Z (m)')
ax[1].set_title('MAM')

ax[2].plot(JJA_heat_KPP,depth_axis,linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[2].fill_betweenx(depth_axis,JJA_heat_KPP-JJA_KPP_std,JJA_heat_KPP+JJA_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[2].plot(JJA_heat_ADV,depth_axis,linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[2].fill_betweenx(depth_axis,JJA_heat_ADV-JJA_adv_std,JJA_heat_ADV+JJA_adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[2].plot(JJA_solar,depth_axis,linewidth=3.0,color='y',label=r'$G_{solar}$')
ax[2].fill_betweenx(depth_axis,JJA_solar-JJA_std_solar,JJA_solar+JJA_std_solar,color='y',label='_nolegend_',alpha=0.1)
ax[2].plot(JJA_heatTOT,depth_axis,linewidth=3.0,color='k',label=r'$\partial T/\partial t$')
ax[2].fill_betweenx(depth_axis,JJA_heatTOT-JJA_std_TOT,JJA_heatTOT+JJA_std_TOT,color='k',label='_nolegend_',alpha=0.1)
ax[2].axhline(JJA_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[2].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[2].set_xlim(xmin,xmax)
ax[2].set_ylim(ds.Z[depthui-1],0)
ax[2].set_xlabel('$degC/s$')
ax[2].set_ylabel('Z (m)')
ax[2].set_title('JJA')

ax[3].plot(SON_heat_KPP,depth_axis,linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[3].fill_betweenx(depth_axis,SON_heat_KPP-SON_KPP_std,SON_heat_KPP+SON_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[3].plot(SON_heat_ADV,depth_axis,linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[3].fill_betweenx(depth_axis,SON_heat_ADV-SON_adv_std,SON_heat_ADV+SON_adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[3].plot(SON_solar,depth_axis,linewidth=3.0,color='y',label=r'$G_{solar}$')
ax[3].fill_betweenx(depth_axis,SON_solar-SON_std_solar,SON_solar+SON_std_solar,color='y',label='_nolegend_',alpha=0.1)
ax[3].plot(SON_heatTOT,depth_axis,linewidth=3.0,color='k',label=r'$\partial T/\partial t$')
ax[3].fill_betweenx(depth_axis,SON_heatTOT-SON_std_TOT,SON_heatTOT+SON_std_TOT,color='k',label='_nolegend_',alpha=0.1)
ax[3].axhline(SON_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[3].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[3].set_xlim(xmin,xmax)
ax[3].set_ylim(ds.Z[depthui-1],0)
ax[3].set_xlabel('$degC/s$')
ax[3].set_ylabel('Z (m)')
ax[3].set_title('SON')

ax[4].plot(Avg_heat_KPP,depth_axis,linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[4].fill_betweenx(depth_axis,Avg_heat_KPP-Avg_KPP_std,Avg_heat_KPP+Avg_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[4].plot(Avg_heat_ADV,depth_axis,linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[4].fill_betweenx(depth_axis,Avg_heat_ADV-Avg_adv_std,Avg_heat_ADV+Avg_adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[4].plot(Avg_solar,depth_axis,linewidth=3.0,color='y',label=r'$G_{solar}$')
ax[4].fill_betweenx(depth_axis,Avg_solar-Avg_std_solar,Avg_solar+Avg_std_solar,color='y',label='_nolegend_',alpha=0.1)
ax[4].plot(Avg_heatTOT,depth_axis,linewidth=3.0,color='k',label=r'$\partial T/\partial t$')
ax[4].fill_betweenx(depth_axis,Avg_heatTOT-Avg_std_TOT,Avg_heatTOT+Avg_std_TOT,color='k',label='_nolegend_',alpha=0.1)
ax[4].axhline(Avg_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[4].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[4].legend(loc='lower left')
ax[4].set_xlim(xmin,xmax)
ax[4].set_ylim(ds.Z[depthui-1],0)
ax[4].set_xlabel('$degC/s$')
ax[4].set_ylabel('Z (m)')
ax[4].set_title('Annual')

plt.tight_layout()
image_str = 'Heat_SeasonalAvg2012to2013_Std.png'
plt.savefig(image_str,format='png')
plt.close()