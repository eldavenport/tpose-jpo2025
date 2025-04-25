# Ellen Davenport April 2025
# This script computes the seasonal and annual average momentum budget terms for the TPOSE 2012-2013 state estimate

import sys
import xarray as xr
import xgcm
import numpy as np
import warnings
import matplotlib.pyplot as plt
from open_tpose import tpose2012to2013
warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 17

# All diagnostics necessary to close the momentum budget
prefix = ['diag_mom_u','diag_state','diag_surf']
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
zMax = 0
depthli = np.argmin(np.abs(depths - zMax))
depthui = np.argmin(np.abs(depths - zMin)) + 1

N = len(ds.time)
ds['time'] = range(0,N,1)

# ------------------------------------------------------------- momentum budget -----------------------------------------------------------------------
grid = xgcm.Grid(ds, periodic=['X','Y']) # create grid object with xgcm
vol = (ds.rAw*ds.hFacW*ds.drF) # cell volume 

# KPP term - needs to be differenced and normalized by cell volume
KPP_momU = grid.diff(ds.VISrI_Um,'Z',boundary='fill',fill_value=0)/vol
KPP_momU = KPP_momU.where(ds.hFacW.values > 0,0) # set any nan fluxes to 0
KPP_momU = KPP_momU + ds.Um_Ext # Combine external forcing with KPP tendency (Um_Ext is 0 everywhere except the surface)

# Remove coriolis from advection 
ds['Um_Advec'] = ds.Um_Advec - ds.Um_Cori

# decorrelation scales -----------------------------------------------------------------------------------------------------------------------
decorr_scale_days = 5 # estimated decorrelation timescale for momentum budget terms at 75m depth
N_ind = (180)/decorr_scale_days # approximate number of days in 2 seasons (we have two of each season)
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

ds['TOTUTEND'] = ds.TOTUTEND/86400 # divide by seconds in a day
# Total tendency - concatenate seasons across the two years
TOT_DJF = xr.concat([ds.TOTUTEND[DJFli:DJFui], ds.TOTUTEND[DJFli13:DJFui13]], dim='time')
TOT_MAM = xr.concat([ds.TOTUTEND[DJFui:MAMui], ds.TOTUTEND[DJFui13:MAMui13]], dim='time')
TOT_JJA = xr.concat([ds.TOTUTEND[MAMui:JJAui], ds.TOTUTEND[MAMui13:JJAui13]], dim='time')
TOT_SON = xr.concat([ds.TOTUTEND[JJAui:SONui], ds.TOTUTEND[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_momTOT = TOT_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') 
MAM_momTOT = TOT_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
JJA_momTOT = TOT_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
SON_momTOT = TOT_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth
Avg_momTOT = ds.TOTUTEND[:,depthli:depthui,latli:latui,lonidx].mean(dim='time').mean(dim='YC') # now this is 1xdepth

DJF_TOT_std = TOT_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
MAM_TOT_std = TOT_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
JJA_TOT_std = TOT_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
SON_TOT_std = TOT_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
Avg_TOT_std = ds.TOTUTEND[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

# KPP term - concatenate seasons across the two years
KPP_DJF = xr.concat([KPP_momU[DJFli:DJFui], KPP_momU[DJFli13:DJFui13]], dim='time')
KPP_MAM = xr.concat([KPP_momU[DJFui:MAMui], KPP_momU[DJFui13:MAMui13]], dim='time')
KPP_JJA = xr.concat([KPP_momU[MAMui:JJAui], KPP_momU[MAMui13:JJAui13]], dim='time')
KPP_SON = xr.concat([KPP_momU[JJAui:SONui], KPP_momU[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_momU_KPP = KPP_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')
MAM_momU_KPP = KPP_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') # now this is 1xdepth
JJA_momU_KPP = KPP_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') # now this is 1xdepth
SON_momU_KPP = KPP_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') # now this is 1xdepth
Avg_momU_KPP = KPP_momU[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time') # now this is 1xdepth

DJF_KPP_std = KPP_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
MAM_KPP_std = KPP_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
JJA_KPP_std = KPP_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
SON_KPP_std = KPP_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
Avg_KPP_std = KPP_momU[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

# ADV term - concatenate seasons across the two years
ADV_DJF = xr.concat([ds.Um_Advec[DJFli:DJFui], ds.Um_Advec[DJFli13:DJFui13]], dim='time')
ADV_MAM = xr.concat([ds.Um_Advec[DJFui:MAMui], ds.Um_Advec[DJFui13:MAMui13]], dim='time')
ADV_JJA = xr.concat([ds.Um_Advec[MAMui:JJAui], ds.Um_Advec[MAMui13:JJAui13]], dim='time')
ADV_SON = xr.concat([ds.Um_Advec[JJAui:SONui], ds.Um_Advec[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_momAdv = (ADV_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) 
MAM_momAdv = (ADV_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth
JJA_momAdv = (ADV_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth
SON_momAdv = (ADV_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth
Avg_momAdv = (ds.Um_Advec[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth

DJF_Adv_std = ADV_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
MAM_Adv_std = ADV_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
JJA_Adv_std = ADV_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
SON_Adv_std = ADV_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
Avg_Adv_std = ds.Um_Advec[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

# pressure gradient - concatenate seasons across the two years
PRESS_DJF = xr.concat([ds.Um_dPhiX[DJFli:DJFui], ds.Um_dPhiX[DJFli13:DJFui13]], dim='time')
PRESS_MAM = xr.concat([ds.Um_dPhiX[DJFui:MAMui], ds.Um_dPhiX[DJFui13:MAMui13]], dim='time')
PRESS_JJA = xr.concat([ds.Um_dPhiX[MAMui:JJAui], ds.Um_dPhiX[MAMui13:JJAui13]], dim='time')
PRESS_SON = xr.concat([ds.Um_dPhiX[JJAui:SONui], ds.Um_dPhiX[JJAui13:SONui13]], dim='time')

# Take mean and std. dev. (convert std. dev. to std. error)
DJF_momDphiX = (PRESS_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) 
MAM_momDphiX = (PRESS_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth
JJA_momDphiX = (PRESS_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth
SON_momDphiX = (PRESS_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth
Avg_momDphiX = (ds.Um_dPhiX[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time')) # now this is 1xdepth

DJF_DphiX_std = PRESS_DJF[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
MAM_DphiX_std = PRESS_MAM[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
JJA_DphiX_std = PRESS_JJA[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
SON_DphiX_std = PRESS_SON[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom
Avg_DphiX_std = ds.Um_dPhiX[:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')/std_err_denom

# plot -----------------------------------------------------------------------------------------------------------------------

xmin = -1*(10**-6)
xmax = 1*(10**-6)

depth_axis = ds.Z[depthli:depthui]
fig, ax = plt.subplots(figsize=(25,10),ncols=5)
ax[0].plot(DJF_momU_KPP,depth_axis,linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[0].fill_betweenx(depth_axis,DJF_momU_KPP-DJF_KPP_std,DJF_momU_KPP+DJF_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[0].plot(DJF_momAdv,ds.Z[depthli:depthui],linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[0].fill_betweenx(depth_axis,DJF_momAdv-DJF_Adv_std,DJF_momAdv+DJF_Adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[0].plot(DJF_momDphiX,ds.Z[depthli:depthui],linewidth=3.0,color='tab:olive',label=r'$G_{press}$')
ax[0].fill_betweenx(depth_axis,DJF_momDphiX-DJF_DphiX_std,DJF_momDphiX+DJF_DphiX_std,color='tab:olive',label='_nolegend_',alpha=0.25)
ax[0].plot(DJF_momTOT,ds.Z[depthli:depthui],linewidth=3.0,color='k',label=r'$\partial u/\partial t$')
ax[0].fill_betweenx(depth_axis,DJF_momTOT-DJF_TOT_std,DJF_momTOT+DJF_TOT_std,color='k',label='_nolegend_',alpha=0.1)
ax[0].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[0].axhline(DJF_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[0].set_ylim(ds.Z[depthui-1],0)
ax[0].set_xlim(xmin,xmax)
ax[0].set_title('Tendency ($m/s^2$)')
ax[0].set_xlabel('$m/s^2$')
ax[0].set_ylabel('Z (m)')
ax[0].set_title('DJF')

ax[1].plot(MAM_momU_KPP,ds.Z[depthli:depthui],linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[1].fill_betweenx(depth_axis,MAM_momU_KPP-MAM_KPP_std,MAM_momU_KPP+MAM_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[1].plot(MAM_momAdv,ds.Z[depthli:depthui],linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[1].fill_betweenx(depth_axis,MAM_momAdv-MAM_Adv_std,MAM_momAdv+MAM_Adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[1].plot(MAM_momDphiX,ds.Z[depthli:depthui],linewidth=3.0,color='tab:olive',label=r'$G_{press}$')
ax[1].fill_betweenx(depth_axis,MAM_momDphiX-MAM_DphiX_std,MAM_momDphiX+MAM_DphiX_std,color='tab:olive',label='_nolegend_',alpha=0.25)
ax[1].plot(MAM_momTOT,ds.Z[depthli:depthui],linewidth=3.0,color='k',label=r'$\partial u/\partial t$')
ax[1].fill_betweenx(depth_axis,MAM_momTOT-MAM_TOT_std,MAM_momTOT+MAM_TOT_std,color='k',label='_nolegend_',alpha=0.1)
ax[1].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[1].axhline(MAM_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[1].set_ylim(ds.Z[depthui-1],0)
ax[1].set_xlim(xmin,xmax)
ax[1].set_title('Tendency ($m/s^2$)')
ax[1].set_xlabel('$m/s^2$')
ax[1].set_ylabel('Z (m)')
ax[1].set_title('MAM')

ax[2].plot(JJA_momU_KPP,ds.Z[depthli:depthui],linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[2].fill_betweenx(depth_axis,JJA_momU_KPP-JJA_KPP_std,JJA_momU_KPP+JJA_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[2].plot(JJA_momAdv,ds.Z[depthli:depthui],linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[2].fill_betweenx(depth_axis,JJA_momAdv-JJA_Adv_std,JJA_momAdv+JJA_Adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[2].plot(JJA_momDphiX,ds.Z[depthli:depthui],linewidth=3.0,color='tab:olive',label=r'$G_{press}$')
ax[2].fill_betweenx(depth_axis,JJA_momDphiX-JJA_DphiX_std,JJA_momDphiX+JJA_DphiX_std,color='tab:olive',label='_nolegend_',alpha=0.25)
ax[2].plot(JJA_momTOT,ds.Z[depthli:depthui],linewidth=3.0,color='k',label=r'$\partial u/\partial t$')
ax[2].fill_betweenx(depth_axis,JJA_momTOT-JJA_TOT_std,JJA_momTOT+JJA_TOT_std,color='k',label='_nolegend_',alpha=0.1)
ax[2].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[2].axhline(JJA_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[2].set_ylim(ds.Z[depthui-1],0)
ax[2].set_xlim(xmin,xmax)
ax[2].set_title('Tendency ($m/s^2$)')
ax[2].set_xlabel('$m/s^2$')
ax[2].set_ylabel('Z (m)')
ax[2].set_title('JJA')

ax[3].plot(SON_momU_KPP,ds.Z[depthli:depthui],linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[3].fill_betweenx(depth_axis,SON_momU_KPP-SON_KPP_std,SON_momU_KPP+SON_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[3].plot(SON_momAdv,ds.Z[depthli:depthui],linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[3].fill_betweenx(depth_axis,SON_momAdv-SON_Adv_std,SON_momAdv+SON_Adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[3].plot(SON_momDphiX,ds.Z[depthli:depthui],linewidth=3.0,color='tab:olive',label=r'$G_{press}$')
ax[3].fill_betweenx(depth_axis,SON_momDphiX-SON_DphiX_std,SON_momDphiX+SON_DphiX_std,color='tab:olive',label='_nolegend_',alpha=0.25)
ax[3].plot(SON_momTOT,ds.Z[depthli:depthui],linewidth=3.0,color='k',label=r'$\partial u/\partial t$')
ax[3].fill_betweenx(depth_axis,SON_momTOT-SON_TOT_std,SON_momTOT+SON_TOT_std,color='k',label='_nolegend_',alpha=0.1)
ax[3].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[3].axhline(SON_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[3].set_ylim(ds.Z[depthui-1],0)
ax[3].set_xlim(xmin,xmax)
ax[3].set_title('Tendency ($m/s^2$)')
ax[3].set_xlabel('$m/s^2$')
ax[3].set_ylabel('Z (m)')
ax[3].set_title('SON')

ax[4].plot(Avg_momU_KPP,ds.Z[depthli:depthui],linewidth=3.0,color='tab:red',label=r'$G_{mix}$')
ax[4].fill_betweenx(depth_axis,Avg_momU_KPP-Avg_KPP_std,Avg_momU_KPP+Avg_KPP_std,color='tab:red',label='_nolegend_',alpha=0.25)
ax[4].plot(Avg_momAdv,ds.Z[depthli:depthui],linewidth=3.0,color='tab:blue',label=r'$G_{adv}$')
ax[4].fill_betweenx(depth_axis,Avg_momAdv-Avg_Adv_std,Avg_momAdv+Avg_Adv_std,color='tab:blue',label='_nolegend_',alpha=0.25)
ax[4].plot(Avg_momDphiX,ds.Z[depthli:depthui],linewidth=3.0,color='tab:olive',label=r'$G_{press}$')
ax[4].fill_betweenx(depth_axis,Avg_momDphiX-Avg_DphiX_std,Avg_momDphiX+Avg_DphiX_std,color='tab:olive',label='_nolegend_',alpha=0.25)
ax[4].plot(Avg_momTOT,ds.Z[depthli:depthui],linewidth=3.0,color='k',label=r'$\partial u/\partial t$')
ax[4].fill_betweenx(depth_axis,Avg_momTOT-Avg_TOT_std,Avg_momTOT+Avg_TOT_std,color='k',label='_nolegend_',alpha=0.1)
ax[4].axvline(0,linestyle='--',linewidth=0.3,color='k')
ax[4].axhline(Avg_EUC_Core,linestyle='--',linewidth=1.5,color='tab:gray',label='EUC Core')
ax[4].set_ylim(ds.Z[depthui-1],0)
ax[4].set_xlim(xmin,xmax)
ax[4].legend(loc='lower left',framealpha=1.0)
ax[4].set_title('Tendency ($m/s^2$)')
ax[4].set_xlabel('$m/s^2$')
ax[4].set_ylabel('Z (m)')
ax[4].set_title('Annual')

plt.tight_layout()
image_str = 'ZonalMom_SeasonalAvg2012to2013_Std.png'
plt.savefig(image_str,format='png')
plt.close()