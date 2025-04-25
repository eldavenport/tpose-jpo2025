# Ellen Davenport April 2025
# This script compares TPOSE temperature to TAO for each 4 month assimilation window of 2012
import xarray as xr
from open_tpose import tpose2012_4month
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import TwoSlopeNorm
import cmocean.cm as cmo

warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 17

zMin = -250
zMax = 0

prefix = ['diag_state']

# open all 4 month windows from 2012 as a separate datasets
[dsJan, dsMar, dsMay, dsJul, dsSep] = tpose2012_4month(prefix)

for ds in [dsJan, dsMar, dsMay, dsJul, dsSep]:
    ds['XC'] = ds.XC.astype(float)
    ds['YC'] = ds.YC.astype(float)
    ds['Z'] = ds.Z.astype(float)
    ds['XG'] = ds.XG.astype(float)
    ds['YG'] = ds.YG.astype(float)
    ds['time'] = range(len(ds.time))

# ---------------------------------------------------------------TAO Data ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/home/edavenport/analysis/EqMixRemix_Processing/Python/TAO_TPOSE_comparisons/2012_TAO_data/t_TAO_140W_2012.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
T_TAO = dsTAO.T_20[:,:,0,0]
T_TAO = np.where(T_TAO > 50, np.nan, T_TAO)
T_TAO = xr.DataArray(T_TAO,coords=[dsTAO.time,depths],dims=['time','depth'])

depthli = np.argmin(np.abs(depths - zMax))
depthui = np.argmin(np.abs(depths - zMin)) + 1

# sample these locations from the TPOSE data
day_intervals = [0,29+31,31+30,31+30,31+31]
T6_140_diff = []
start = 0
for (ds, dsDt) in zip([dsJan, dsMar, dsMay, dsJul, dsSep],day_intervals):
    N = len(ds.time)
    start = start + dsDt # move start date by two months
    stop = start + N # go for as many days are in the run
    ds['time'] = range(start,stop)

    T_TAO_140 = T_TAO[start:stop,:]
    T6_140 = ds.THETA.interp(XC=[220.0],YC=[0.0],Z=T_TAO_140.depth,time=T_TAO_140.time,method='linear')

    temp = T6_140.values
    T6_140 = T_TAO_140.copy(deep=True)
    T6_140.values = temp[:,:,0,0]
    T6_140 = T6_140 + T_TAO_140 - T_TAO_140
    T_140_diff = T6_140 - T_TAO_140

    print(T_140_diff[:,depthli:depthui])

    T6_140_diff.append(T_140_diff)

# -----------------------------------------------------------Plotting -----------------------------------------------------------------------------------------

diff_levels= np.arange(-4.5,4.5,0.05)
labels = ['Jan-Apr','Mar-Jun','May-Aug','Jul-Oct','Sep-Dec']

print('plotting')
fig, ax = plt.subplots(figsize=(10,18),nrows=5)

for (T, axCurrent, label) in zip(T6_140_diff, ax, labels):

    (T[:,depthli:depthui].T).plot.contourf(ax=axCurrent,cmap=cmo.balance,levels=diff_levels,norm=TwoSlopeNorm(vmin=-4.5,vcenter=0,vmax=4.5),cbar_kwargs={'label':'$deg C$'})
    axCurrent.set_xlabel('Day in 2012')
    axCurrent.set_ylabel('Z (m)')
    title = '(TPOSE - TAO) ' + label
    axCurrent.set_title(title)

plt.tight_layout()
image_str = 'TAO_TPOSE6_4month_2012_Tdiff.png'
plt.savefig(image_str,format='png')
plt.close()
