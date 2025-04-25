# This script includes functions to open tpose output directly from the MITgcm (not from netcdf)
# The output from the MITgcm is in a binary format, the number of iterations/model timestep varies between the runs, as does the location of the output
import xarray as xr
from xmitgcm import open_mdsdataset
import numpy as np

# Open all years of TPOSE as a single dataset
# Different runs are stored in different places and have different model timesteps
def tpose2012to2016(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 31+29 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'

    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    folder_months = ['mar2012/diags/','may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/']
    folder_days = np.array([61,61,62,112])
    itPerFile = [72, 72, 72, 48]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    #2013
    data_parent_dir = '/data/SO3/averdy/TPOSE6/'

    num_diags = 31+28 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2013/diags_daily/'

    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
    tpose_ds = xr.concat([tpose_ds,new],'time')

    #
    folder_months = ['mar2013/diags_daily/','may2013/diags_daily/','jul2013/diags_daily/','sep2013/diags_daily/','nov2013/diags_daily/']
    folder_days = np.array([61,61,62,61,51])
    itPerFile = np.array([72,72,72,72,72])

    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    # 2014-2016
    num_diags = 31+28 + offset #
    itPerFile = 48 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2014/diags_daily/'

    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
    tpose_ds = xr.concat([tpose_ds,new],'time')

    folder_months = ['mar2014/diags_daily/','may2014/diags_daily/','jul2014/diags_daily/','sep2014/diags_daily/','nov2014/diags_daily',
                    'jan2015/diags_daily/','mar2015/diags_daily/','may2015/diags_daily/','jul2015/diags_daily/','sep2015/diags_daily/','nov2015/diags_daily',
                    'jan2016/diags_daily/','mar2016/diags_daily/','may2016/diags_daily/','jul2016/diags_daily/','sep2016/diags_daily/','nov2016/diags_daily']
    folder_days = np.array([61,61,62,61,61,  #2014
                            31+28,61,61,62,61,61, # 2015
                            31+29,61,61,62,61,51,]) # 2016 (leap year)
    itPerFile = np.array([48,72,72,72,72,
                          72,48,72,72,72,72,
                          72,72,72,72,72,72])
    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    return tpose_ds

# Open KPP diagnostics from all years of TPOSE as a single dataset, excludes JF 2012
def tpose2012to2016_kpp(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts at the beginning of the model run)

    num_diags = 61 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'mar2012/diags/'

    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    folder_months = ['may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/',]
    folder_days = np.array([61,62,112])
    itPerFile = [72, 72, 48]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    #2013
    data_parent_dir = '/data/SO3/averdy/TPOSE6/'

    num_diags = 31+28 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2013/diags_daily/'

    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
    tpose_ds = xr.concat([tpose_ds,new],'time')

    # nov 2013 doesn't exist yet, once it does then all months will be treated the same pretty much
    folder_months = ['mar2013/diags_daily/','may2013/diags_daily/','jul2013/diags_daily/','sep2013/diags_daily/','nov2013/diags_daily/']
    folder_days = np.array([61,61,62,61,51])
    itPerFile = np.array([72,72,72,72,72])

    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    # 2014-2016
    num_diags = 31+28 + offset #
    itPerFile = 48 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2014/diags_daily/'

    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
    tpose_ds = xr.concat([tpose_ds,new],'time')

    folder_months = ['mar2014/diags_daily/','may2014/diags_daily/','jul2014/diags_daily/','sep2014/diags_daily/','nov2014/diags_daily',
                    'jan2015/diags_daily/','mar2015/diags_daily/','may2015/diags_daily/','jul2015/diags_daily/','sep2015/diags_daily/','nov2015/diags_daily',
                    'jan2016/diags_daily/','mar2016/diags_daily/','may2016/diags_daily/','jul2016/diags_daily/','sep2016/diags_daily/','nov2016/diags_daily']
    folder_days = np.array([61,61,62,61,61,  #2014
                            31+28,61,61,62,61,61, # 2015
                            31+29,61,61,62,61,51,]) # 2016 (leap year)
    itPerFile = np.array([48,72,72,72,72,
                          72,48,72,72,72,72,
                          72,72,72,72,72,72])
    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    print('Days in 2012-2016: (should be 1767)')
    print(len(tpose_ds.time))
    return tpose_ds

# Open only 2012-2013 TPOSE as a single dataset
def tpose2012to2013(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 31+29 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'

    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    folder_months = ['mar2012/diags/','may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/',]
    folder_days = np.array([61,61,62,112])
    itPerFile = [72, 72, 72, 48]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    #2013-2014
    data_parent_dir = '/data/SO3/averdy/TPOSE6/'

    num_diags = 31+28 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2013/diags_daily/'

    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
    tpose_ds = xr.concat([tpose_ds,new],'time')

    folder_months = ['mar2013/diags_daily/','may2013/diags_daily/','jul2013/diags_daily/','sep2013/diags_daily/','nov2013/diags_daily/']
    folder_days = np.array([61,61,62,61,51])
    itPerFile = np.array([72,72,72,72,72])

    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    print('Days in 2012-2013: (should be 731)')
    print(len(tpose_ds.time))
    return tpose_ds

# Open only 2012
def tpose2012(prefix):

    print('opening 2012')
    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 31+29 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'

    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':125,'XC':125,'YG':25,'YC':25,'Zl':1,'Z':1})

    folder_months = ['mar2012/diags/','may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/']
    folder_days = np.array([61,61,62,112])
    itPerFile = [72, 72, 72, 48]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':125,'XC':125,'YG':25,'YC':25,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    print('Days in 2012: (should be 366)')
    print(len(tpose_ds.time))
    return tpose_ds

# Open iteration 0 of TPOSE 2012 
def tpose2012_iter0(prefix):

    print('opening 2012')
    #2012
    data_parent_dir  = '/data/SO3/averdy/TPOSE6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 31+29 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags_iter0/'

    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    folder_months = ['mar2012/diags_iter0/','may2012/diags_iter0/','jul2012/diags_iter0/','sep2012/diags_iter0/','nov2012/diags_iter0/']
    folder_days = np.array([61,61,62,61,51])
    itPerFile = [72, 72, 72, 72, 72]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    print('Days in 2012: (should be 366)')
    print(len(tpose_ds.time))
    return tpose_ds

# Open hourly diagnostics from October 2012
def tposeOct2012_hourly(prefix):

    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    itPerFile = 2 # 1 day
    startDay = 1442
    endDay = 2880
    intervals = range(startDay,endDay+itPerFile,itPerFile)
    data_dir = data_parent_dir + 'sep2012/diags_iter7_hourly/'

    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    return tpose_ds

# Open all four month assimilation windows from 2012 and return them as a tuple of datasets
def tpose2012_4month(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    num_diags = 31+29+30+30  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'

    dsJan = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    num_diags = 122  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'mar2012/diags/'

    dsMar = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    num_diags = 123  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'may2012/diags/'

    dsMay = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    num_diags = 123  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jul2012/diags/'

    dsJul = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})

    num_diags = 121  #
    itPerFile = 48 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'sep2012/diags_iter7_daily/'

    dsSep = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix).chunk({'time':25,'XG':250,'XC':250,'YG':50,'YC':50,'Zl':1,'Z':1})


    return dsJan, dsMar, dsMay, dsJul, dsSep
