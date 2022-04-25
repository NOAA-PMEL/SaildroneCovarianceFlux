"""Functions to read and write Saildrone data.

"""

#------------------------------------------------------------------------
#---- Analysis tools:
import numpy as np
import xarray as xr
import datetime
import cf_units
#---- System tools:
import sys
import os
import warnings
warnings.filterwarnings(action='ignore', module='xarray',
                        category=FutureWarning)
#---- My code:
import config
#------------------------------------------------------------------------

def load_high_freq_multifile(sdy,sdid):
    """Returns dataset containing all high frequency data for a single Saildrone
    
    sdy [int or str] -- year of mission (2017, etc)
    sdid [int or str] -- Saildrone ID (1005, etc.)
    """
    filenames = high_freq_filenames(sdy,sdid)
    # Get first dataset.
    ds_all = load_swap_time(filenames[0])
    print(filenames[0])
    # Loop over and append other files in list.
    for fn in filenames[1:]:
        print(fn)
        ds_next = load_swap_time(fn)
        # Check for overlap with ds_all
        # This assumes that 'filenames' is in chronological order.
        old_max = ds_all.time.max().data
        new_min = ds_next.time.min().data
        if old_max >= new_min:
            # Join the data if it is well-behaved...
            if ds_all.isel(time=(ds_all.time >= new_min)).\
               equals(ds_next.isel(time=(ds_next.time <= old_max))):
                ds_all = xr.concat([ds_all,
                                    ds_next.isel(time=(ds_next.time > old_max))],
                                   dim='time')
            else:
                print(ds_all.isel(time=(ds_all.time >= new_min)))
                print(ds_next.isel(time=(ds_next.time <= old_max)))
                sys.exit('Non-identical overlapping data: cannot continue joining files.')
        # ...and join it if there is no overlap.
        else:
            ds_all = xr.concat([ds_all, ds_next], dim='time')
    #
    return ds_all

def load_swap_time(fn):
    """Loads a single Saildrone data file and makes time a dimension.
    
    Input:
    fn [str] -- data filename to load

    Also 
    Drops the 'trajectory' variable 
    Gets rid of unlabelled missing values that seem to occur quite often.
    """
    ds = xr.open_dataset(fn)
    ds = ds.swap_dims({'row':'time'})
    if 'trajectory' in ds.keys():
        ds = ds.drop('trajectory')
    # Get rid of "-1e34" missing values. 
    for v in list(ds.keys()):
        ds[v] = ds[v].where(ds[v] > -1.e34, other=np.nan)
    #
    return ds

def high_freq_filenames(sdy,sdid):
    """Returns list of high frequency files for a single Saildrone.
    
    sdy [int or str] -- year of mission (2017, etc)
    sdid [int or str] -- Saildrone ID (1005, etc.)
    """
    data_dir = config.data_dir
    master_dict = {
        '2017':{
            '1005':[
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20170902_20170930.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20171001_20171031.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20171101_20171130.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20171201_20171231.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20180101_20180131.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20180201_20180228.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20180301_20180331.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1005_uvw_20180401_20180505.nc'],
            '1006':[
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20170902_20170930.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20171001_20171031.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20171101_20171130.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20171201_20171231.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20180101_20180131.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20180201_20180228.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20180301_20180331.nc',
                data_dir
                + '10Hz/tpos_2017_hf_10hz_1006_uvw_20180401_20180505.nc']
        },
        '2018':{
            '1005':[
                data_dir
                + '10Hz/tpos_2018_hf_10hz_1005_uvw_20181003_20181031.nc',
                data_dir
                + '10Hz/tpos_2018_hf_10hz_1005_uvw_20181101_20181130.nc',
                data_dir
                + '10Hz/tpos_2018_hf_10hz_1005_uvw_20181201_20181231.nc',
                data_dir
                + '10Hz/tpos_2018_hf_10hz_1005_uvw_20190101_20190131.nc'],
            '1006':[
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1006_uvw_20181003_20181031.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1006_uvw_20181101_20181130.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1006_uvw_20181201_20181231.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1006_uvw_20190101_20190131.nc'
                ],
            '1029':[
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1029_uvw_20181003_20181031.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1029_uvw_20181101_20181130.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1029_uvw_20181201_20181231.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1029_uvw_20190101_20190131.nc'],
            '1030':[
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1030_uvw_20181003_20181031.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1030_uvw_20181101_20181130.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1030_uvw_20181201_20181231.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1030_uvw_20190101_20190131.nc',
                data_dir + '10Hz/' +
                'tpos_2018_hf_10hz_1030_uvw_20190201_20190305.nc']
        },
        '2019':{
            '1066':[
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1066_uvw_20190609_20190701.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1066_uvw_20190701_20190801.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1066_uvw_20190801_20190901.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1066_uvw_20190901_20191001.nc'],
            '1067':[
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20190618_20190701.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20190701_20190801.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20190801_20190901.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20190901_20191001.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20191001_20191101.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20191101_20191201.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1067_uvw_20191201_20200101.nc'],
            '1068':[
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20190608_20190701.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20190701_20190801.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20190801_20190901.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20190901_20191001.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20191001_20191101.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20191101_20191201.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1068_uvw_20191201_20200101.nc'],
            '1069':[
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20190608_20190701.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20190701_20190801.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20190801_20190901.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20190901_20191001.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20191001_20191101.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20191101_20191201.nc',
                data_dir
                + '10Hz/tpos_2019_hf_20hz_1069_uvw_20191201_20191227.nc']
        }
    }
    result = master_dict[str(sdy)][str(sdid)]
    #
    return result


def load_L1_data(yr, ID, downsample=False):
    """Loads a QC'ed dataset for year yr and Saildrone ID.

    Inputs yr and ID should be strings. downsample is boolean
    indicating whether to use the 20 Hz (False) or 10 Hz (True)
    data for 2019.
    Output is an xarray dataset (with QC flags).
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_hf_10hz_1005_uvw_20170904_20180430_QC_L1_5min',
            '1006':'tpos_2017_hf_10hz_1006_uvw_20170904_20180505_QC_L1_5min'},
        '2018':{
            '1005':'tpos_2018_hf_10hz_1005_uvw_20181006_20181106_QC_L1_5min',
            '1006':'tpos_2018_hf_10hz_1006_uvw_20181006_20190114_QC_L1_5min',
            '1029':'tpos_2018_hf_10hz_1029_uvw_20181009_20190108_QC_L1_5min',
            '1030':'tpos_2018_hf_10hz_1030_uvw_20181009_20190303_QC_L1_5min'},
        '2019':{
            '1066':'tpos_2019_hf_20hz_1066_uvw_20190626_20190916_QC_L1_5min',
            '1067':'tpos_2019_hf_20hz_1067_uvw_20190626_20191217_QC_L1_5min',
            '1068':'tpos_2019_hf_20hz_1068_uvw_20190625_20191212_QC_L1_5min',
            '1069':'tpos_2019_hf_20hz_1069_uvw_20190626_20191212_QC_L1_5min'}
    }
    filenames_ds = {
        '2019':{
            '1066':'tpos_2019_hf_10hz_1066_uvw_20190626_20190916_QC_L1',
            '1067':'tpos_2019_hf_10hz_1067_uvw_20190626_20191217_QC_L1',
            '1068':'tpos_2019_hf_10hz_1068_uvw_20190625_20191212_QC_L1',
            '1069':'tpos_2019_hf_10hz_1069_uvw_20190626_20191212_QC_L1'}
    }
    data_dir = config.data_dir + '10Hz/'
    file_suffix = '.nc'
    if downsample:
        fn = data_dir + filenames_ds[yr][ID] + file_suffix
    else:
        fn = data_dir + filenames[yr][ID] + file_suffix
    #
    print('Loading file:')
    print(fn)
    #
    result = xr.open_dataset(fn)
    return result


def load_1min_data(yr, ID):
    """Loads 1-minute data for year yr and Saildrone ID.

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_1min_1005_allVars_20170901_20180506',
            '1006':'tpos_2017_1min_1006_allVars_20170901_20180515'},
        '2018':{
            '1005':'tpos_2018_1min_1005_allVars_20180914_20181223',
            '1006':'tpos_2018_1min_1006_allVars_20180914_20190127',
            '1029':'tpos_2018_1min_1029_allVars_20180907_20190924',
            '1030':'tpos_2018_1min_1030_allVars_20180907_20190306'},
        '2019':{
            '1066':'tpos_2019_1min_1066_allVars_20190608_20200107',
            '1067':'tpos_2019_1min_1067_allVars_20190608_20200107',
            '1068':'tpos_2019_1min_1068_allVars_20190608_20200107',
            '1069':'tpos_2019_1min_1069_allVars_20190608_20200107'}
    }
    data_dir = config.data_dir + '1min/'
    file_suffix = '.nc'
    print('Loading file:')
    print(data_dir + filenames[yr][ID] + file_suffix)
    #
    result = load_swap_time(data_dir + filenames[yr][ID] + file_suffix)
    return result


def load_L2_data(yr, ID, downsample=False, av_period=30):
    """Loads a QC'ed dataset for year yr and Saildrone ID.

    Inputs yr and ID should be strings. downsample is boolean
    indicating whether to use the 20 Hz (downsample=False) or 
    10 Hz (downsample=True) data for 2019.
    av_period is an integer giving the flux averaging period (in minutes).

    Output is an xarray dataset (with QC flags).
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_hf_10hz_1005_uvw_20170904_20180430_',
            '1006':'tpos_2017_hf_10hz_1006_uvw_20170904_20180505_'},
        '2018':{
            '1005':''},
        '2019':{
            '1066':'tpos_2019_hf_20hz_1066_uvw_20190626_20190916_',
            '1067':'tpos_2019_hf_20hz_1067_uvw_20190626_20191217_',
            '1068':'tpos_2019_hf_20hz_1068_uvw_20190625_20191212_',
            '1069':'tpos_2019_hf_20hz_1069_uvw_20190626_20191212_'}
    }
    filenames_ds = {
        '2019':{
            '1066':'tpos_2019_hf_10hz_1066_uvw_20190626_20190916_',
            '1067':'tpos_2019_hf_10hz_1067_uvw_20190626_20191217_',
            '1068':'tpos_2019_hf_10hz_1068_uvw_20190625_20191212_',
            '1069':'tpos_2019_hf_10hz_1069_uvw_20190626_20191212_'}
    }
    data_dir = config.data_dir + '10Hz/'
    file_suffix = 'min_fluxes_L2.nc'
    if downsample:
        fn = data_dir + filenames_ds[str(yr)][str(ID)] + str(av_period) + file_suffix
    else:
        fn = data_dir + filenames[str(yr)][str(ID)] + str(av_period) + file_suffix
    #
    print('Loading file:')
    print(fn)
    #
    result = xr.open_dataset(fn)
    return result


def load_L2_relative_data(yr, ID, downsample=False, av_period=10):
    """Loads a QC'ed dataset for year yr and Saildrone ID.

    Inputs yr and ID should be strings. downsample is boolean
    indicating whether to use the 20 Hz (downsample=False) or 
    10 Hz (downsample=True) data for 2019.
    av_period is an integer giving the flux averaging period (in minutes).

    Output is an xarray dataset (with QC flags).
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_hf_10hz_1005_uvw_20170904_20180430_',
            '1006':'tpos_2017_hf_10hz_1006_uvw_20170904_20180505_'},
        '2018':{
            '1005':''},
        '2019':{
            '1066':'tpos_2019_hf_20hz_1066_uvw_20190626_20190916_',
            '1067':'tpos_2019_hf_20hz_1067_uvw_20190626_20191217_',
            '1068':'tpos_2019_hf_20hz_1068_uvw_20190625_20191212_',
            '1069':'tpos_2019_hf_20hz_1069_uvw_20190626_20191212_'}
    }
    filenames_ds = {
        '2019':{
            '1066':'tpos_2019_hf_10hz_1066_uvw_20190626_20190916_',
            '1067':'tpos_2019_hf_10hz_1067_uvw_20190626_20191217_',
            '1068':'tpos_2019_hf_10hz_1068_uvw_20190625_20191212_',
            '1069':'tpos_2019_hf_10hz_1069_uvw_20190626_20191212_'}
    }
    data_dir = config.data_dir + '10Hz/'
    file_suffix = 'min_fluxes_L2_relative_wind_direction.nc'
    if downsample:
        fn = data_dir + filenames_ds[str(yr)][str(ID)] + str(av_period) + file_suffix
    else:
        fn = data_dir + filenames[str(yr)][str(ID)] + str(av_period) + file_suffix
    #
    print('Loading file:')
    print(fn)
    #
    result = xr.open_dataset(fn)
    return result


def load_bulk_data(yr, ID):
    """Loads dataset with bulk fluxes and associated variables.
    
    Inputs yr and ID should be strings where:
        yr is the mission year, in {2017, 2018, 2019};
        ID is the Saildrone ID, in {1005, 1006, etc.} .

    Output is an xarray dataset with just one dimension, time, which 
    has 10-minute time steps.
    """
    fn = config.data_dir + 'derived/' + \
        'TPOS' + str(yr) + 'Saildrone' + str(ID) + 'curf.cdf'
    t_var_name = 'T10MIN'
    
    print('Loading file:')
    print(fn)
    
    ds_raw = xr.open_dataset(fn, decode_times=False).\
        rename({t_var_name:'time'})
    ds_raw.time.attrs['calendar'] = 'gregorian'
    result = xr.decode_cf(ds_raw, decode_times=True)
    
    if 'DEPTH_ADCP' in result.coords.keys():
        result = result.squeeze('DEPTH_ADCP')
    if 'DEPTH_ADCP1' in result.coords.keys():
        result = result.squeeze('DEPTH_ADCP1')
    if 'DEPTH_ADCP2' in result.coords.keys():
        result = result.squeeze('DEPTH_ADCP2')
    #
    return result


def load_ADCP_data(yr, ID):
    if int(yr) == 2017:
        fn = config.data_dir + 'ADCP/' + \
            'TPOS2017-2018_SD' + str(ID) + 'adcp_1.0.cdf'
        print('Loading file:')
        print(fn)
        t_var_name = 'NEWT'
        ds_raw = xr.open_dataset(fn, decode_times=False).\
            rename({t_var_name:'time'})
        ds_raw.time.attrs['calendar'] = 'gregorian'
        result = xr.decode_cf(ds_raw, decode_times=True)
    else:
        sys.exit('ADCP data only available for 2017!')
    #
    return result


def load_1Hz_LW_data(yr, ID):
    """Loads 1 Hz longwave data for year yr and Saildrone ID.

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filenames = {
        '2018':{
            '1005':'tpos_2018_hf_1hz_1005_LW_20181003_20190127',
            '1006':'tpos_2018_hf_1hz_1006_LW_20181003_20190128',
            '1029':'tpos_2018_hf_1hz_1029_LW_20181004_20190128',
            '1030':'tpos_2018_hf_1hz_1030_LW_20181004_20190305'},
        '2019':{
            '1066':'tpos_2019_hf_1hz_1066_LW_20190608_20200106',
            '1067':'tpos_2019_hf_1hz_1067_LW_20190618_20200103',
            '1068':'tpos_2019_hf_1hz_1068_LW_20190608_20200107',
            '1069':'tpos_2019_hf_1hz_1069_LW_20190608_20191228'}
    }
    data_dir = config.data_dir + '1Hz/'
    file_suffix = '.nc'
    print('Loading file:')
    print(data_dir + filenames[yr][ID] + file_suffix)
    #
    result = load_swap_time(data_dir + filenames[yr][ID] + file_suffix)
    return result


def load_highfreq_SW_data(yr, ID):
    """Loads shortwave data for year yr and Saildrone ID.

    Requires some processing as the shortwave data isn't at the 
    time resolution of the files (e.g., SW at 5 Hz vs files at
    10 or 20 Hz).

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filelists = {
        '2018':{
            '1005':[
                'tpos_2018_hf_10hz_1005_SW_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1005_SW_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1005_SW_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1005_SW_20190101_20190131.nc'
            ],
            '1006':[
                'tpos_2018_hf_10hz_1006_SW_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1006_SW_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1006_SW_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1006_SW_20190101_20190131.nc'
            ],
            '1029':[
                'tpos_2018_hf_10hz_1029_SW_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1029_SW_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1029_SW_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1029_SW_20190101_20190131.nc'
            ],
            '1030':[
                'tpos_2018_hf_10hz_1030_SW_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1030_SW_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1030_SW_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1030_SW_20190101_20190131.nc',
                'tpos_2018_hf_10hz_1030_SW_20190201_20190305.nc'
            ]
        },
        '2019':{
            '1066':[
                'sd1066_2019_20hz_1066_SW_20190608_20190630.nc',
                'sd1066_2019_20hz_1066_SW_20190701_20190731.nc',
                'sd1066_2019_20hz_1066_SW_20190801_20190831.nc',
                'sd1066_2019_20hz_1066_SW_20190901_20190930.nc',
                'sd1066_2019_20hz_1066_SW_20191001_20191031.nc',
                'sd1066_2019_20hz_1066_SW_20191101_20191106.nc'
            ],
            '1067':[
                'sd1067_2019_20hz_1067_SW_20190618_20190630.nc',
                'sd1067_2019_20hz_1067_SW_20190701_20190731.nc',
                'sd1067_2019_20hz_1067_SW_20190801_20190831.nc',
                'sd1067_2019_20hz_1067_SW_20190901_20190930.nc',
                'sd1067_2019_20hz_1067_SW_20191001_20191031.nc',
                'sd1067_2019_20hz_1067_SW_20191101_20191130.nc',
                'sd1067_2019_20hz_1067_SW_20191201_20191231.nc',
                'sd1067_2019_20hz_1067_SW_20200101_20200103.nc'
            ],
            '1068':[
                'sd1068_2019_20hz_1068_SW_20190608_20190630.nc',
                'sd1068_2019_20hz_1068_SW_20190701_20190731.nc',
                'sd1068_2019_20hz_1068_SW_20190801_20190831.nc',
                'sd1068_2019_20hz_1068_SW_20190901_20190930.nc',
                'sd1068_2019_20hz_1068_SW_20191001_20191031.nc',
                'sd1068_2019_20hz_1068_SW_20191101_20191130.nc',
                'sd1068_2019_20hz_1068_SW_20191201_20191231.nc',
                'sd1068_2019_20hz_1068_SW_20200101_20200107.nc'
            ],
            '1069':[
                'sd1069_2019_20hz_1069_SW_20190608_20190630.nc',
                'sd1069_2019_20hz_1069_SW_20190701_20190731.nc',
                'sd1069_2019_20hz_1069_SW_20190801_20190831.nc',
                'sd1069_2019_20hz_1069_SW_20190901_20190930.nc',
                'sd1069_2019_20hz_1069_SW_20191001_20191031.nc',
                'sd1069_2019_20hz_1069_SW_20191101_20191130.nc',
                'sd1069_2019_20hz_1069_SW_20191201_20191228.nc'
            ]
        }
    }
    # Load all the data.
    fl = filelists[yr][ID]
    data_dir = config.data_dir + '10Hz/'
    ds_array = []
    for f in fl:
        print(data_dir + f)
        ds_array.append(load_swap_time(data_dir + f))
    print('Merging....')
    ds = xr.merge(ds_array, combine_attrs='override')
    
    print('Subsetting to 5 Hz...')
    result = ds.sel(time=(
        (ds.time.data - np.array(ds.time.data, dtype='datetime64[200ms]'))
        == np.timedelta64(0, 'ms')
    ))
    print('Subsetted data captures ' +
          str(np.sum(~np.isnan(result.SW_IRRAD_TOTAL.data))) +
          ' data points out of ' +
          str(np.sum(~np.isnan(ds.SW_IRRAD_TOTAL.data))))
    #
    return result


def load_highfreq_Temp_WindSt_data(yr, ID):
    """Loads temperature and WIND_STATUS data for year yr and Saildrone ID.

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filelists = {
        '2017':{
            '1005':[
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20170902_20170930.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20171001_20171031.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20171101_20171130.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20171201_20171231.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20180101_20180131.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20180201_20180228.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20180301_20180331.nc',
                'tpos_2017_hf_10hz_1005_Temp_WindSt_20180401_20180505.nc'
            ],
            '1006':[
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20170902_20170930.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20171001_20171031.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20171101_20171130.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20171201_20171231.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20180101_20180131.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20180201_20180228.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20180301_20180331.nc',
                'tpos_2017_hf_10hz_1006_Temp_WindSt_20180401_20180505.nc'
            ]
        },
        '2018':{
            '1005':[
                'tpos_2018_hf_10hz_1005_Temp_WindSt_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1005_Temp_WindSt_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1005_Temp_WindSt_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1005_Temp_WindSt_20190101_20190131.nc'
            ],
            '1006':[
                'tpos_2018_hf_10hz_1006_Temp_WindSt_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1006_Temp_WindSt_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1006_Temp_WindSt_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1006_Temp_WindSt_20190101_20190131.nc'
            ],
            '1029':[
                'tpos_2018_hf_10hz_1029_Temp_WindSt_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1029_Temp_WindSt_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1029_Temp_WindSt_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1029_Temp_WindSt_20190101_20190131.nc'
            ],
            '1030':[
                'tpos_2018_hf_10hz_1030_Temp_WindSt_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1030_Temp_WindSt_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1030_Temp_WindSt_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1030_Temp_WindSt_20190101_20190131.nc',
                'tpos_2018_hf_10hz_1030_Temp_WindSt_20190201_20190305.nc'
            ]
        },
        '2019':{
            '1066':[
                'sd1066_2019_20hz_1066_Temp_WindSt_20190608_20190630.nc',
                'sd1066_2019_20hz_1066_Temp_WindSt_20190701_20190731.nc',
                'sd1066_2019_20hz_1066_Temp_WindSt_20190801_20190831.nc',
                'sd1066_2019_20hz_1066_Temp_WindSt_20190901_20190930.nc'
            ],
            '1067':[
                'sd1067_2019_20hz_1067_Temp_WindSt_20190618_20190630.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20190701_20190731.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20190801_20190831.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20190901_20190930.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20191001_20191031.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20191101_20191130.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20191201_20191231.nc',
                'sd1067_2019_20hz_1067_Temp_WindSt_20200101_20200103.nc'
            ],
            '1068':[
                'sd1068_2019_20hz_1068_Temp_WindSt_20190608_20190630.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20190701_20190731.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20190801_20190831.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20190901_20190930.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20191001_20191031.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20191101_20191130.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20191201_20191231.nc',
                'sd1068_2019_20hz_1068_Temp_WindSt_20200101_20200107.nc'
            ],
            '1069':[
                'sd1069_2019_20hz_1069_Temp_WindSt_20190608_20190630.nc',
                'sd1069_2019_20hz_1069_Temp_WindSt_20190701_20190731.nc',
                'sd1069_2019_20hz_1069_Temp_WindSt_20190801_20190831.nc',
                'sd1069_2019_20hz_1069_Temp_WindSt_20190901_20190930.nc',
                'sd1069_2019_20hz_1069_Temp_WindSt_20191001_20191031.nc',
                'sd1069_2019_20hz_1069_Temp_WindSt_20191101_20191130.nc',
                'sd1069_2019_20hz_1069_Temp_WindSt_20191201_20191228.nc'
            ]
        }
    }
    # Load and merge all the data.
    fl = filelists[yr][ID]
    data_dir = config.data_dir + '10Hz/'
    ds_array = []
    for f in fl:
        print(data_dir + f)
        ds_array.append(load_swap_time(data_dir + f))
    print('Merging....')
    result = xr.merge(ds_array, combine_attrs='override')
    test = xr.concat(ds_array, dim='time', combine_attrs='override')
    #
    return result


def load_highfreq_raw_wind_data(yr, ID):
    """Loads raw wind data for year yr and Saildrone ID.
       (Raw = anemometer-relative coordinates)

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filelists = {
        '2017':{
            '1005':[
                'tpos_2017_hf_10hz_1005_uvwRaw_20170902_20170930.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20171001_20171031.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20171101_20171130.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20171201_20171231.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20180101_20180131.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20180201_20180228.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20180301_20180331.nc',
                'tpos_2017_hf_10hz_1005_uvwRaw_20180401_20180505.nc'
            ],
            '1006':[
                'tpos_2017_hf_10hz_1006_uvwRaw_20170902_20170930.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20171001_20171031.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20171101_20171130.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20171201_20171231.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20180101_20180131.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20180201_20180228.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20180301_20180331.nc',
                'tpos_2017_hf_10hz_1006_uvwRaw_20180401_20180505.nc'
            ]
        },
        '2018':{
            '1005':[
                'tpos_2018_hf_10hz_1005_uvwRaw_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1005_uvwRaw_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1005_uvwRaw_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1005_uvwRaw_20190101_20190131.nc'
            ],
            '1006':[
                'tpos_2018_hf_10hz_1006_uvwRaw_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1006_uvwRaw_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1006_uvwRaw_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1006_uvwRaw_20190101_20190131.nc'
            ],
            '1029':[
                'tpos_2018_hf_10hz_1029_uvwRaw_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1029_uvwRaw_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1029_uvwRaw_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1029_uvwRaw_20190101_20190131.nc'
            ],
            '1030':[
                'tpos_2018_hf_10hz_1030_uvwRaw_20181003_20181031.nc',
                'tpos_2018_hf_10hz_1030_uvwRaw_20181101_20181130.nc',
                'tpos_2018_hf_10hz_1030_uvwRaw_20181201_20181231.nc',
                'tpos_2018_hf_10hz_1030_uvwRaw_20190101_20190131.nc',
                'tpos_2018_hf_10hz_1030_uvwRaw_20190201_20190305.nc'
            ]
        },
        '2019':{
            '1066':[
                'sd1066_2019_20hz_1066_uvwRaw_20190608_20190630.nc',
                'sd1066_2019_20hz_1066_uvwRaw_20190701_20190731.nc',
                'sd1066_2019_20hz_1066_uvwRaw_20190801_20190831.nc',
                'sd1066_2019_20hz_1066_uvwRaw_20190901_20190930.nc',
                'sd1066_2019_20hz_1066_uvwRaw_20191001_20191031.nc',
                'sd1066_2019_20hz_1066_uvwRaw_20191101_20191106.nc'
            ],
            '1067':[
                'sd1067_2019_20hz_1067_uvwRaw_20190618_20190630.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20190701_20190731.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20190801_20190831.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20190901_20190930.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20191001_20191031.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20191101_20191130.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20191201_20191231.nc',
                'sd1067_2019_20hz_1067_uvwRaw_20200101_20200103.nc'
            ],
            '1068':[
                'sd1068_2019_20hz_1068_uvwRaw_20190608_20190630.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20190701_20190731.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20190801_20190831.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20190901_20190930.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20191001_20191031.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20191101_20191130.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20191201_20191231.nc',
                'sd1068_2019_20hz_1068_uvwRaw_20200101_20200107.nc'
            ],
            '1069':[
                'sd1069_2019_20hz_1069_uvwRaw_20190608_20190630.nc',
                'sd1069_2019_20hz_1069_uvwRaw_20190701_20190731.nc',
                'sd1069_2019_20hz_1069_uvwRaw_20190801_20190831.nc',
                'sd1069_2019_20hz_1069_uvwRaw_20190901_20190930.nc',
                'sd1069_2019_20hz_1069_uvwRaw_20191001_20191031.nc',
                'sd1069_2019_20hz_1069_uvwRaw_20191101_20191130.nc',
                'sd1069_2019_20hz_1069_uvwRaw_20191201_20191228.nc'
            ]
        }
    }
    # Load and merge all the data.
    fl = filelists[yr][ID]
    data_dir = config.data_dir + '10Hz/'
    ds_array = []
    for f in fl:
        print(data_dir + f)
        ds_array.append(load_swap_time(data_dir + f))
    print('Concatenating...')
    result = xr.concat(ds_array, dim='time', combine_attrs='override')
    #
    return result


def exclude_json(ds, sdyr, sdid, exc_file=None):
    """Reads 'bad data' events from a JSON file and sets data to NaN.
    """
    import json
    
    # Get a list of events from the JSON file.
    if exc_file is None:
        fn = config.data_dir + 'derived/' + 'exclude.json'
    else:
        fn = exc_file
    with open(fn, 'r') as myfile:
        file_contents=myfile.read()
    data_list = json.loads(file_contents)
    
    # Copy dataset to modify.
    ds_out = ds.copy()
    
    # Loop over events and exclude data.
    for ie, ev in enumerate(data_list):
        if ((ev['mission'] == str(sdyr)) &
            (ev['saildrone'] == str(sdid))):
            st_time = np.datetime64(
                datetime.datetime.strptime(ev['start_time'],
                                           "%d-%b-%Y %H:%M:%S")
            )
            end_time = np.datetime64(
                datetime.datetime.strptime(ev['end_time'],
                                           "%d-%b-%Y %H:%M:%S")
            )
            if ev['variables'] in ['ALL', 'all', 'All']:
                var_list = ds_out.keys()
            else:
                var_list = ev['variables'].split(',')
            for v in var_list:
                if v in ds.keys():
                    ds_out[v].loc[dict(time=((ds_out.time >= st_time) &
                                             (ds_out.time <= end_time)))] = \
                                                 np.nan
                else:
                    print('No variable ' + v + ' in data set.')
        else:
            continue
    #
    return ds_out
