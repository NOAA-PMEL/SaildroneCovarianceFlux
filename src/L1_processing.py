"""Performs data processing to get to 'L1' high frequency data.

Usage:
    $ python L1_processing year drone [--downsample=x] [--period=y]
where
    year is the mission year (2017, 2018, 2019, ...)
    drone is the Saildrone ID (1005, 1006, etc.)
and optionally:
    --downsample=x (x is an integer): downsample the input time series to
          x Hz before QCing etc. [Default: no downsampling]
    --period=y (y is an integer): use periods of y minutes for the spike 
          check algorithm. [Default: 30 minutes]

Processing steps:
    Time axis evenly spaced - no missing, repeated or non-monotonic time steps
    Spikes flagged

Some parameters of the processing steps are currently hardwired.
"""

#---------------------------------------------------------------------
import sys
import os
import numpy as np
import scipy.stats
import xarray as xr
import pandas as pd
import datetime
import cf_units
#---- My code:
import config
import SD_mission_details
from SD_IO import *
from SD_QC import *
from utils_time_chunk import *
from utils_timeseries import *
#---------------------------------------------------------------------

def main():
    
    print('---------- Loading data ----------')
    argv = sys.argv
    optional_args = {}
    assert len(argv) >= 3,\
        'At least 2 command line arguments required.'
    if len(argv) > 3:
        for i in argv[3:]:
            optional_args[i.split('=')[0][2:]] = i.split('=')[1]
    #
    ds_all = load_high_freq_multifile(argv[1], argv[2])
    ds_tws = load_highfreq_Temp_WindSt_data(argv[1], argv[2])
    print('Combining datasets...')
    if np.all(ds_all.time == ds_tws.time):
        ds_all['WIND_STATUS'] = ds_tws['WIND_STATUS']
        ds_all['WIND_SONICTEMP'] = ds_tws['WIND_SONICTEMP']
    else:
        ds_all = xr.merge([ds_all, ds_tws],
                          compat='override',
                          join='outer', fill_value=np.nan,
                          combine_attrs='no_conflicts')
    dt_ms = SD_mission_details.saildrone_frequency[argv[1]][argv[2]]['dt_ms']
    freq_hz_in = SD_mission_details.saildrone_frequency\
        [argv[1]][argv[2]]['freq_hertz']
    
    # Initialize description of QC methods, to save to file.
    QC_methods = ''
    
    print('---------- Formatting time series ----------')
    if 'downsample' in optional_args.keys():
        assert int(optional_args['downsample']) < freq_hz_in,\
            '--downsample must be from high frequency to lower frequency.'
        print('Downsampling...')
        freq_hz = int(optional_args['downsample'])
        ds_all = downsample(ds_all, freq_hz)
        QC_methods = QC_methods + 'Downsampled from ' + str(freq_hz_in) + \
            ' Hz to ' + str(freq_hz) + ' Hz. '
    else:
        freq_hz = freq_hz_in
    qc_times_unique(ds_all, fix=False)
    qc_times_monotonic(ds_all)
    even = qc_times_evenly_spaced(ds_all, tol=np.timedelta64(1, 'ms'))
    if not even:
        print('Making time steps even')
        ds_all = qc_times_make_even(ds_all, dt_ms)
        QC_methods = QC_methods + 'Time axis regularized. '
    # Select sub-period with nearly continuous high-freq wind data.
    st_time = SD_mission_details.saildrone_dates\
        [argv[1]][argv[2]]['start'].astype('datetime64[D]')
    if 'end_anemometer' in SD_mission_details.saildrone_dates\
       [argv[1]][argv[2]].keys():
         end_time = SD_mission_details.saildrone_dates\
            [argv[1]][argv[2]]['end_anemometer'].astype('datetime64[D]')\
            + np.timedelta64(1,'D')
    else:
        end_time = SD_mission_details.saildrone_dates\
            [argv[1]][argv[2]]['end'].astype('datetime64[D]')\
            + np.timedelta64(1,'D')
    ds_all = ds_all.sel(time=((ds_all.time >= st_time) &
                              (ds_all.time < end_time)))
    
    print('---------- Removing spikes ----------')
    #
    ds_all = qc_wind_status(ds_all,
                            ['UWND', 'VWND', 'WWND', 'WIND_SONICTEMP'],
                            qflag=False)
    QC_methods = QC_methods + 'Anemometer data removed where WIND_STATUS > 0. '
    ds_all = ds_all.drop('WIND_STATUS')
    #
    if 'period' in optional_args.keys():
        period_min = int(optional_args['period'])
    else:
        period_min = 30
    period_len = freq_hz*60*period_min
    # Cut into chunks.
    ds_2D = reshape_1D_nD_numpy(
        ds_all, [np.int64(len(ds_all.time)/period_len), period_len]
    )
    # Define which variables to remove spikes from.
    var_names = ['UWND', 'VWND', 'WWND', 'WIND_SONICTEMP']
    # Detrend.
    print('Detrending')
    ds_detrend, ds_trend, ds_slopes, ds_intercepts =  \
        ds_detrend_chunks(ds_2D, var_names)
    # Remove outliers.
    ds_despike, outlier_counts = despike_MAD(ds_detrend,
                                             var_names,
                                             SD_threshold=5.0,
                                             qflag=True)
    # Add trend back in.
    print('Adding trend back')
    for v in var_names:
        ds_despike[v] = ds_despike[v] + ds_trend[v]
        ds_despike[v].attrs = ds_2D[v].attrs
        ds_despike[v].attrs['QC_methods'] = \
            'Data removed where WIND_STATUS > 0. Spikes flagged. '
    # Reshape from chunks back to 1D.
    ds_1D = reshape_2D_1D_numpy(ds_despike)
    # Update QC methods description.
    QC_methods = QC_methods + 'Spikes flagged: where detrended ' + \
        'data are outside of +/- 5*1.4826*MAD of the ' + str(period_min) + \
        '-minute median. '

    """
    UNCOMMENT TO DO THE INTERPOLATION.
    print('---------- Filling gaps ----------')
    # Update methods string.
    QC_methods = QC_methods + 'Gaps interpolated: bad or missing ' + \
        'data periods of up to 0.5 seconds ' + \
        'filled by linear interpolation. '
    # Loop over variables to interpolate.
    for v in var_names:
        # Set the bad data to nan so we can interpolate it.
        orig = ds_1D[v].where(
            ds_1D[v + '_FLAG_PRIMARY'] != get_flag1('bad'),
            other=np.nan
        )
        # Interpolate gaps.
        interp = orig.interpolate_na(
            dim='time',
            method='linear',
            use_coordinate=True,
            max_gap=np.timedelta64(500,'ms'),
            keep_attrs=True
        )
        # Change flags.
        replaced = (np.isnan(orig)) & (~np.isnan(interp))
        ds_1D[v + '_FLAG_PRIMARY'].data[replaced] = get_flag1('good')
        ds_1D[v + '_FLAG_SECONDARY'].data[replaced] = (
            ds_1D[v + '_FLAG_SECONDARY'].data[replaced] +
            get_flag2_addition('interpolated')
        )
        # Add interpolated data back to original.
        ds_1D[v].data[replaced] = interp.data[replaced]
        ds_despike[v].attrs['QC_methods'] = \
            ds_despike[v].attrs['QC_methods'] + \
            'Gaps interpolated. '
        # Tidy up.
        del(interp)
        del(orig)
    """
    
    print('---------- Saving data ----------')
    ds_all = ds_1D
    # Edit metadata.
    for v in ['latitude','longitude','UWND','VWND','WWND','WIND_SONICTEMP']:
        ds_all[v].attrs['actual_range'] = \
            np.array([ds_all[v].min(), ds_all[v].max()])
    ds_all.attrs['Easternmost_Easting'] = \
        np.float64(ds_all.longitude[ds_all.longitude != 0].max().data)
    ds_all.attrs['Westernmost_Easting'] = \
        np.float64(ds_all.longitude[ds_all.longitude != 0].min().data)
    ds_all.attrs['geospatial_lon_max'] = \
        ds_all.attrs['Easternmost_Easting']
    ds_all.attrs['geospatial_lon_min'] = \
        ds_all.attrs['Westernmost_Easting']
    ds_all.attrs['Northernmost_Northing'] = \
        np.float64(ds_all.latitude[ds_all.latitude != 0].max().data)
    ds_all.attrs['Southernmost_Northing'] = \
        np.float64(ds_all.latitude[ds_all.latitude != 0].min().data)
    ds_all.attrs['geospatial_lat_max'] = \
        ds_all.attrs['Northernmost_Northing']
    ds_all.attrs['geospatial_lat_min'] = \
        ds_all.attrs['Southernmost_Northing']
    ds_all.attrs['time_coverage_start'] = \
        str(ds_all.time.min().data)
    ds_all.attrs['time_coverage_end'] = \
        str(ds_all.time.max().data)
    ds_all.attrs['QC_methods'] = QC_methods
    ds_all.attrs['QC_date'] = \
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds_all.attrs['QC_creator'] = 'Jack Reeves Eyre'
    # Define encoding.
    t_units = 'seconds since 2017-01-01 00:00:00'
    t_cal = 'standard'
    fill_val = 9.969209968386869e+36
    wr_enc = {'UWND':{'_FillValue':fill_val},
              'VWND':{'_FillValue':fill_val},
              'WWND':{'_FillValue':fill_val},
              'WIND_SONICTEMP':{'_FillValue':fill_val},
              'latitude':{'_FillValue':fill_val},
              'longitude':{'_FillValue':fill_val},
              'time':{'units':t_units,'calendar':t_cal,
                      'dtype':'float64'}}
    # Create output filename.
    data_dir = config.data_dir
    write_filename = data_dir + '10Hz/tpos_' + \
        argv[1] + \
        '_hf_' + str(freq_hz) + 'hz_' + \
        argv[2] + '_uvw_' + \
        pd.to_datetime(str(ds_all.time.min().data)).strftime('%Y%m%d') + '_' + \
        pd.to_datetime(str(ds_all.time.max().data)).strftime('%Y%m%d') + \
        '_QC_L1_' + str(period_min) + 'min.nc'
    print('Saving to:')
    print(write_filename)
    ds_all.to_netcdf(path=write_filename,
                     mode='w',format='NETCDF4_CLASSIC',
                     encoding=wr_enc,unlimited_dims=['time'])
    #
    return ds_1D


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    ds_all = main()
