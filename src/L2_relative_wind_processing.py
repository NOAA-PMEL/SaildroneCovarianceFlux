"""Performs data processing to get to 'L2' data - fluxes, etc.

Usage:
    $ python L2_processing year drone period
where
    year is the mission year (2017, 2018, 2019, ...)
    drone is the Saildrone ID (1005, 1006, etc.)
    period is the length of period for flux calculation (in minutes)
    downsample (optional, default=False) boolean describing whether to 
               downsample the 2019 data from 20 Hz to 10 Hz.

Processing steps:
    

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
import metpy
from metpy import calc
from metpy.units import units
#---- My code:
import config
import SD_mission_details
from SD_IO import *
from SD_QC import *
from utils_time_chunk import *
from utils_timeseries import *
#---------------------------------------------------------------------

def main():
    #
    # Parse command line options.
    downsample_2019 = False
    argv = sys.argv
    if len(argv) == 5:
        if argv[4] == 'downsample':
            downsample_2019 = True
        else:
            sys.exit("Command line option " + argv[4] + " not understood.")
    elif len(argv) != 4:
        sys.exit("Incorrect number of command line arguments (3 or 4 required)")
    assert int(argv[3]) == 10, \
        'Relative wind L2 flux method require 10-minute averaging period.'
    #
    print('---------- Loading data ----------')
    ds_all = load_L1_data(argv[1], argv[2], downsample=downsample_2019)
    # Get this mission's time step in milliseconds and frequency in Hertz.
    dt_ms = SD_mission_details.saildrone_frequency[argv[1]][argv[2]]['dt_ms']
    dt_64 = np.timedelta64(dt_ms, 'ms')
    freq_hz = SD_mission_details.saildrone_frequency\
        [argv[1]][argv[2]]['freq_hertz']
    if downsample_2019:
        dt_ms = int(dt_ms*2)
        dt_64 = dt_64*2
        freq_hz = int(freq_hz/2)
    # Get 1-minute data.
    ds_lo = load_1min_data(argv[1], argv[2])
    # Get ADCP data (from bulk file).
    ds_bulk = load_bulk_data(argv[1], argv[2])
    #
    print('---------- Pre-processing... ----------')
    print('Subset to offset times...')
    # Ensures that for, e.g., 10-minute periods, the first covariance
    # window covers hh:05:00 to hh:14:59.99, i.e., centered on hh:10:00.
    st_time_offset = np.timedelta64(30*int(argv[3]), 's')
    ds_all = ds_all.sel(time=slice(ds_all.time[0] + st_time_offset,
                                   ds_all.time[-1] - st_time_offset))
    print('Correcting vertical wind direction...')
    # We want upward wind speed.
    if str(argv[1]) in ['2019']:
        ds_all['WWND'].data = -1.0*ds_all['WWND'].data
        ds_all['WWND'].attrs['long_name'] = 'Upward wind speed'
        ds_all['WWND'].attrs['standard_name'] = 'upward_air_velocity'
    else:
        ds_all['WWND'].attrs['long_name'] = 'Upward wind speed'
        ds_all['WWND'].attrs['standard_name'] = 'upward_air_velocity'
    print('Applying flags...')
    # Apply all wind component flags to all anemometer variables.
    all_wind_ok = (
        ((ds_all['UWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_all['UWND_FLAG_PRIMARY'] == get_flag1('not_evaluated'))) &
        ((ds_all['VWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_all['VWND_FLAG_PRIMARY'] == get_flag1('not_evaluated'))) &
        ((ds_all['WWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_all['WWND_FLAG_PRIMARY'] == get_flag1('not_evaluated')))
    )
    for v in ['UWND', 'VWND', 'WWND', 'WIND_SONICTEMP']:
        ds_all[v] = ds_all[v].where(all_wind_ok, other=np.nan)
    # Then do the sonic temperature flags separately.
    for v in ['WIND_SONICTEMP']:
        ds_all[v] = ds_all[v].where(
            (ds_all[v + '_FLAG_PRIMARY'] == get_flag1('good')) |
            (ds_all[v + '_FLAG_PRIMARY'] == get_flag1('not_evaluated')),
            other=np.nan
        ) 
    #print('Interpolating...')
    print('Reshaping...')
    period_len = freq_hz*60*int(argv[3])
    ds_2D = reshape_1D_nD_numpy(
        ds_all, [np.int64(len(ds_all.time)/period_len), period_len]
    )
    print('Calculating current-relative winds...')
    ds_bulk = ds_bulk.assign_coords(time=(ds_bulk.time
                                          - np.timedelta64(5, 'm')))
    ds_mean = ds_2D.mean(dim='sample_time', skipna=True, keep_attrs=True)
    U_rel = ds_mean['UWND'] - ds_bulk['UCUR10MIN']
    V_rel = ds_mean['VWND'] - ds_bulk['VCUR10MIN']
    ds_mean['UREL'] = xr.full_like(ds_mean['UWND'], np.nan)
    ds_mean['VREL'] = xr.full_like(ds_mean['VWND'], np.nan)
    ds_mean['UREL'].loc[dict(time=U_rel.time.data)] = U_rel
    ds_mean['VREL'].loc[dict(time=V_rel.time.data)] = V_rel
    
    print('---------- Calculating covariance fluxes ----------')
    print('Calculating mean wind directions...')
    ds_mean = wind_dir_twice(ds_mean, 'UREL', 'VREL')
    #
    print('Transforming to streamwise flow...')
    ds_2D = rotation_xy(ds_2D, 'UWND', 'VWND',
                        np.radians(ds_mean['arctan_VoverU'].data), 'math')
    #
    print('Detrending...')
    ds_detrend, ds_trend, ds_slopes, ds_intercepts =  \
        ds_detrend_chunks(ds_2D, ['UWND','VWND',
                                  'WWND',
                                  'U_streamwise','U_crossstream',
                                  'WIND_SONICTEMP'])
    #
    print('Calculating covariances...')
    ds_detrend_mean = ds_detrend.mean(dim='sample_time',
                                      skipna=True, keep_attrs=True)
    ds_anom = ds_detrend - ds_detrend_mean
    ds_anom['time_first'] = ds_2D['time_first']
    ds_anom['time_last'] = ds_2D['time_last']
    ds_tau = dir_cov_time(ds_anom, frame_type='both')
    
    print('---------- Calculating thermodynamic quantities ----------')
    # Initialize arrays.
    rho = xr.DataArray(np.nan*np.zeros(len(ds_anom.time), dtype=np.float64),
                       coords=[ds_anom.time],
                       dims=['time'])
    rho_count = xr.full_like(rho, np.nan, dtype=np.int32)
    # Loop over times and calculate density one-at-a-time.
    for it in range(len(rho.time)):
        time_sub_bool = (
            (ds_lo.time > ds_2D.time_first[it] - dt_64/2) &
            (ds_lo.time < ds_2D.time_last[it] + dt_64/2)
        )
        T = ds_lo['TEMP_AIR_MEAN'].sel(time=time_sub_bool).mean().data*\
            units.degC
        T_count = np.sum(~np.isnan(ds_lo['TEMP_AIR_MEAN'].\
                                   sel(time=time_sub_bool)))
        RH = ds_lo['RH_MEAN'].sel(time=time_sub_bool).mean().data/100.0
        RH_count = np.sum(~np.isnan(ds_lo['RH_MEAN'].\
                                    sel(time=time_sub_bool)))
        p = ds_lo['BARO_PRES_MEAN'].sel(time=time_sub_bool).mean().data*\
            units.hPa
        p_count = np.sum(~np.isnan(ds_lo['BARO_PRES_MEAN'].\
                                   sel(time=time_sub_bool)))
        r = metpy.calc.mixing_ratio_from_relative_humidity(RH, T, p)
        rho.data[it] = np.array(metpy.calc.density(p, T, r))
        rho_count.data[it] = np.min([T_count, RH_count, p_count])
    # Multiply covariances by density.
    if 'taux' in ds_tau:
        ds_tau['taux'] = rho*ds_tau['taux']
        ds_tau['tauy'] = rho*ds_tau['tauy']
    if 'taus' in ds_tau:
        ds_tau['taus'] = rho*ds_tau['taus']
        ds_tau['tauc'] = rho*ds_tau['tauc']
    
    print('---------- Creating output dataset ----------')
    ds_tau['air_density'] = rho
    ds_tau['air_density_count'] = rho_count
    ds_tau['UWND'] = ds_mean['UWND']
    ds_tau['VWND'] = ds_mean['VWND']
    ds_tau['WWND'] = ds_mean['WWND']
    ds_tau['relative_wind_dir'] = ds_mean['wind_dir']
    ds_tau['arctan_VoverU_relative'] = ds_mean['arctan_VoverU']
    ds_tau['WIND_SONICTEMP'] = ds_mean['WIND_SONICTEMP']
    ds_tau['latitude'] = ds_mean['latitude']
    ds_tau['longitude'] = ds_mean['longitude']
    #
    ds_tau, wr_enc = add_tau_metadata(ds_tau, argv[3], freq_hz)
    ds_tau['arctan_VoverU_relative'].attrs['method'] = 'Uses relative wind: arctan[(V_air - V_water)/ (U_air - U_water)]'
    ds_tau['relative_wind_dir'].attrs['method'] = 'Uses relative wind with components (U_air - U_water) and (V_air - V_water)'
    ds_tau.attrs['license'] = ds_all.attrs['license']
    ds_tau.attrs['original_title'] = ds_all.attrs['title']
    #
    print('---------- Writing to file ----------')
    write_filename = config.data_dir + '10Hz/' + \
        'tpos_' + argv[1] + \
        '_hf_' + str(freq_hz) + 'hz_' + \
        argv[2] + '_uvw_' + \
        pd.to_datetime(str(ds_tau.time.min().data)).strftime('%Y%m%d') + \
        '_' + \
        pd.to_datetime(str(ds_tau.time.max().data)).strftime('%Y%m%d') + \
        '_' + \
        str(argv[3]) + 'min_fluxes_L2_relative_wind_direction.nc'
    print('Saving file:')
    print(write_filename)
    ds_tau.to_netcdf(path=write_filename,
                     mode='w',format='NETCDF4_CLASSIC',
                     encoding=wr_enc,unlimited_dims=['time'])
    return ds_tau, wr_enc


#---------------------------------------------------------------------


def dir_cov_time(ds, frame_type='both'):
    """Calculate covariances from anomaly data using time domain method.

    Inputs:
        ds [xarray dataset] -- must contain ANOMALIES. 
            (Anomalies here implies <mean along sample_time> = 0.)
            The variables must include wind components named
            UWND, VWND, WWND, U_streamwise and U_crossstream
            (or some combination depending on 'frame_type').
            Also needs coordinate 'time' and 'sample_time'.
        frame_type [one of 'earth', 'both', 'stream'] -- the type of 
            coordinate reference frame of the wind components in ds.
            'earth' calculates north and east components, requiring
                UWND and VWND.
            'stream' calculates along stream and across stream components
                requiring U_streamwise and U_crossstream. 
            'both' does both.
  
    Outputs:
        ds_out [xarray dataset] -- contains covariances, with dimension
            coordinate 'time' only. Also contains the count of data points 
            that go into each flux/stress data point.
    """
    # Initialize output dataset.
    ds_out = xr.Dataset()
    ds_out.coords['time'] = ds.coords['time']
    ds_out.coords['n_bounds'] = np.array([1,2])
    ds_out['time_bounds'] = (
        ('time', 'n_bounds'),
        xr.concat([ds['time_first'].expand_dims('n_bounds', 1),
                   ds['time_last'].expand_dims('n_bounds', 1)],
                  dim='n_bounds')
    )
    
    # The -1 multiplication in the calculations below assumes:
    #     w is positive upwards;
    #     the resulting wind stess is downward, i.e., accelerates
    #     the ocean in the direction of the wind.
    
    # Calculate Earth coordinate (North and East) stress components.
    if frame_type in ['both', 'earth']:
        ds_out['taux'] = -1.0*(ds['UWND']*ds['WWND']).mean(
            dim='sample_time',skipna=True, keep_attrs=True)
        ds_out['tauy'] = -1.0*(ds['VWND']*ds['WWND']).mean(
            dim='sample_time',skipna=True, keep_attrs=True)
        ds_out['taux_count'] = (ds['UWND']*ds['WWND']).count(
            dim='sample_time', keep_attrs=True)
        ds_out['tauy_count'] = (ds['VWND']*ds['WWND']).count(
            dim='sample_time', keep_attrs=True)
    
    # Calculate streamwise components.
    if frame_type in ['both', 'stream']:
        ds_out['taus'] = -1.0*(ds['U_streamwise']*ds['WWND']).mean(
            dim='sample_time',skipna=True, keep_attrs=True)
        ds_out['tauc'] = -1.0*(ds['U_crossstream']*ds['WWND']).mean(
            dim='sample_time',skipna=True, keep_attrs=True)
        ds_out['taus_count'] = (ds['U_streamwise']*ds['WWND']).count(
            dim='sample_time', keep_attrs=True)
        ds_out['tauc_count'] = (ds['U_crossstream']*ds['WWND']).count(
            dim='sample_time', keep_attrs=True)
    
    # Calculate buoyancy flux if available.
    if 'WIND_SONICTEMP' in ds.keys():
        ds_out['wp_Tsp'] = (ds['WIND_SONICTEMP']*ds['WWND']).mean(
            dim='sample_time',skipna=True, keep_attrs=True)
        ds_out['wp_Tsp_count'] = (ds['WIND_SONICTEMP']*ds['WWND']).count(
            dim='sample_time', keep_attrs=True)
    #
    return ds_out


def add_tau_metadata(ds, pd, freq):
    """Add metadata and define netCDF encoding for wind stress dataset.
    """
    # Encoding.
    t_units = 'seconds since 2017-01-01 00:00:00'
    t_cal = 'standard'
    fill_val = 9.969209968386869e+36
    fill_val_int = np.iinfo(np.int32).min + 2
    encodg = {'air_density':{'_FillValue':fill_val},
              'air_density_count':{'_FillValue':fill_val_int,'dtype':'int32'},
              'UWND':{'_FillValue':fill_val},
              'VWND':{'_FillValue':fill_val},
              'WWND':{'_FillValue':fill_val},
              'relative_wind_dir':{'_FillValue':fill_val},
              'arctan_VoverU_relative':{'_FillValue':fill_val},
              'WIND_SONICTEMP':{'_FillValue':fill_val},
              'latitude':{'_FillValue':fill_val},
              'longitude':{'_FillValue':fill_val},
              'time':{'units':t_units,'calendar':t_cal,
                      '_FillValue':fill_val,
                      'dtype':'float64'},
              'time_bounds':{'units':t_units,'calendar':t_cal,
                             '_FillValue':fill_val,
                             'dtype':'float64'}}
    if 'taux' in ds:
        encodg['taux'] = {'_FillValue':fill_val}
        encodg['tauy'] = {'_FillValue':fill_val}
        encodg['taux_count'] = {'_FillValue':fill_val_int,'dtype':'int32'}
        encodg['tauy_count'] = {'_FillValue':fill_val_int,'dtype':'int32'}
    if 'taus' in ds:
        encodg['taus'] = {'_FillValue':fill_val}
        encodg['tauc'] = {'_FillValue':fill_val}
        encodg['taus_count'] = {'_FillValue':fill_val_int,'dtype':'int32'}
        encodg['tauc_count'] = {'_FillValue':fill_val_int,'dtype':'int32'}
    if 'wp_Tsp' in ds:
        encodg['wp_Tsp'] = {'_FillValue':fill_val}
        encodg['wp_Tsp_count'] = {'_FillValue':fill_val_int,'dtype':'int32'}
    
    # Metadata.
    ds_out = ds.copy()
    tau_common_attrs = {
        'units':'N m-2',
        'method':'time domain covariance: time average of rho*(-u\'w\')',
        'averaging_period':(str(pd) + ' minutes'),
        'original_nominal_sampling_frequency':(str(freq) + ' Hz')
    }
    #
    if 'taux' in ds:
        ds_out['taux'].attrs = tau_common_attrs
        ds_out['taux'].attrs['standard_name'] = \
            'surface_downward_eastward_stress'
        ds_out['taux'].attrs['long_name'] = \
            'Zonal wind stress at ocean surface'
        ds_out['taux'].attrs['sign_convention'] = \
            'A positive downward eastward stress is a downward flux of eastward momentum, which accelerates the ocean eastward and the atmosphere westward.'
        ds_out['taux_count'].attrs = {'long_name':'Count of wind samples used in taux calculation per averaging period'}
    #
    if 'tauy' in ds:
        ds_out['tauy'].attrs = tau_common_attrs
        ds_out['tauy'].attrs['standard_name'] = \
            'surface_downward_northward_stress'
        ds_out['tauy'].attrs['long_name'] = \
            'Meridional wind stress at ocean surface'
        ds_out['tauy'].attrs['sign_convention'] = \
            'A positive downward northward stress is a downward flux of northward momentum, which accelerates the ocean northward and the atmosphere southward.'
        ds_out['tauy_count'].attrs = {'long_name':'Count of wind samples used in tauy calculation per averaging period'}
    #
    if 'taus' in ds:
        ds_out['taus'].attrs = tau_common_attrs
        ds_out['taus'].attrs['standard_name'] = \
            'surface_downward_downwind_stress'
        ds_out['taus'].attrs['long_name'] = \
            'Downwind (or streamwise or alongstream, using relative wind) wind stress at ocean surface'
        ds_out['taus'].attrs['sign_convention'] = \
            'A positive downward downwind stress is a downward flux of momentum in the direction of the mean relative wind, which accelerates the ocean in the direction of the mean relative wind and accelerates the atmosphere in the opposite direction.'
        ds_out['taus_count'].attrs = {'long_name':'Count of wind samples used in taus calculation per averaging period'}
    #
    if 'tauc' in ds:
        ds_out['tauc'].attrs = tau_common_attrs
        ds_out['tauc'].attrs['standard_name'] = \
            'surface_downward_crosswind_stress'
        ds_out['tauc'].attrs['long_name'] = \
            'Crosswind (relative wind) wind stress at ocean surface'
        ds_out['tauc'].attrs['sign_convention'] = \
            'A positive downward crosswind stress is a downward flux of momentum at a right angle to the mean relative wind, which accelerates the ocean in the direction 90 degrees anticlockwise of the mean relative wind, and accelerates the atmosphere in the opposite direction.'
        ds_out['tauc_count'].attrs = {'long_name':'Count of wind samples used in tauc calculation per averaging period'}
    #
    if 'wp_Tsp' in ds:
        ds_out['wp_Tsp'].attrs['units'] = 'K m s-1',
        ds_out['wp_Tsp'].attrs['method'] = \
            'time domain covariance: time average of (T_s\'w\')',
        ds_out['wp_Tsp'].attrs['averaging_period'] = \
            (str(pd) + ' minutes'),
        ds_out['wp_Tsp'].attrs['original_nominal_sampling_frequency'] = \
            (str(freq) + ' Hz')
        ds_out['wp_Tsp'].attrs['long_name'] = 'Upward kinematic flux of sonic temperature at ocean surface'
        ds_out['wp_Tsp'].attrs['alternative_name'] = 'Upward buoyancy flux at ocean surface' 
        ds_out['wp_Tsp'].attrs['sign_convention'] = 'Positive means the atmosphere gains buoyancy. Product of w perturbations (with w positive upwards) and sonic temperature perturbations (with positive warmer and more buoyant).'
        ds_out['wp_Tsp_count'].attrs = {'long_name':'Count of wind and sonic temperature samples used in wp_Tsp calculation per averaging period'}
    #
    ds_out['air_density'].attrs = {
        'units':'kg m-3',
        'standard_name':'air_density',
        'long_name':'Near surface air density',
        'method':'1-minute temperature, pressure and relative humidity; no height corrections applied; averaged over time to \'averaging_period\'; density calculated using MetPy package.',
        'averaging_period':(str(pd) + ' minutes')}
    ds_out['air_density_count'].attrs = {'long_name':'Count of T, RH and p data points used in air_density calculation per averaging period'}
    #
    ds_out.attrs = {
        'postprocessing_source_code':'L2_processing.py',
        'pp_creator_name':'Jack Reeves Eyre',
        'pp_date_created':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pp_description':'Direct covariance fluxes derived from Saildrone high resolution data'
    }
    #
    return ds_out, encodg

################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    ds_out, wr_enc = main()
