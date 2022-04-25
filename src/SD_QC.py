"""Functions to check Saildrone data quality and formatting.
"""
#------------------------------------------------------------------------
#---- Analysis tools:
import numpy as np
import xarray as xr
import scipy.stats
import datetime
import cf_units
#---- System tools:
import sys
import os
import warnings
warnings.filterwarnings(action='ignore', module='xarray',
                        category=FutureWarning)
#---- My code:
#
#------------------------------------------------------------------------

def qc_times_unique(ds, fix=False):
    """Tests whether all time steps in a dataset are unique.
    
    Can optionally attempt to fix the problem ('fix=True') but this will fail
    if the repeated time steps have different data.
    
    Inputs:
        ds [xarray dataset] -- dataset to test
        fix [boolean] -- whether or not to attempt to remove duplicate times
    
    Return value is boolean if 'fix=False', or xarray dataset if 'fix=True'.
    """
    
    print('---------- qc_times_unique ----------')
    
    uniq = len(np.unique(ds.time.data)) == len(ds.time.data)
    if uniq:
        print('time axis has all unique values')
        ds_uniq = ds
    else:
        print('time axis has non-unique values')
        # If not unique, get rid of repeats (if specified in option fix).
        if fix:
            _, uniq_inds = np.unique(ds.time, return_index=True)
            uniq_mask = np.array([(i in uniq_inds)
                                  for i in range(len(ds.time))])
            print('Number of repeated time steps:  ' + str(np.sum(~uniq_mask)))
            ds_uniq = ds.isel(time=uniq_mask)
            ds_rep = ds.isel(time=~uniq_mask)
            for it, t in enumerate(ds_rep.time):
                print(t.data)
                all_the_same = ds_rep.isel(time=it).equals(
                    ds_uniq.sel(time=str(t.data)))
                if not all_the_same:
                    for v in list(ds.keys()):
                        print(str(ds_uniq[v].sel(time=str(t.data)).data) +
                              ' ::: ' +
                              str(ds_rep[v].isel(time=it).data))
                    sys.exit('Cannot get rid of duplicate time steps: they have different data')
    #
    print('')
    if fix:
        return ds_uniq
    else:
        return uniq


def qc_times_monotonic(ds):
    """Tests whether time coordinate monotonically increases.
    
    Input:
        ds [xarray dataset] -- dataset to test
    
    Return value is boolean: True if strictly monotonically increasing, 
    False otherwise.
    """
    
    print('---------- qc_times_monotonic ----------')
    
    # Calculate time differences.
    delta_t = ds.time.data[1:] - ds.time.data[:-1]
    # Test whether all are strictly positive.
    str_monot = np.all(delta_t > np.timedelta64(0,'s'))
    if str_monot:
        print('time axis is strictly monotonically increasing')
    else:
        print('time axis is not strictly monotonically increasing')
    #
    print('')
    return str_monot


def qc_times_evenly_spaced(ds, tol=None):
    """Tests if time coordinate is evenly spaced, within optional tolerance.
    
    Inputs
        ds [xarray dataset] -- dataset to test
        tol [None or numpy timedelta64] -- optional tolerance
    
    Return value is True if:
        |maximum(time delta) - minimum(time delta)| <= tol (tol defined)
        or
        maximum(time delta) == minimum(time delta) (tol not defined)
    False otherwise.
    """
    
    print('---------- qc_times_evenly_spaced ----------')
    
    # Calculate time differences.
    delta_t = ds.time.data[1:] - ds.time.data[:-1]
    #
    # Check if max and min are equal and are real numbers.
    if tol:
        even_spaced = (delta_t.max() - delta_t.min() <= tol) and \
            (~np.isnat(delta_t.max()))
    else:
        even_spaced = (delta_t.max() == delta_t.min()) and \
            (~np.isnat(delta_t.max()))
    if even_spaced:
        print('time axis is evenly spaced')
        print('(max delta_t = '
              + str(delta_t.max()/np.timedelta64(1,'s'))
              + ')')
        print('(min delta_t = '
              + str(delta_t.min()/np.timedelta64(1,'s'))
              + ')')
    else:
        print('time axis is not evenly spaced')
        print('(max delta_t = '
              + str(delta_t.max()/np.timedelta64(1,'s'))
              + ')')
        print('(min delta_t = '
              + str(delta_t.min()/np.timedelta64(1,'s'))
              + ')')
    #
    print('')
    return even_spaced


def qc_times_make_even(ds, dt_ms, tol=None):
    """Fill gaps in ds with NaNs, to make a continuous, even time coordinate.
    
    ds [xarray dataset] -- xarray dataset
    dt_ms [int or float] -- desired time step, in milliseconds.
    tol [None or numpy timedelta64] -- optional tolerance
    
    This uses a manual (ie., not xarray built-in) method that should be quicker 
    for fairly small numbers of missing time steps.
    """
    
    print('---------- qc_times_make_even ----------')
    
    # Get time step as a string (for resampling) and a datetime64.
    dt_str = '{:.0f}'.format(dt_ms) + 'ms'
    dt_64 = np.timedelta64(int(dt_ms), 'ms')
    # Calculate time differences and where they are not the desired length.
    delta_t = ds.time.data[1:] - ds.time.data[:-1]
    if tol:
        delta_t_wrong = np.argwhere(np.abs(delta_t - dt_64) > tol)
    else:
        delta_t_wrong =  np.argwhere(delta_t/np.timedelta64(1,'ms') != dt_ms)
    
    # If there are missing time steps, create a list of sections to join:
    # At the end of this process, this list will contain alternating sections:
    # 'original', 'new', 'original', ..., 'new', 'original'
    # which can then be appended together.
    if len(delta_t_wrong) > 0:
        print(str(len(delta_t_wrong)) + ' gaps to fill')
        join_list = []
        #
        # Loop over wrong delta_t values:
        for ii in range(len(delta_t_wrong)):
            i_gap = int(delta_t_wrong[ii])
            # Get section before join (empty dataset if i_gap == 0).
            if ii == 0:
                join_list.append(ds.isel(time=slice(None,i_gap)))
            else:
                i_last_gap = int(delta_t_wrong[ii-1])
                join_list.append(ds.isel(time=slice(i_last_gap+2,i_gap)))
            # Get join section.
            join_list.append(ds.isel(time=slice(i_gap,i_gap+2)).\
                resample(time=dt_str, keep_attrs=True).asfreq())
            # Get section after join (only needed for last gap).
            if ii == (len(delta_t_wrong) - 1):
                join_list.append(ds.isel(time=slice(i_gap+2,None)))
        # Combine them all.
        ds_even = xr.concat(join_list,dim='time',combine_attrs='override')
    
    # If the time series is already evenly spaced, just return the
    # original dataset.
    else:
        ds_even = ds
    #
    print('')
    return ds_even

def qc_latlon_diff(ds, tol, t_thresh, remove=False):
    """Check for erroneous lat and lon values.
 
    ds must have time, latitude and longitude variables.
    tol [numeric scalar] -- the max allowable lat/lon step, in DEGREES
    t_thresh [numpy timeldelta64] -- time steps larger than this are 
                                     screened out (e.g., to deal with 
                                     long gaps in a time series) when 
                                     checking lat/lon steps.

    Returns: ds_out (ds with [if remove=True] bad lat/lon values removed)
    and list of 'flagged' timesteps.

    This function checks the largest latitude and longitude differences 
    and flags/removes values that are likely too large. 
    """
    print('---------- qc_latlon_diff ----------')
    
    # Calculate and print deltas.
    delta_t = ds.time.data[1:] - ds.time.data[:-1]
    delta_lat = ds['latitude'].data[1:] - ds['latitude'].data[:-1]
    delta_lon = ds['longitude'].data[1:] - ds['longitude'].data[:-1]
    print('---- latitude:')
    print('max       = ' + str(np.nanmax(delta_lat)))
    print('min       = ' + str(np.nanmin(delta_lat)))
    print('max abs   = ' + str(np.nanmax(np.abs(delta_lat))))
    print('min abs   = ' + str(np.nanmin(np.abs(delta_lat))))
    print('mean abs  = ' + str(np.nanmean(np.abs(delta_lat))))
    print('---- longitude:')
    print('max       = ' + str(np.nanmax(delta_lon)))
    print('min       = ' + str(np.nanmin(delta_lon)))
    print('max abs   = ' + str(np.nanmax(np.abs(delta_lon))))
    print('min abs   = ' + str(np.nanmin(np.abs(delta_lon))))
    print('mean abs  = ' + str(np.nanmean(np.abs(delta_lon))))
    
    # For small enough time steps, flag distances larger than tol.
    wrong_1 = [i for i, x in enumerate(
        (delta_t < t_thresh) &
        ((np.abs(delta_lon) > tol) |
         (np.abs(delta_lat) > tol))
    ) if x]
    for i in wrong_1:
        print(str(ds.time.data[i]) + ' -> ' +
              str( ds.time.data[i+1]))
        print(str(ds['latitude'].data[i]) + ' -> ' +
              str(ds['latitude'].data[i+1]))
        print(str(ds['longitude'].data[i]) + ' -> ' +
              str(ds['longitude'].data[i+1]))
        print('')
    
    # Remove flagged values if required. 
    ds_out = ds.copy()
    if remove & (len(wrong_1) > 0):
        print('Removing bad lat and lon values.')
        lat_nansum_before = np.sum(np.isnan(ds_out.latitude))
        lon_nansum_before = np.sum(np.isnan(ds_out.longitude))
        for ii in wrong_1:
            lat_median = ds.latitude.\
                sel(time=((ds.time > (ds.time[ii] - t_thresh)) &
                          (ds.time < (ds.time[ii] + t_thresh)))).\
                median(dim='time', skipna=True)
            lon_median = ds.longitude.\
                sel(time=((ds.time > (ds.time[ii] - t_thresh)) &
                          (ds.time < (ds.time[ii] + t_thresh)))).\
                median(dim='time', skipna=True)
            if (np.abs(ds.latitude[ii] - lat_median) > tol):
                ds_out.latitude.data[ii] = np.nan
            if (np.abs(ds.latitude[ii+1] - lat_median) > tol):
                ds_out.latitude.data[ii+1] = np.nan
            if (np.abs(ds.longitude[ii] - lon_median) > tol):
                ds_out.longitude.data[ii] = np.nan
            if (np.abs(ds.longitude[ii+1] - lon_median) > tol):
                ds_out.longitude.data[ii+1] = np.nan
        lat_nansum_after = np.sum(np.isnan(ds_out.latitude))
        lon_nansum_after = np.sum(np.isnan(ds_out.longitude))
        print('Removed ' + str((lat_nansum_after - lat_nansum_before).data) +
              ' latitude values')
        print('Removed ' + str((lon_nansum_after - lon_nansum_before).data) +
              ' longitude values') 
        delta_t = ds_out.time.data[1:] - ds_out.time.data[:-1]
        delta_lat = ds_out['latitude'].data[1:] \
            - ds_out['latitude'].data[:-1]
        delta_lon = ds_out['longitude'].data[1:] \
            - ds_out['longitude'].data[:-1]
        print('---- latitude:')
        print('max       = ' + str(np.nanmax(delta_lat)))
        print('min       = ' + str(np.nanmin(delta_lat)))
        print('max abs   = ' + str(np.nanmax(np.abs(delta_lat))))
        print('min abs   = ' + str(np.nanmin(np.abs(delta_lat))))
        print('mean abs  = ' + str(np.nanmean(np.abs(delta_lat))))
        print('---- longitude:')
        print('max       = ' + str(np.nanmax(delta_lon)))
        print('min       = ' + str(np.nanmin(delta_lon)))
        print('max abs   = ' + str(np.nanmax(np.abs(delta_lon))))
        print('min abs   = ' + str(np.nanmin(np.abs(delta_lon))))
        print('mean abs  = ' + str(np.nanmean(np.abs(delta_lon))))
    #
    print('')
    return ds_out, wrong_1


def qc_latlon_remove_zero_pairs(ds):
    """Finds where latitude and longitude are both zero and converts to nan.

    ds must have 'latitude' and 'longitude' variables.
    """
    latlon_both_zero = (ds.latitude == 0.0) & (ds.longitude == 0.0)
    ds_out = ds.copy()
    if np.any(latlon_both_zero):
        print('---------- qc_latlon_remove_zero_pairs ----------')
        lat_nansum_before = np.sum(np.isnan(ds_out.latitude))
        lon_nansum_before = np.sum(np.isnan(ds_out.longitude))
        ds_out.latitude[latlon_both_zero] = np.nan
        ds_out.longitude[latlon_both_zero] = np.nan
        lat_nansum_after = np.sum(np.isnan(ds_out.latitude))
        lon_nansum_after = np.sum(np.isnan(ds_out.longitude))
        print('Removed ' + str((lat_nansum_after - lat_nansum_before).data) +
              ' latitude values')
        print('Removed ' + str((lon_nansum_after - lon_nansum_before).data) +
              ' longitude values')
        #
    print('')
    return ds_out

def qc_wind_status(ds, var_names, qflag=False):
    """Remove wind data where WIND_STATUS is not equal to zero.

    Inputs:
        ds [xarray dataset] -- dataset containing a WIND_STATUS variable
            and other data variables to which the process is applied. 
        var_names [list of strings] -- the names of the variables to 
            apply the process to. E.g., ['UWND', 'VWND', 'WWND']
            or ['UWND', 'VWND', 'WWND', 'WIND_SONICTEMP']
        qflag [boolean] -- if True, leave the 'spike' data in and add a flag 
            variable; if False, remove the spike data.

    Outputs:
        ds_out [xarray dataset] -- depends on qflag:
            If qflag==False, this is the same as ds but with values set to 
            np.nan where ds exceeds the MAD threshold.
            If qflag==True, this is the same as ds but with Flag variables
            added for each variable in var_names.
    """
    if qflag:
        print('Flagging WIND_STATUS != 0')
    else:
        print('Removing WIND_STATUS != 0')
    
    # Create solution variables.
    ds_out = ds.copy()
    
    # Loop over variables to calculate outliers for each.
    for v in var_names:
        print('WIND_STATUS != 0 AND ' + v + ' != NaN: ' +
              str(np.sum( (ds['WIND_STATUS'].data != 0) &
                          ~np.isnan(ds[v].data)           )))
        # Construct flag variables.
        if qflag:
            f1, f2 = initialize_flags(ds_out[v])
            f1 = f1.where(ds_out.WIND_STATUS == 0, other=get_flag1('bad'))
            vflag1_name = v + '_FLAG_PRIMARY'
            if (vflag1_name in ds_out.keys()):
                ds_out[vflag1_name] = ds_out[vflag1_name].where(
                    ((ds_out[vflag1_name] == get_flag1('bad')) |
                     (ds_out[vflag1_name] == get_flag1('missing'))),
                    other=f1
                )
            else:
                ds_out[vflag1_name] = f1
        # OR just remove the spikes.
        else:
            ds_out[v] = ds_out[v].where(
                ds_out.WIND_STATUS == 0,
                other=np.nan
            )
    #
    return ds_out


def qc_wind_zero_check(ds, var_names=['UWND','VWND','WWND']):
    """Check for suspicious frequency of zeros in wind data.
    
    Input:
        ds -- xarray dataset with wind component variables,
              e.g., UWND, VWND, WWND.
        var_names -- list of variable names for three wind components.
    """
    un = var_names[0]
    vn = var_names[1]
    wn = var_names[2]
    print('---------- Coincident zeros ----------')
    print(un + ',' + vn + ',' + wn + ': ' + str(np.sum(
        (ds[un].data == 0.0) & (ds[vn].data == 0.0) & (ds[wn].data == 0.0)
    )))
    print(un + ',' + vn + ': ' + str(np.sum(
        (ds[un].data == 0.0) & (ds[vn].data == 0.0)
    )))
    print(un + ',' + wn + ': ' + str(np.sum(
        (ds[un].data == 0.0) & (ds[wn].data == 0.0)
    )))
    print(vn + ',' + wn + ': ' + str(np.sum(
        (ds[vn].data == 0.0) & (ds[wn].data == 0.0)
    )))
    print('---------- Frequency of zeros ----------')
    print('(Ratio relative to adjacent bins: 0.01, 0.02, 0.03, etc.)')
    for v in var_names:
        count = 0
        for ws in np.arange(0.01, 0.1, 0.01):
            count = np.nanmax([count,
                               np.sum(ds[v].data == ws),
                               np.sum(ds[v].data == -1.0*ws)])
        print(v + ': ' + str(
            np.sum(ds[v].data == 0.0)/count
        ))
    #
    return


def despike_MAD(ds, var_names, SD_threshold=5.0, qflag=False):
    """Despike variables in ds using median absolute deviation method.

    Inputs:
        ds [xarray dataset] -- dataset of 2D arrays with dimensions 
             (time x sample_time). The median absolute deviations are
             calculated for each chunk.
        var_names [list] -- the names of variables to de-spike
        SD_threshold [float] -- the multiple of standard deviations to 
            consider as outliers. Typically around 5 or 6.
        qflag [boolean] -- if True, leave the 'spike' data in and add a flag 
            variable; if False, remove the spike data.

    Outputs:
        ds_despike [xarray dataset] -- same as ds but with values set to 
            np.nan where ds exceeds the MAD threshold.
        outlier_count [xarray dataset] -- a 1-dimensional ('time') array 
            containing counts for each chunk of the number 
            of data points in ds that exceed the MAD threshold. 

    This function is usually be used on de-trended data, but could
    be used on any dataset that meets the dimensionality criteria.
    """
    if qflag:
        print('Flagging outliers')
    else:
        print('Removing outliers')
    
    # Create solution variables.
    ds_despike = ds.copy()
    outlier_counts = np.sum(
        np.isnan(ds[var_names]),axis=1
    ).assign(time=ds.time)*np.nan
    # Loop over variables to calculate outliers for each.
    for v in var_names:
        chunk_medians = np.nanmedian(ds[v], axis=1, keepdims=True)
        chunk_MADs = scipy.stats.median_abs_deviation(
            ds[v], axis=1, nan_policy='omit'
        )
        chunk_MAD_ratios = ((ds[v]-chunk_medians)
                            /np.expand_dims(chunk_MADs,axis=1))
        # Construct flag variables.
        if qflag:
            f1, f2 = initialize_flags(ds[v])
            f1.data = np.where(
                np.abs(chunk_MAD_ratios) < 1.4826*SD_threshold,
                get_flag1('good'), get_flag1('bad')
            )
            f1 = f1.where(
                ~np.isnan(ds[v]), get_flag1('missing')
            )
            f2.data = np.where(
                np.abs(chunk_MAD_ratios) >= 1.4826*SD_threshold,
                f2 + get_flag2_addition('spike'), f2
            )
            vflag1_name = v + '_FLAG_PRIMARY'
            vflag2_name = v + '_FLAG_SECONDARY'
            if (vflag1_name in ds_despike.keys()):
                ds_despike[vflag1_name] = ds_despike[vflag1_name].where(
                    ds_despike[vflag1_name] == get_flag1('bad'),
                    f1
                )
            else:
                ds_despike[vflag1_name] = f1
            if (vflag2_name in ds_despike.keys()):
                ds_despike[vflag2_name] = ds_despike[vflag2_name] + f2
            else:
                ds_despike[vflag2_name] = f2
        # OR just remove the spikes.
        else:
            ds_despike[v] = ds_despike[v].where(
                np.abs(chunk_MAD_ratios) < 1.4826*SD_threshold,
                other=np.nan
            )
        # Count the spikes per chunk.
        outlier_counts[v].data = np.sum(
            np.abs(chunk_MAD_ratios) >= 1.4826*SD_threshold,
            axis=1
        )
        print('Variable ' + v + ': ' +
              str(np.sum(outlier_counts[v].data[:])) +
              ' outliers out of ' +
              str(np.sum(~np.isnan(ds[v].data[:]))) +
              ' original data points')
    #
    return ds_despike, outlier_counts


def qc_hard_limits(ds,
                   var_names, lims_lowerupper,
                   qflag=False):
    """Applies hard limits to variables in ds.

    Inputs:
        ds [xarray dataset] 
        var_names [list of strings] -- list of variables in ds
            that the hard limits will be applied to. 
            Must have an entry in the list of limits.
        lims_lowerupper [numpy array] -- limits, lower then upper,
            for each variable in var_names. 
            Size must be (len(var_names), 2).
        qflag [boolean] -- whether to flag data outside the limits
            or set them to NaN.
    
    Output:
        ds_out [xarray dataset] -- same as ds but with values in 
            var_names flagged or set to NaN where they fall 
            outside the limits.
    """
    
    ds_out = ds.copy()
    
    for iv, v in enumerate(var_names):
        # Construct flag variables.
        if qflag:
            f1, f2 = initialize_flags(ds[v])
            f1.data = np.where(
                ((ds_out[v] >= lims_lowerupper[iv,0]) &
                 (ds_out[v] <= lims_lowerupper[iv,1])),
                get_flag1('good'), get_flag1('bad')
            )
            f1 = f1.where(
                ~np.isnan(ds_out[v]), other=get_flag1('missing')
            )
            f2.data = np.where(
                ((ds_out[v] >= lims_lowerupper[iv,0]) &
                 (ds_out[v] <= lims_lowerupper[iv,1])),
                f2, f2 + get_flag2_addition('hard_limit')
            )
            vflag1_name = v + '_FLAG_PRIMARY'
            vflag2_name = v + '_FLAG_SECONDARY'
            if (vflag1_name in ds_out.keys()):
                ds_out[vflag1_name] = ds_out[vflag1_name].where(
                    ds_out[vflag1_name] == get_flag1('bad'),
                    other=f1
                )
            else:
                ds_out[vflag1_name]= f1
            if (vflag2_name in ds_out.keys()):
                ds_out[vflag2_name] = ds_out[vflag2_name] + f2
            else:
                ds_out[vflag2_name] = f2
        # OR just remove the spikes.
        else:
            ds_out[v] = ds_out[v].where(
                ((ds_out[v] >= lims_lowerupper[iv,0]) &
                 (ds_out[v] <= lims_lowerupper[iv,1])),
                other=np.nan
            )
    #
    return ds_out


def qc_rate_change(ds, var_names, dv, dt, qflag=False):
    """Tests for excessive time rate of change.

    Inputs:
        ds [xarray dataset] 
        var_names [list of strings] -- list of variables in ds
            that the test will be applied to. 
            Must have an entry in the list of limits.
        dv [numpy float array] -- the maximum allowed change in the  
            variable over time dt. 
            Size must be len(var_names).
        dt [numpy timedelta64] -- the time interval over which the 
            rate is evaluated. Either one per variable or a single value.
            Should be of the order of the timestep of ds.
        qflag [boolean] -- whether to flag data that fails the test
            or set them to NaN.
    
    Output:
        ds_out [xarray dataset] -- same as ds but with values in 
            var_names flagged or set to NaN where they fail the test.

    Note that changes are evaluated over consecutive data points: each missing
    data point in the original array results in two missing data points in
    the rate of change test. 

    """
    ds_out = ds.copy()
    
    # Sort out the time interval.
    if np.isscalar(dt) == 1:
        dt_arr = np.tile(dt, len(dv))
    elif len(dt) == len(dv):
        dt_arr = dt
    else:
        sys.exit('qc_rate_change: dt must have length of 1 or len(dv).')
        
    # Loop over variables.
    for iv, v in enumerate(var_names):
        # Calculate difference array as a fraction of dv.
        v_diff = xr.zeros_like(ds_out[v])
        v_diff.data[1:] = (ds_out[v].data[1:] - ds_out[v].data[:-1])/dv[iv]
        # Calculate time difference array as a fraction of dt.
        t_diff = xr.ones_like(ds_out[v])
        t_diff.data[1:] = (ds_out['time'].data[1:]
                           - ds_out['time'].data[:-1])\
                           /dt_arr[iv]
        # Calculate rate array.
        dv_dt = v_diff/t_diff
        dv_dt[0] = np.nan
        # Construct flag variables.
        if qflag:
            f1, f2 = initialize_flags(ds[v])
            f1.data = np.where(
                np.abs(dv_dt) <= 1.0,
                get_flag1('good'), get_flag1('bad')
            )
            f1 = f1.where(~np.isnan(ds_out[v]), other=get_flag1('missing'))
            f2 = np.where(
                np.abs(dv_dt) > 1.0,
                f2 + get_flag2_addition('time_rate_of_change'),
                f2
            )
            vflag1_name = v + '_FLAG_PRIMARY'
            vflag2_name = v + '_FLAG_SECONDARY'
            if (vflag1_name in ds_out.keys()):
                ds_out[vflag1_name] = ds_out[vflag1_name].where(
                    ds_out[vflag1_name] == get_flag1('bad'),
                    other=f1
                )
            else:
                ds_out[vflag1_name] = f1
            if (vflag2_name in ds_out.keys()):
                ds_out[vflag2_name] = ds_out[vflag2_name] + f2
            else:
                ds_out[vflag2_name] = f2
        # OR just remove the spikes.
        else:
            ds_out[v] = ds_out[v].where(np.abs(dv_dt) <= 1.0,
                                        other=np.nan)
    #
    return ds_out


def get_flag2_addition(test_type):
    """Returns the flag value to add to the secondary flag for 'test_type'.

    Input:
        test_type [string] -- one of the tests listed, or 'all'.
    Output:
        flag [int] -- the flag to add for that test, or the 
                      entire dictionary for 'all'.

    New test_types can be added -- the corresponding flag values should
    go up by powers of 2 up to 2^14 (1,2,4,8,16,...16384) for a total
    of 15 tests.
    
    If we need more flags, can change to a 32 bit flag variable."""
    test_to_flag = {
        'spike':1,
        'interpolated':2,
        'unphysical':4,
        'climatology':8,
        'hard_limit':16,
        'time_rate_of_change':32,
        'expert_review':16384
    }
    #
    if test_type == 'all':
        result = test_to_flag
    else:
        result = test_to_flag[test_type]
    return result


def get_flag1(q, int_type=np.int8):
    """Returns the primary flag value for quality q.

    Input:
        q [string] -- one of 'good', 'not_evaluated', 
                             'suspect', 'bad', 'missing'
                             'all' (returns entire dictionary)
        int_type [numpy type] -- e.g., np.int8

    Output:
        flag [numpy integer of class int_type] -- numerical flag value 
    """
    q_to_flag = {
        'good':1,
        'not_evaluated':2,
        'suspect':3,
        'bad':4,
        'missing':9
    }
    #
    if q == 'all':
        result = q_to_flag
    else:
        result = np.array([q_to_flag[q]]).astype(int_type)
    return result


def initialize_flags(da):
    """Initialize arrays of primary and secondary flags for given array shape.

    Input:
        da [xarray data array] -- just passed for shape and coords/dims
    Output:
        flag1 [xarray data array] -- primary flag (good, bad etc.)
            same shape, coords, dims as da.
        flag2 [xarray data array] -- secondary flag (which tests failed)
            same shape, coords, dims as da.
    """
    flag1 = get_flag1('not_evaluated')*xr.ones_like(da,dtype=np.int8)
    flag1.attrs['long_name'] = 'Primary data quality flag'
    flag1.attrs['description'] = 'Describes overall quality of data (good, bad, etc.).'
    flag1.attrs['values'] = str(list(get_flag1('all').values()))
    flag1.attrs['meaning'] = str(get_flag1('all'))
    #
    flag2 = xr.zeros_like(da,dtype=np.int16)
    flag2.attrs['long_name'] = 'Secondary data quality flag'
    flag2.attrs['description'] = 'Describes reasons for flagging (tests failed etc.).'
    flag2.attrs['values'] = '16-bit integer (positive only): 0-32767'
    flag2.attrs['meaning'] = 'Additive flags: ' + str(get_flag2_addition('all'))
    return flag1, flag2
    

def xarray_boolean_contingency(a, a_name, b, b_name):
    """Prints out contingency table of boolean xarray arrays a and b.

    a, b -- xarray data arrays containing only True & False values. 
           Must have time coordinates.
    a_name, b_name -- strings, names to use for more useful printouts.

    As long as there is some overlapping time, then a dictionary is 
    returned containing the contingency table info. Otherwise, the 
    dictionary is empty.
    """
    both = (a) & (b)
    a_only = (a) & (~b)
    b_only = (~a) & (b)
    neither = (~a) & (~b)
    if len(both) == 0:
        result = {}
    else:
        result = {
            'both':np.sum(both.data),
            a_name + ' only':np.sum(a_only.data),
            b_name + ' only':np.sum(b_only.data),
            'neither':np.sum(neither.data)
        }
        print('both:')
        print(str(np.sum(both.data)))
        print(a_name + ' only:')
        print(str(np.sum(a_only.data)))
        print(b_name + ' only:')
        print(str(np.sum(b_only.data)))
        print('neither:')
        print(str(np.sum(neither.data)))
    #
    return result
