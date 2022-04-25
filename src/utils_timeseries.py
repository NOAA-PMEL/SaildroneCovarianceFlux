""" Utilities applied to entire time series of high-frequency Saildrone data.
"""

#---------------------------------------------------------------------
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import cf_units
#---------------------------------------------------------------------

def downsample(ds, freq_out, offset=np.timedelta64(0,'ns')):
    """Downsamples a dataset to a lower frequency, e.g., 20 Hz to 10 Hz.

    Inputs:
        ds [xarray dataset] -- input dataset with a 'time' coord. dimension
        freq [float] -- desired output frequency: must be less than input
        offset [numpy timedelta64] -- the start point: must be a multiple
                                      of the period corresponding to the 
                                      input dataset frequency.
                                      E.g., if input frequency is 20 Hz, 
                                      this could 0, 50, 100, ... milliseconds.

    Output:
        ds_sub [xarray dataset] -- the subsampled dataset: ds with some
                                   fraction of the timesteps excluded. 
    """
    # Set up time origin and divisor.
    time_origin = np.datetime64('2017-01-01').astype('<M8[ns]') + offset
    time_div = np.timedelta64(np.int(1000/freq_out), 'ms')
    
    # Get time in units of "divisors from origin",
    # i.e., array that is (time - time_origin)/time_div
    time_divs_from_origin = (ds.time.data - time_origin)/time_div
    # Work out which time steps are a whole number of divisors from origin.
    whole_divs = (np.around(time_divs_from_origin, decimals=1) -
                  np.around(time_divs_from_origin, decimals=0)) == 0.0
    
    # Subset the dataset.
    ds_sub = ds.sel(time=whole_divs)
    #
    return ds_sub


def wind_dir_twice(ds, u_name, v_name):
    """Calculates two wind directions: meteorological & mathematical.

    Meteorological means compass bearing from which the wind comes.
    Mathematical means angle anticlockwise from east of the direction
    TOWARDS which the wind blows.
    
    Inputs:
        ds [xarray dataset] -- dataset containing eastward and 
                               northward wind components.
        u_name, v_name [string] -- the names of the u and v components
                                   in ds (e.g., 'UWND', 'VWND')

    Output: 
        ds_out [xarray dataset] -- the same as ds but with two new varibles:
                                   'wind_dir' (meteorological)
                                   'arctan_VoverU' (mathematical)
    Both outputs are in DEGREES.
    """
    ds_out = ds.copy()
    ds_out['arctan_VoverU'] = \
        np.degrees(np.arctan2(ds[v_name],ds[u_name]))
    ds_out['wind_dir'] = np.mod(
        270.0 - np.degrees(np.arctan2(ds[v_name],ds[u_name])),
        360
    )
    # Add metadata.
    ds_out['arctan_VoverU'].attrs = {
        'units':'degrees',
        'long_name':'Mathematical convention wind direction: angle towards which the wind blows, anticlockwise from eastward.',
        'standard_name':'wind_towards_direction'
    }
    ds_out['wind_dir'].attrs = {
        'units':'degrees',
        'long_name':'Meteorological convention wind direction: compass bearing from which the wind comes.',
        'standard_name':'wind_from_direction'
    }
    #
    return ds_out


def rotation_xy(ds, u_name, v_name, angles, angle_convention):
    """Gives streamwise and cross-stream wind components in 'angles' coordinate.

    Inputs:
        ds [xarray dataset] -- dataset containing eastward and 
                               northward wind components.
        u_name, v_name [string] -- the names of the u and v components
                                   in ds (e.g., 'UWND', 'VWND')
        angles [numpy array or xarray data array] -- the angles of the 
                                                     new coordinate system
                                                     to rotate to. Must
                                                     broadcast with ds.
                                                     MUST BE IN RADIANS.
        angle_convention ['math' or 'met'] -- the convention used in the
                                              definition of angles.
            'math' -- towards, anticlockwise from E
            'met' -- from, clockwise from N

    Output:
        ds_out [xarray dataset] -- the same as ds but with two new varibles:
                                   'U_streamwise' and 'U_crossstream': 
                                   the components in the new coordinates.
    """
    # Convert the orientation of the new coordinate system.
    if angle_convention == 'met':
        A = np.radians(270.0) - angles
    elif angle_convention == 'math':
        A = angles
    else:
        sys.exit('ACHTUNG: angle_convention must be \'math\' or \'met\' in rotation_xy')
    
    # Make sure that the angle and original dataset broadcast.
    if (angles.shape == ()):
        cosA = np.cos(A)
        sinA = np.sin(A)
    elif (ds[u_name].shape == angles.shape):
        cosA = np.cos(A)
        sinA = np.sin(A)
    elif ((ds[u_name].shape[0] == angles.shape[0]) &
          (len(angles.shape) == 1)):
        cosA = np.expand_dims(np.cos(A), axis=1)
        sinA = np.expand_dims(np.sin(A), axis=1)
    elif ((ds[u_name].shape[1] == angles.shape[0]) &
          (len(angles.shape) == 1)):
        cosA = np.expand_dims(np.cos(A), axis=0)
        sinA = np.expand_dims(np.sin(A), axis=0)
    else:
        sys.exit('ACHTUNG: ds and angles are not compatible shapes in rotation_xy')
    
    # Do the rotation.
    ds_out = ds.copy()
    ds_out['U_streamwise'] = ds[u_name]*cosA + ds[v_name]*sinA
    ds_out['U_crossstream'] = -1.0*ds[u_name]*sinA + ds[v_name]*cosA
    
    # Add metadata.
    ds_out['U_streamwise'].attrs = {
        'units':ds[u_name].attrs['units'],
        'long_name':'Streamwise wind speed: in direction of mean wind.',
        'standard_name':'streamwise_wind_speed'
    }
    ds_out['U_crossstream'].attrs = {
        'units':ds[u_name].attrs['units'],
        'long_name':'Cross-stream wind speed: normal to mean wind in right-handed coordinate system.',
        'standard_name':'cross_stream_wind_speed'
    }
    return ds_out
    

def angle_diff_0_180(a1, a2):
    """Calculates the absolute value of the smallest angle between headings.

    a1 and a2 -- numpy arrays of angles: must be the same size.
    """
    #
    delta_a = np.abs((a1 - a2 +180.0) % 360 - 180.0)
    #
    return delta_a
    

def angle_diff_pm180(a1, a2):
    """Calculates the smallest angle between headings.

    a1 and a2 -- numpy arrays of angles: must be the same size.
    """
    #
    delta_a = (a1 - a2 +180.0) % 360 - 180.0
    #
    return delta_a


def multistep_smooth(ds, var_list, f_mid, f_out,
                     av_pres_time_bnds=None):
    """Smooth current data (e.g., at 10 minutes) to a higher frequency
       (e.g.,  10 Hz) via an intermediate frequency (e.g., 0.1 Hz).
       Preserves the mean. Can be VERY SLOW !!

    Inputs:
        ds [xarray dataset] -- (at lower frequency) with at least time and
                               the variables listed in var_list.
        var_list [list of strings] -- list of variable names for U and V
                                      components of currents.
        f_mid [numeric] -- intermediate frequency in Hertz.
        f_out [numeric] -- frequency of output in Hertz.
        av_pres_time_bnds -- not currently used.

    """
    #----- Preamble.
    
    # Convert time/frequency parameters.
    dt_in = ds.time.data[1] - ds.time.data[0]
    dt_mid = np.timedelta64(int(1000.0/f_mid), 'ms')
    dt_out = np.timedelta64(int(1000.0/f_out), 'ms')
    dt_mid_str = str(int(1000.0/f_mid)) + 'ms'
    dt_out_str = str(int(1000.0/f_out)) + 'ms'
    n_lo = len(ds.time)
    
    # Add dummy index added to make window selection easier.
    ds['dummy_index'] = xr.zeros_like(ds['LAT'])
    ds['dummy_index'].data = np.arange(len(ds['time'].data))
    
    #----- Low frequency to mid frequency using matrix method.

    # Resample.
    ds_mid = ds.resample(time=dt_mid_str).nearest()
    for vv in var_list:
        ds_mid[vv + '_sm'] = ds_mid[vv].copy(deep=True)

    # Create smoothing matrix.
    n_mid = len(ds_mid.time)
    sm_mat = (np.tri(n_mid, k=1) - np.tri(n_mid, k=-2))/3.0
    sm_mat[0, 0:2] = 0.5
    sm_mat[-1, -2:] = 0.5
    # Create averaging matrix.
    av_mat = np.zeros((n_mid, n_mid))
    for it in ds['dummy_index'].data:
        n_it = np.sum(ds_mid['dummy_index'].data == it)
        first_it = np.asarray(ds_mid['dummy_index'].data == it).nonzero()[0][0]
        last_it = np.asarray(ds_mid['dummy_index'].data == it).nonzero()[0][-1]
        av_mat[first_it:last_it+1, first_it:last_it+1] = \
            (1.0/n_it)*np.ones((n_it, n_it))
    
    # Repeat alternating smooth and average-correct steps.
    for ii in range(n_mid):
        if ii % 100 == 0: print(ii)
        # Smooth:
        for vv in var_list:
            ds_mid[vv + '_sm'].data = np.matmul(sm_mat, ds_mid[vv + '_sm'].data)
        # Average correct:
        for vv in var_list:
            ds_mid[vv + '_sm'].data = \
                ds_mid[vv + '_sm'].data \
                - np.matmul(av_mat, ds_mid[vv + '_sm'].data) \
                + np.matmul(av_mat, ds_mid[vv].data)
    
    #----- Mid frequency to high frequency using simpler interpolation.
    ds_out = ds_mid.resample(time=dt_out_str).nearest()
    ds_int = ds_mid.resample(time=dt_out_str).interpolate('linear')
    
    #----- Tidying up.
    #
    return ds_out, ds_int
