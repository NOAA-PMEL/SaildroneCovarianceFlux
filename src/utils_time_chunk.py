""" Utilities applied to individual chunks of high-frequency Saildrone data.

Chunks are usually 10 minutes to 1 hour.
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

def reshape_1D_nD_numpy(ds, new_shape):
    """Re-shapes all variables in a dataset 

    ...under some strict assumptions of their format (a "time" 
    coordinate dimension; complete, evenly spaced time steps;
    all variables the same original shape; length of 'ds' variables
    is equal to the product of sizes given in 'new_shape').
    
    The basic idea is to produce an output dataset that replaces a long 'time'
    array-shape with ('time' x 'sample_time').
    
    Uses numpy as it should be a lot quicker than xarray built-ins.
    """
    #
    # Reshape time and get first and last time stamps.
    time_new = ds.coords['time'].data.reshape(new_shape)
    time_first  = np.min(time_new,1)
    time_last  = np.max(time_new,1)
    sample_time = time_new[0,:] - time_new[0,0]
    #
    # Get other variables.
    data_dict = {}
    for v in list(ds.keys()):
        data_dict[v] = (['time', 'sample_time'], ds[v].data.reshape(new_shape))
    # Add time bounds variables.
    data_dict['time_first'] = (['time'],time_first)
    data_dict['time_last'] = (['time'],time_last)
    #
    # Construct output dataset.
    ds_out = xr.Dataset(data_dict,
                        coords={'time':time_first,
                                'sample_time':sample_time},
                        attrs=ds.attrs)
    # Add variable attributes.
    for v in list(ds.keys()):
        ds_out[v].attrs = ds[v].attrs
    #
    return ds_out


def detrend_chunks(Y):
    """Detrends a chunks of time series in a 2D array.
    
    Input:
         Y [xarray data array] -- 2D array with dimensions 
             (time x sample_time). The detrending is done
             along the sample_time dimension. 
    
    Output:
        Y_det [xarray data array] -- the detrended time series
        Y_hat [xarray data array] -- the trend that was subtracted
        slope [xarray data array] -- the slopes of the trendlines
        intercept [xarray data array] -- the intercepts of the trendlines

    Y_det and Y_hat have the same shape as Y. 
    slope and intercept are 1-dimensional with dimension time only. 
    """
    # Create the 2D sample_time data array.
    s_t_2D = xr.zeros_like(Y, dtype=Y.sample_time.data.dtype)
    s_t_2D.data = np.tile(Y.sample_time, (len(Y.time),1))
    s_t_2D = s_t_2D.where(~np.isnan(Y))
    # Create a numpy float64 version of this (in milliseconds).
    # This will be the explanatory variable.
    t = s_t_2D.data/np.timedelta64(1,'ms')
    
    # Calculate means (and add singleton dimension for broadcasting).
    Y_m = np.reshape(np.nanmean(Y,axis=1),
                    (Y.shape[0],1))
    t_m = np.reshape(np.nanmean(t,axis=1),
                    (Y.shape[0],1))
    
    # Calculate numerator and denominator for slope.
    a_num = np.reshape(np.nansum((Y-Y_m)*(t-t_m),axis=1),
                       (Y.shape[0],1))
    a_den = np.reshape(np.nansum((t-t_m)**2,axis=1),
                       (Y.shape[0],1))
    # Then calculate the slope (a) and intercept (b).
    a = a_num/a_den
    b = Y_m - a*t_m
    # Calculate estimate and subtract.
    Y_hat = xr.full_like(Y, fill_value=np.nan)
    Y_hat.data = b + a*t
    Y_det = Y - Y_hat
    
    # Construct output data arrays.
    slope = xr.DataArray(a[:,0],
                         coords=[Y.coords['time']],
                         dims=['time'],
                         attrs={'units':Y.attrs['units'] + ' millisecond-1',
                                'long_name':
                                'slope of simple linear regression of ' +
                                Y.attrs['long_name'] + ' vs. time'})
    intercept = xr.DataArray(b[:,0],
                             coords=[Y.coords['time']],
                             dims=['time'],
                             attrs={'units':Y.attrs['units'],
                                    'long_name':
                                    'intercept of simple linear ' +
                                    'regression of ' +
                                    Y.attrs['long_name'] + ' vs. time'})
    Y_hat.attrs = Y.attrs
    Y_hat.attrs['long_name'] = 'trend of ' \
        + Y.attrs['long_name'] \
        + ' estimated by simple linear regression' 
    Y_det.attrs['long_name'] = 'detrended ' \
        + Y.attrs['long_name']
    #
    return Y_det, Y_hat, slope, intercept


def ds_detrend_chunks(ds, var_names):
    """Wrapper to perform detrend_chunks on variables var_names in ds.
    """
    # Set up solution variables.
    ds_detrend = ds.copy()
    ds_trend = ds.copy()
    ds_slopes = np.sum(
        np.isnan(ds[var_names]),axis=1
    ).assign(time=ds.time)*np.nan
    ds_intercepts = ds_slopes.copy()
    # Loop over variables and do each.
    for v in var_names:
        ds_detrend[v], ds_trend[v], ds_slopes[v], ds_intercepts[v] = \
            detrend_chunks(ds[v])
    #
    return ds_detrend, ds_trend, ds_slopes, ds_intercepts


def reshape_2D_1D_numpy(ds):
    """Re-shapes all variables in a dataset to 1D

    ...under some strict assumptions of their format ("time"
    and "sample_time" coordinate dimensions).
    
    The basic idea is to do the opposite of reshape_1D_nD_numpy.
    
    Uses numpy as it should be a lot quicker than xarray built-ins.
    """
    #
    # Create and reshape time variables.
    time_2D = np.add(
        np.expand_dims(ds['time_first'].data, 1),
        np.expand_dims(ds.coords['sample_time'].data, 0)
    )
    time_1D = np.ravel(time_2D, order='C')
    #
    # Get other variables.
    data_dict = {}
    for v in list(ds.keys()):
        # Don't copy the time_first and time_last.
        if v in ['time_first', 'time_last']:
            continue
        # Copy & reshape everything else.
        data_dict[v] = (['time'], np.ravel(ds[v].data, order='C'))
    #
    # Construct output dataset.
    ds_out = xr.Dataset(data_dict,
                        coords={'time':time_1D},
                        attrs=ds.attrs)
    # Add variable attributes.
    for v in list(ds.keys()):
        if v in ['time_first', 'time_last']:
            continue
        ds_out[v].attrs = ds[v].attrs
    #
    return ds_out
