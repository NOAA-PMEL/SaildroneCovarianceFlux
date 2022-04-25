"""Plot polar diagram of winds, fluxes etc. with different reference direction.

Usage:
    $ python wind_stress_polar.py

Details are hardcoded (for now).

Fields plotted:
    1. line showing longitudinal wind stress component for every 5 degree
       polar direction bin (relative to tau_xy version)
    2. wind direction from bulk data
    3. wind direction from L1 data
    4. relative wind from L1 data
    5. wind stress from tau_xy calculation

"""

#------------------------------------------------------------------------------
import sys
import os
import warnings
#---- Analysis tools:
import numpy as np
import scipy.stats
import xarray as xr
import pandas as pd
import datetime
import cf_units
#---- Plotting tools:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import cm, gridspec, rcParams, colors
from matplotlib.patches import ConnectionPatch
#----
warnings.filterwarnings(action='ignore',
                        module='xarray', category=FutureWarning)
#---- Custom functions:
sys.path.append( os.path.expanduser('~') +
                 '/Documents/code/SaildroneCovarianceFlux/src')
import config
import SD_mission_details
from SD_IO import *
from SD_QC import *
from utils_time_chunk import *
from utils_timeseries import *
#------------------------------------------------------------------------------

def main():
    
    # -------------------- Pre-amble --------------------
    
    # Parse input command-line variables.
    argv = sys.argv
    assert len(argv) == 1,\
        'Command line arguments will be ignored!'
    SD_yr = '2019'
    SD_id = '1067'
    DC_period = 10
    DC_period_64 = np.timedelta64(DC_period, 'm')
    
    # -------------------- Define events --------------------
    
    center_time = np.datetime64('2019-11-02T13:50:00.0')
    
    # -------------------- Load data --------------------
    
    # Load bulk data.
    ds_bulk = load_bulk_data(str(SD_yr), str(SD_id))
    # Subset to event.
    ds_bulk = ds_bulk.sel(time=center_time)
    
    # Load L1 data.
    freq_hertz = SD_mission_details.\
        saildrone_frequency[SD_yr][SD_id]['freq_hertz']
    ds_L1 = load_L1_data(SD_yr, SD_id, downsample=False)
    ds_L1_long = ds_L1.sel(time=slice(center_time - 1.5*DC_period_64,
                                      center_time + 1.5*DC_period_64
                                      - np.timedelta64(25, 'ms')))
    # Prep wind data.
    if freq_hertz == 20:
            ds_L1_long = ds_L1_long.resample(time='50ms').\
                nearest(tolerance='1ms').\
                sel(time=slice(center_time - 1.5*DC_period_64,
                               center_time + 1.5*DC_period_64
                               - np.timedelta64(25, 'ms')))
    ds_L1_long = prep_L1_data(ds_L1_long, SD_yr)
    ds_L1 = ds_L1_long.sel(time=slice(center_time - 0.5*DC_period_64,
                                      center_time + 0.5*DC_period_64
                                      - np.timedelta64(25, 'ms')))
    
    # -------------------- Calculate wind directions --------------------
    
    wind_dirs = {}
    wind_mags = {}
    
    # From bulk data.
    wind_dirs['current'] = np.mod(
        270.0 - np.degrees(np.arctan2(ds_bulk['VCUR10MIN'],
                                      ds_bulk['UCUR10MIN'])),
        360
    )
    wind_mags['current'] = np.sqrt((ds_bulk['VCUR10MIN']**2) +
                                   (ds_bulk['UCUR10MIN']**2))
    
    # From L1 data.
    wind_dirs['wind'] = np.mod(
        270.0 - np.degrees(np.arctan2(ds_L1['VWND'].mean(),
                                      ds_L1['UWND'].mean())),
        360
    )
    wind_mags['wind'] = np.sqrt((ds_bulk['VWND']**2) +
                                (ds_bulk['UWND']**2))
    wind_dirs['relative'] = np.mod(
        270.0 - np.degrees(np.arctan2(ds_L1['VWND'].mean()
                                      - ds_bulk['VCUR10MIN'],
                                      ds_L1['UWND'].mean()
                                      - ds_bulk['UCUR10MIN'])),
        360
    )
    wind_mags['relative'] = np.sqrt(
        ((ds_L1['VWND'].mean() - ds_bulk['VCUR10MIN'])**2) +
        ((ds_L1['UWND'].mean() - ds_bulk['UCUR10MIN'])**2)
    )
    
    # -------------------- Calculate wind stress directions --------------------
    
    DC_period_len = freq_hertz*60*DC_period
    stress_dirs = {}
    
    # x-y coordinates.
    tau_xy = dir_cov_wrapper(ds_L1_long,
                             ['UWND', 'VWND'],
                             DC_period_len).sel(time=center_time)
    tau_xy_mag = np.sqrt((tau_xy['tau_UWND']**2) + (tau_xy['tau_VWND']**2))
    stress_dirs['xy'] = np.mod(
        270.0 - np.degrees(np.arctan2(tau_xy['tau_VWND'], tau_xy['tau_UWND'])),
        360
    )
    
    # All directions.
    stress_angles = np.arange(0.0, 359.0, 5.0)
    stress_mags_rel = np.nan*np.arange(0.0, 359.0, 5.0)
    for ia, aa in enumerate(stress_angles):
        ds_rot = rotation_xy(ds_L1_long, 'UWND', 'VWND',
                             np.radians(aa), 'met')
        tau_rot = dir_cov_wrapper(ds_rot,
                                  ['U_streamwise'],
                                  DC_period_len)
        stress_mags_rel[ia] = \
            tau_rot['tau_U_streamwise'].sel(time=center_time)\
            /tau_xy_mag
        
    # -------------------- Plot --------------------
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'},
                           figsize=(10, 6))
    # !!!!!
    # All wind directions are "from" so we need to subtract
    # 180 degrees (pi radians) from them.
    # !!!!!
    
    ax.plot(np.radians(stress_angles) - np.pi, np.abs(stress_mags_rel),
            color='black', linestyle=':', linewidth=0.5,
            label=r'$\frac{\tau_{\alpha}}{|\mathbf{\tau}|}$')
    
    ax.plot(np.array([0, np.radians(stress_dirs['xy'].data)]) - np.pi,
            np.array([0, 1.0]),
            color='black', linestyle='-', linewidth=1.0,
            label=r'$\frac{\tau_{xy}}{|\mathbf{\tau}|}$')
    
    ax.plot(np.array([np.radians(wind_dirs['current']),
                      np.radians(wind_dirs['wind'])]) - np.pi,
            np.array([wind_mags['current'], wind_mags['wind']]),
            color='#f58231', linestyle='-', linewidth=1.0,
            label=r'$\mathbf{U_{rel}}$' + ' (' + r'$\mathrm{m~s^{-1}}$' + ')')
    
    #ax.plot(np.array([0.0,
    #                  np.radians(wind_dirs['relative'])]) - np.pi,
    #        np.array([0.0, wind_mags['relative']]),
    #        color='#f58231', linestyle='-', linewidth=1.0)
    
    ax.plot(np.array([0, np.radians(wind_dirs['wind'])]) - np.pi,
            np.array([0, wind_mags['wind']]),
            color='#808000', linestyle='-', linewidth=1.0,
            label=r'$\mathbf{U}$' + ' (' + r'$\mathrm{m~s^{-1}}$' + ')')
    
    ax.plot(np.array([0, np.radians(wind_dirs['current'])]) - np.pi,
            np.array([0, wind_mags['current']]),
            color='#4363d8', linestyle='-', linewidth=1.0,
            label=r'$\mathbf{U_{curr}}$' + ' (' + r'$\mathrm{m~s^{-1}}$' + ')')
    
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(270.0)  # Move radial labels away from plotted line
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_title(SD_yr + ', SD ' + SD_id + ' | ' +
                 pd.to_datetime(str(center_time)).
                 strftime('%Y-%m-%d %H:%M') + ' | ' +
                 "{:.1f}".format(ds_bulk['LAT'].data) + r'$^{\circ}N\ $' +
                 "{:.1f}".format(-1.0*ds_bulk['LON'].data) + r'$^{\circ}W$',
                 va='bottom')
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.0, 1.0))
    
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_DirCov/low_wind_checks/' + \
        'wind_stress_polar_' + \
        str(SD_yr) + '_' + str(SD_id) + '_' + \
        pd.to_datetime(str(center_time)).strftime('%Y%m%d_%H%M') + \
        '.'
    plot_file_format = 'pdf'
    plt.savefig(plot_filename + plot_file_format,
                bbox_inches='tight', format=plot_file_format)
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format, 
                bbox_inches='tight', format=plot_file_format, dpi=500)
    #
    plt.close()
    #
    return 


def prep_L1_data(ds, yy):
    
    ds_out = ds.copy()
    
    if str(yy) in ['2019']:
        ds_out['WWND'].data = -1.0*ds_out['WWND'].data
        ds_out['WWND'].attrs['long_name'] = 'Upward wind speed'
        ds_out['WWND'].attrs['standard_name'] = 'upward_air_velocity'
    else:
        ds_out['WWND'].attrs['long_name'] = 'Upward wind speed'
        ds_out['WWND'].attrs['standard_name'] = \
            'upward_air_velocity'
    all_wind_ok = (
        ((ds_out['UWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_out['UWND_FLAG_PRIMARY'] == get_flag1('not_evaluated'))) &
        ((ds_out['VWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_out['VWND_FLAG_PRIMARY'] == get_flag1('not_evaluated'))) &
        ((ds_out['WWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_out['WWND_FLAG_PRIMARY'] == get_flag1('not_evaluated')))
    )
    for v in ['UWND', 'VWND', 'WWND']:
        ds_out[v] = ds_out[v].where(all_wind_ok, other=np.nan)
    
    #
    return ds_out


def dir_cov_wrapper(ds, var_list, DC_P):
    print('Calculating covariances...')
    
    # Reshape to 2D.
    ds_2D = reshape_1D_nD_numpy(
        ds, [np.int64(len(ds.time)/DC_P), DC_P]
    )
    
    # Detrend.
    ds_detrend, ds_trend, ds_slopes, ds_intercepts =  \
        ds_detrend_chunks(ds_2D, var_list)
    
    # Calculate anomalies.
    ds_detrend_mean = ds_detrend.mean(dim='sample_time',
                                      skipna=True, keep_attrs=True)
    ds_anom = ds_detrend - ds_detrend_mean
    
    # Calculate covariances. 
    ds_tau = xr.Dataset()
    ds_tau.coords['time'] = ds_2D.coords['time']
    ds_tau['time_first'] = ds_2D['time_first']
    ds_tau['time_last'] = ds_2D['time_last']
    for var in var_list:
        ds_tau['tau_' + var] = \
            -1.0*(ds_anom[var]*ds_anom['WWND']).mean(
                dim='sample_time', skipna=True, keep_attrs=True
            )
        ds_tau['tau_' + var + '_count'] = \
            (ds_anom[var]*ds_anom['WWND']).count(
                dim='sample_time', keep_attrs=True
            )
        if np.any(ds_tau['tau_' + var + '_count'] < 0.99*DC_P):
            print('Warning: check data count for tau_' + var)
            print(ds_tau['tau_' + var + '_count'].data)
        ds_tau[var] = ds_2D[var].mean(
            dim='sample_time', skipna=True, keep_attrs=True
        )
        ds_tau[var + '_count'] = ds_2D[var].count(
            dim='sample_time', keep_attrs=True
        )
        if np.any(ds_tau[var + '_count'] < 0.99*DC_P):
            print('Warning: check data count for ' + var)
            print(ds_tau[var + '_count'].data)
    
    # Sort out time coordinate to be middle of interval.
    ds_tau = ds_tau.assign_coords(time=(ds_tau.time +
                                        ds_2D.sample_time[int(DC_P/2)]))
    
    #
    return ds_tau

    
###########################################
# Now actually execute the script.
###########################################
if __name__ == '__main__':
    main()
