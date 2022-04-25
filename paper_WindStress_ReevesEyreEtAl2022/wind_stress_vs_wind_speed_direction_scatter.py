"""Scatter plots of wind stress magnitude and directions from different methods.

Usage:
    $ python wind_stress_vs_wind_speed_direction_scatter.py year drone
where 
    year is in (2017, 2018, 2019)
    drone is in (1005, 1006, etc.)
"""

#------------------------------------------------------------------------------
import sys
import os
import warnings
#---- Analysis tools:
import numpy as np
import scipy.stats
from scipy.signal import detrend
from scipy.signal.windows import hann
from scipy.fft import fft
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
matplotlib.rcParams['mathtext.default'] = 'regular'
#----
warnings.filterwarnings(action='ignore',
                        module='xarray', category=FutureWarning)
#---- Custom functions:
sys.path.append( os.path.expanduser('~') +
                 '/Documents/code/SaildroneCovarianceFlux/src')
import config
import SD_mission_details
from SD_IO import *
from utils_timeseries import *
g_const = 9.81
#------------------------------------------------------------------------------


def main():
    
    # Get command line arguments.
    argv = sys.argv
    SD_yr = argv[1]
    SD_id = argv[2]
    
    # Load the data and details.
    DC_period = 10
    DC_period_64 = np.timedelta64(DC_period, 'm')
    ds_ground = load_L2_data(SD_yr, SD_id,
                             downsample=False, av_period=DC_period)
    ds_ground = exclude_json(ds_ground, str(SD_yr), str(SD_id))
    ds_ground = ds_ground.assign_coords(time=ds_ground.time
                                        + np.timedelta64(5, 'm'))
    ds_bulk = load_bulk_data(SD_yr, SD_id)
    ds_ground['UWND_rel'] = ds_bulk['UWND'] - ds_bulk['UCUR10MIN']
    ds_ground['VWND_rel'] = ds_bulk['VWND'] - ds_bulk['VCUR10MIN']
    freq_hertz = SD_mission_details.\
        saildrone_frequency[SD_yr][SD_id]['freq_hertz']
    
    # Remove time steps without enough underlying data.
    min_count = DC_period*60*freq_hertz*0.99
    all_wind_ok = (
        (ds_ground['taux_count'] >= min_count) &
        (ds_ground['tauy_count'] >= min_count) &
        (~np.isnan(ds_ground['UWND_rel'])) &
        (~np.isnan(ds_ground['VWND_rel']))
    )
    for v in ['UWND', 'VWND', 'taux', 'tauy', 'UWND_rel', 'VWND_rel']:
        ds_ground[v] = ds_ground[v].where(all_wind_ok, other=np.nan)
    
    # Calculate the required quantities.
    tau_dirs = wind_dir_twice(ds_ground, 'taux', 'tauy')
    wnd_dirs = wind_dir_twice(ds_ground, 'UWND' ,'VWND')
    wnd_rel_dirs = wind_dir_twice(ds_ground, 'UWND_rel' ,'VWND_rel')
    dir_diff = angle_diff_0_180(tau_dirs['wind_dir'],
                                wnd_dirs['wind_dir'])
    dir_rel_diff = angle_diff_0_180(tau_dirs['wind_dir'],
                                    wnd_rel_dirs['wind_dir'])
    wnd_speed = np.sqrt(ds_ground['UWND']**2 + ds_ground['VWND']**2)
    wnd_rel_speed = np.sqrt(ds_ground['UWND_rel']**2 + ds_ground['VWND_rel']**2)
    
    # Get wave data for 2019
    if str(SD_yr) == '2019':
        ds_all = load_1min_data(argv[1], argv[2])
        P_wave = ds_all['WAVE_DOMINANT_PERIOD']\
            .dropna('time', how='any')\
            .resample(time='10min').nearest(tolerance='20min')\
            .sel(time=ds_ground.time)
        c_p = g_const*P_wave/(2.0*np.pi)
        wave_age = xr.where(wnd_speed >= 0.01, c_p/wnd_speed, np.nan)
        H_s = ds_all['WAVE_SIGNIFICANT_HEIGHT']\
            .dropna('time', how='any')\
            .resample(time='10min').nearest(tolerance='20min')\
            .sel(time=ds_ground.time)
    
    # Make the plots.
    fig, axs = plt.subplots(1,2, figsize=(10,8),
                            sharey=True)
    fig.suptitle(str(SD_yr) + ', SD ' + str(SD_id))
    
    # Make ground-relative wind direction plots.
    sc = axs[0].scatter(wnd_rel_speed, dir_rel_diff,
                        marker='.',
                        s=wave_age.data, #(H_s.data**4)/4.0,
                        c=H_s.data, #wave_age.data,
                        vmax=3.5, vmin=1.0,
                        cmap='RdBu_r',
                        alpha=0.5)
    wspd_bins = np.arange(0.0, wnd_speed.max() + 2.0, 2.0)
    dir_bins = np.arange(0.0, 181.0, 2.5)
    custom_cols = ['#b58900', '#d33682',
                   '#cb4b16', '#6c71c4',
                   '#dc322f', '#268bd2',
                   '#2aa198']
    for iw in range(len(wspd_bins) - 1):
        iw_bool = ((wnd_speed >= wspd_bins[iw]) &
                   (wnd_speed < wspd_bins[iw+1]))
        wspd_lower_str = "{:.0f}".format(wspd_bins[iw])
        wspd_upper_str = "{:.0f}".format(wspd_bins[iw+1])
        leg_label = wspd_lower_str + r'$\ \leqslant |U| < \ $' + wspd_upper_str
        axs[1].hist(dir_diff.data[iw_bool], bins=dir_bins,
                    density=True, histtype='step',
                    orientation='horizontal', cumulative=True,
                    color=custom_cols[iw%len(custom_cols)],
                    label=leg_label)
    
    # Sort axes.
    for ax in axs:
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        ax.grid()
    axs[0].set_yticks(np.arange(0.0, 181.0, 22.5))
    axs[0].set_ylabel(r'$|\mathit{\phi(\mathbf{\tau},\mathbf{U_{rel}})}|$' +
                      ' (degrees)')
    axs[0].set_xlabel(r'$|\mathbf{U_{rel}}|$' + ' (' +
                      r'$m~s^{-1}$' + ')')
    axs[1].set_xlabel('Cumulative probability density')
    # Draw legends.
    axs[1].legend(loc='upper left') 
    # produce a legend with a cross section of sizes/colors from the scatter
    handles, labels = sc.legend_elements(prop="sizes", num=4, alpha=0.5)
    legend1 = axs[0].legend(handles, labels,
                            loc="upper right", title="Size =\nWave age")
    axs[0].add_artist(legend1)
    handles2, labels2 = sc.legend_elements(prop="colors", alpha=0.5)
    legend2 = axs[0].legend(handles2, labels2,
                            loc="center right", title="Sig. wave\nheight (m)")
    
    # Show or save the figures.
    #plt.show()
    figure_fnm = config.plot_dir + \
                 'Saildrone_DirCov/low_wind_checks/' + \
                 'wind_stress_vs_wind_speed_direction_scatter_' + \
                 str(argv[1]) + '_' + str(argv[2]) + \
                 '.'
    file_format = 'pdf'
    plt.savefig(figure_fnm + file_format,
                format=file_format)
    file_format = 'png'
    plt.savefig(figure_fnm + file_format,
                format=file_format, dpi=500)
    #
    return


#-------------------------------------------------------------------------------

def angle_diff_0_180(a1, a2):
    """Calculates the absolute value of the smallest angle between headings.

    a1 and a2 -- xarrays of angles in degrees.
    """
    #
    diff = a1 - a2
    delta_a = np.abs((diff +180.0) % 360 - 180.0)
    #
    return delta_a


def load_L2_water_relative(yr, ID):
    """Loads a flux dataset for year yr and Saildrone ID.

    Includes streamwise wind stress calculated using mean 
    water-relative winds.

    Inputs yr and ID should be strings. downsample is boolean
    indicating whether to use the 20 Hz (downsample=False) or 
    10 Hz (downsample=True) data for 2019.
    Output is an xarray dataset (with QC flags).
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_hf_10hz_1005_uvw_20170904_20180430_30min_fluxes_L2_relative_wind_direction',
            '1006':'tpos_2017_hf_10hz_1006_uvw_20170904_20180505_30min_fluxes_L2_relative_wind_direction'
        }
    }
    data_dir = config.data_dir + '10Hz/'
    file_suffix = '.nc'
    fn = data_dir + filenames[str(yr)][str(ID)] + file_suffix
    #
    print('Loading file:')
    print(fn)
    #
    result = xr.open_dataset(fn)
    return result
    
#-------------------------------------------------------------------------------

###########################################
# Now actually execute the script.
###########################################
if __name__ == '__main__':
    main()
