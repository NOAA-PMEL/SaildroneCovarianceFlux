"""Make scatter plots of flow tilt angle for all Saildrones.

Usage:
    $ python vertical_wind_flow_tilt_angle.py

Puts all drones on one figure. 
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
from SD_QC import *
from utils_time_chunk import *
from utils_timeseries import *
#------------------------------------------------------------------------------

def main():
    
    # List of saildrones.
    years = np.array([2017, 2017, 2018, 2018, 2018, 2018,
                      2019, 2019, 2019, 2019])
    drones = np.array([1005, 1006, 1005, 1006, 1029, 1030,
                       1066, 1067, 1068, 1069])
    row_list = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    col_list = np.array([0, 1, 0, 1, 2, 3, 0, 1, 2, 3])
    ds_list = []
    
    # Loop over saildrones and get data for each.
    for yr, sd in zip(years, drones):
        ds_1min = load_1min_data(str(yr), str(sd))
        # Exclude bad data from manual JSON file.
        ds_1min = exclude_json(ds_1min, str(yr), str(sd))
        # Change sign of vertical wind speed - should be upward, a
        # and it originally comes as downward for 2018 onwards.
        if yr > 2017:
            ds_1min['WWND_MEAN'] = -1.0*ds_1min['WWND_MEAN']
            ds_1min['WWND_MEAN'].attrs['long_name'] = \
                'Upward wind speed'
            ds_1min['WWND_MEAN'].attrs['standard_name'] = \
                'upward_air_velocity'
        ds_1min = vertical_angle(ds_1min,
                                 {'u':'UWND_MEAN', 'v':'VWND_MEAN',
                                  'w':'WWND_MEAN'})
        
        # Select sub-period with nearly continuous high-freq wind data.
        st_time = SD_mission_details.saildrone_dates\
            [str(yr)][str(sd)]['start'].astype('datetime64[D]')
        if 'end_anemometer' in SD_mission_details.saildrone_dates\
            [str(yr)][str(sd)].keys():
            end_time = SD_mission_details.saildrone_dates\
                [str(yr)][str(sd)]['end_anemometer'].astype('datetime64[D]')\
                + np.timedelta64(1,'D')
        else:
            end_time = SD_mission_details.saildrone_dates\
                [str(yr)][str(sd)]['end'].astype('datetime64[D]')\
                + np.timedelta64(1,'D')
        ds_1min = ds_1min.sel(time=((ds_1min.time >= st_time) &
                                    (ds_1min.time < end_time)))
        ds_list.append(ds_1min)
    
    # Plot scatter.
    plot_angle_scatter(ds_list, years, drones,
                       row_list, col_list,
                       x_var='ROLL')
    plot_w_scatter(ds_list, years, drones,
                   row_list, col_list,
                   x_var='ROLL')
    # 
    return


def vertical_angle(ds, comp_dict):
    """Calculate angle of 3D wind to horizontal.

    Inputs:
        ds [xarray dataset] -- must have wind variables as 
                               detailed in comp_dict
        comp_dict [python dictionary] -- dictionary of wind component variable
                                         names in ds. Must have:
            'w' -- vertical wind variable name;
            ('u' AND 'v') OR 'speed' -- horizontal wind speed component names.
    """
    ds_out = ds.copy()
    # Get wind speed.
    if 'speed' in comp_dict.keys():
        wind_speed = ds[comp_dict['speed']]
    else:
        wind_speed = np.sqrt((ds[comp_dict['u']])**2 + (ds[comp_dict['v']])**2)
        ds_out['wind_speed_hor'] = wind_speed
    # Calculate angle.
    ds_out['tilt_angle'] = np.degrees(np.arctan2(ds[comp_dict['w']],
                                                 wind_speed))
    #
    return ds_out


def plot_angle_scatter(ds_l, yrs, ids, rows, cols, x_var='PITCH'):
    
    # Get years and count for each - to arrange subplots.
    yrs_uniq, yr_counts = np.unique(yrs, return_counts=True)
    
    # Set up figure.
    fig, axs = plt.subplots(len(yrs_uniq), np.max(yr_counts),
                            sharex=True, sharey=True,
                            figsize=(10,6))
    
    # Loop over saildrones.
    for i_sd, sd in enumerate(ids):
        ax = axs[rows[i_sd], cols[i_sd]]
        ax.set_title(' ' + str(yrs[i_sd]) + '\n SD ' + str(sd),
                     loc='left', y=1.0, pad=-28)
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        sc = ax.scatter(ds_l[i_sd][x_var].data,
                        ds_l[i_sd]['tilt_angle'].data,
                        c=ds_l[i_sd]['wind_speed_hor'].data,
                        vmax=15.0, vmin=2.0, cmap='plasma',
                        s=0.25, marker='.', alpha=0.5)
        #ax.scatter(ds_l[i_sd][x_var].data[ds_l[i_sd]['wind_speed_hor'] < 4.0],
        #           ds_l[i_sd]['tilt_angle'].data[ds_l[i_sd]['wind_speed_hor']
        #                                         < 4.0],
        #           c='red', s=0.25, marker='.', alpha=0.5)
        ax.set_xlim(-30.0, 30.0)
        ax.set_ylim(-10.0, 10.0)
        bin_centers, bin_medians, bin_5p, bin_95p = \
            bin_tilts(ds_l[i_sd], x_var, bin_width=5.0, bin_min_count=20)
        ax.errorbar(bin_centers, bin_medians, yerr=[bin_medians - bin_5p,
                                                    bin_95p - bin_medians],
                    fmt='ko', ecolor='black', markersize=2)
        ax.plot([-30.0, 30.0], [0.0, 0.0],
                c='gray', linewidth=0.5, alpha=0.5)
        ax.plot([0.0, 0.0], [-10.0, 10.0],
                c='gray', linewidth=0.5, alpha=0.5)
    
    # Other plot details.
    for ax in axs[:,0].flatten():
        ax.set_ylabel('Flow tilt (degrees)')
    for ax in axs[-1,:].flatten():
        ax.set_xlabel(x_var + ' (' + ds_l[-1][x_var].attrs['units'] + 's)')
    for ax in axs.flatten():
        if not ax.lines: ax.set_visible(False)
    handles, labels = sc.legend_elements(prop="colors", num=5, alpha=0.5)
    legend = axs[0,1].legend(handles, labels,
                             title='Wind speed (' + r'$m~s^{-1}$' + ')',
                             loc="center left", bbox_to_anchor=(1, 0.5))
    
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_explore/MotionCorrection/' + \
        'vertical_wind_flow_tilt_angle.'
    #plot_file_format = 'pdf'
    #plt.savefig(plot_filename + plot_file_format, format=plot_file_format)
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format,
                format=plot_file_format, dpi=500)
    #
    return


def plot_w_scatter(ds_l, yrs, ids, rows, cols, x_var='PITCH'):
    
    # Get years and count for each - to arrange subplots.
    yrs_uniq, yr_counts = np.unique(yrs, return_counts=True)
    
    # Set up figure.
    fig, axs = plt.subplots(len(yrs_uniq), np.max(yr_counts),
                            sharex=True, sharey=True,
                            figsize=(10,6))
    
    # Loop over saildrones.
    for i_sd, sd in enumerate(ids):
        ax = axs[rows[i_sd], cols[i_sd]]
        ax.set_title(' ' + str(yrs[i_sd]) + '\n ' + str(sd),
                     loc='left', y=1.0, pad=-28)
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        sc = ax.scatter(ds_l[i_sd][x_var].data,
                        ds_l[i_sd]['WWND_MEAN'].data,
                        c=ds_l[i_sd]['wind_speed_hor'].data,
                        vmax=15.0, vmin=2.0, cmap='plasma',
                        s=0.25, marker='.', alpha=0.5)
        #ax.scatter(ds_l[i_sd][x_var].data[ds_l[i_sd]['wind_speed_hor'] < 4.0],
        #           ds_l[i_sd]['tilt_angle'].data[ds_l[i_sd]['wind_speed_hor']
        #                                         < 4.0],
        #           c='red', s=0.25, marker='.', alpha=0.5)
        ax.set_xlim(-30.0, 30.0)
        ax.set_ylim(-2.0, 2.0)
        bin_centers, bin_medians, bin_5p, bin_95p = \
            bin_quants(ds_l[i_sd], 'WWND_MEAN', x_var,
                       bin_width=5.0, bin_min_count=20)
        ax.errorbar(bin_centers, bin_medians, yerr=[bin_medians - bin_5p,
                                                    bin_95p - bin_medians],
                    fmt='ko', ecolor='black', markersize=2)
        ax.plot([-30.0, 30.0], [0.0, 0.0],
                c='gray', linewidth=0.5, alpha=0.5)
        ax.plot([0.0, 0.0], [-2.0, 2.0],
                c='gray', linewidth=0.5, alpha=0.5)
    
    # Other plot details.
    for ax in axs[:,0].flatten():
        ax.set_ylabel(r'$\mathit{w}$' + ' (' + r'$m~s^{-1}$' + ')')
    for ax in axs[-1,:].flatten():
        ax.set_xlabel(x_var + ' (' + ds_l[-1][x_var].attrs['units'] + 's)')
    for ax in axs.flatten():
        if not ax.lines: ax.set_visible(False)
    handles, labels = sc.legend_elements(prop="colors", num=5, alpha=0.5)
    legend = axs[0,1].legend(handles, labels,
                             title='Wind speed (' + r'$m~s^{-1}$' + ')',
                             loc="center left", bbox_to_anchor=(1, 0.5))
    
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_explore/MotionCorrection/' + \
        'vertical_wind_flow_tilt_speed.'
    #plot_file_format = 'pdf'
    #plt.savefig(plot_filename + plot_file_format, format=plot_file_format)
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format,
                format=plot_file_format, dpi=500)
    #
    return


def bin_tilts(ds, xvarname, bin_width=0.5, bin_min_count=20):
    bin_edges = np.arange(np.floor(ds[xvarname].min()),
                          np.ceil(ds[xvarname].max()) + 0.0001,
                          bin_width)
    bin_c = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_m = np.nan*np.zeros(len(bin_c))
    bin_5 = np.nan*np.zeros(len(bin_c))
    bin_95 = np.nan*np.zeros(len(bin_c))
    for i_b in range(len(bin_c)):
        quants = np.nanquantile(
            ds['tilt_angle'][(ds[xvarname] >= bin_edges[i_b]) &
                             (ds[xvarname] < bin_edges[i_b + 1])],
            np.array([0.05, 0.5, 0.95])
        )
        count = np.sum(~np.isnan(
            ds['tilt_angle'][(ds[xvarname] >= bin_edges[i_b]) &
                             (ds[xvarname] < bin_edges[i_b + 1])]
        ))
        if count >= bin_min_count:
            bin_5[i_b] = quants[0]
            bin_m[i_b] = quants[1]
            bin_95[i_b] = quants[2]
    #
    return bin_c, bin_m, bin_5, bin_95


def bin_quants(ds, yvarname, xvarname, bin_width=0.5, bin_min_count=20):
    bin_edges = np.arange(np.floor(ds[xvarname].min()),
                          np.ceil(ds[xvarname].max()) + 0.0001,
                          bin_width)
    bin_c = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_m = np.nan*np.zeros(len(bin_c))
    bin_5 = np.nan*np.zeros(len(bin_c))
    bin_95 = np.nan*np.zeros(len(bin_c))
    for i_b in range(len(bin_c)):
        quants = np.nanquantile(
            ds[yvarname][(ds[xvarname] >= bin_edges[i_b]) &
                         (ds[xvarname] < bin_edges[i_b + 1])],
            np.array([0.05, 0.5, 0.95])
        )
        count = np.sum(~np.isnan(
            ds[yvarname][(ds[xvarname] >= bin_edges[i_b]) &
                         (ds[xvarname] < bin_edges[i_b + 1])]
        ))
        if count >= bin_min_count:
            bin_5[i_b] = quants[0]
            bin_m[i_b] = quants[1]
            bin_95[i_b] = quants[2]
    #
    return bin_c, bin_m, bin_5, bin_95


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
