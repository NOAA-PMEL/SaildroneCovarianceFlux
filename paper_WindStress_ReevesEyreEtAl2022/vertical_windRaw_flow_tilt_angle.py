"""Make scatter plots of raw wind flow tilt angle for all Saildrones.

Usage:
    $ python vertical_windRaw_flow_tilt_angle.py

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
    #years = [2017]
    #drones = [1005]
    years = np.array([2017, 2017, 2018, 2018, 2018, 2018,
                      2019, 2019, 2019, 2019])
    drones = np.array([1005, 1006, 1005, 1006, 1029, 1030,
                       1066, 1067, 1068, 1069])
    row_list = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    col_list = np.array([0, 1, 0, 1, 2, 3, 0, 1, 2, 3])
    ds_list = []
    
    # Loop over saildrones and get data for each.
    for yr, sd in zip(years, drones):
        
        # ---- Main dataset.
        ds_1min = load_1min_data(str(yr), str(sd))
        if not qc_times_unique(ds_1min):
            ds_1min = qc_times_unique(ds_1min, fix=True)
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
        
        # ---- Raw wind data.
        ds_1min_raw = load_1min_raw_wind(str(yr), str(sd))
        ds_1min_recalc = load_1min_recalc_data(str(yr), str(sd))
        # Use the "recalc" data to mask out the "raw" data.
        ds_common = xr.merge([ds_1min_recalc, ds_1min_raw],
                             compat='override', join='inner')
        all_data_ok = ((~np.isnan(ds_common['UWND_MEAN'])) &
                       (~np.isnan(ds_common['VWND_MEAN'])) &
                       (~np.isnan(ds_common['WWND_MEAN'])))
        ds_common = ds_common.where(all_data_ok)
        ds_common['UV_UNCOR_WIND_SPEED'] = np.sqrt(
            (ds_common['UWND_UNCOR_MEAN']**2) +
            (ds_common['VWND_UNCOR_MEAN']**2)
        )
        ds_common['RELATIVE_WIND_DIR'] = np.degrees(
            np.arctan2(ds_common['VWND_UNCOR_MEAN'],
                       ds_common['UWND_UNCOR_MEAN'])
        )
        ds_common['W_UNCOR_TILT_ANGLE'] = np.degrees(
            np.arctan2(ds_common['WWND_UNCOR_MEAN'],
                       ds_common['UV_UNCOR_WIND_SPEED'])
        )
        
        # Merge the datasets.
        ds_1min = xr.merge([ds_1min,
                            ds_common[['RELATIVE_WIND_DIR',
                                       'W_UNCOR_TILT_ANGLE']]],
                           compat='override', join='left')
        
        # Add some metadata for plotting.
        ds_1min['RELATIVE_WIND_DIR'].attrs['axis_label'] = \
            'relative wind direction (degrees)'
        ds_1min['W_UNCOR_TILT_ANGLE'].attrs['axis_label'] = \
            'raw wind tilt angle (degrees)'
        ds_1min['tilt_angle'].attrs['axis_label'] = \
            'motion-corrected wind tilt angle (degrees)'
        
        # Limit the data to cut out low wind speeds.
        #ds_1min = ds_1min.where(ds_1min['wind_speed_hor'] >= 2.0)
        
        # Add the dataset to the list.
        ds_list.append(ds_1min)
        
        # Make plots for this drone.
        plot_one_hist(ds_1min, str(yr), str(sd))
        plot_one_tilt_angle_sensor(ds_1min, str(yr), str(sd))
        plot_one_tilt_angle(ds_1min, str(yr), str(sd))
        
    # Make all-drone plots.
    plot_angle_scatter_all(ds_list,
                           years, drones, row_list, col_list,
                           'RELATIVE_WIND_DIR', 'W_UNCOR_TILT_ANGLE',
                           'ROLL')
    plot_angle_scatter_all(ds_list,
                           years, drones, row_list, col_list,
                           'RELATIVE_WIND_DIR', 'tilt_angle',
                           'ROLL')
    #
    return


################################################################################

def plot_one_hist(ds, yr, ID):
    
    # Setup.
    fig, axs = plt.subplots(1, 1,
                            figsize=(7, 7))
    fig.suptitle(yr + ', Saildrone ' + ID)
    axs.tick_params(axis='x', which='both', bottom=True, top=True)
    axs.tick_params(axis='y', which='both', left=True, right=True)
    axs.set_xlabel('relative wind direction / degrees')
    axs.set_ylabel('count')
    axs.set_xlim(-180.0, 180.0)
    
    # Plot the data.
    axs.hist(ds['RELATIVE_WIND_DIR'].data,
             bins=72,
             density=False, histtype='step',
             log=True)
    
    # Save the figure.
    plot_filename = os.path.expanduser('~') + \
        '/Documents/plots/Saildrone_explore/MotionCorrection/' + \
        'vertical_windRaw_flow_tilt_angle_hist_' + yr + '_' + ID + '.png'
    plt.savefig(plot_filename, format='png', dpi=300)
    #
    return


def plot_one_tilt_angle_sensor(ds, yr, ID):
    
    # Setup.
    fig, axs = plt.subplots(1, 1,
                            figsize=(7, 7))
    fig.suptitle(yr + ', Saildrone ' + ID)
    axs.tick_params(axis='x', which='both', bottom=True, top=True)
    axs.tick_params(axis='y', which='both', left=True, right=True)
    axs.set_xlabel('relative wind direction / degrees')
    axs.set_ylabel(
        'arctan(' +
        r'$\frac{w_{sensor}}{\sqrt{u_{sensor}^{2} + v_{sensor}^{2}}}$' +
        ') / degrees')
    axs.set_ylim(-20.0,20.0)
    axs.set_xlim(-60.0, 60.0)
    
    # Plot the data.
    axs.scatter(ds['RELATIVE_WIND_DIR'].data,
                ds['W_UNCOR_TILT_ANGLE'].data,
                s=0.1)
    
    # Save the figure.
    plot_filename = os.path.expanduser('~') + \
        '/Documents/plots/Saildrone_explore/MotionCorrection/' + \
        'vertical_windRaw_flow_tilt_angle_sensor_scatter_' +\
        yr + '_' + ID + '.png'
    plt.savefig(plot_filename, format='png', dpi=300)
    #
    return


def plot_one_tilt_angle(ds, yr, ID):
    
    # Setup.
    fig, axs = plt.subplots(1, 1,
                            figsize=(7, 7))
    fig.suptitle(yr + ', Saildrone ' + ID)
    axs.tick_params(axis='x', which='both', bottom=True, top=True)
    axs.tick_params(axis='y', which='both', left=True, right=True)
    axs.set_xlabel('relative wind direction / degrees')
    axs.set_ylabel(
        'arctan(' +
        r'$\frac{w_{MC}}{\sqrt{u_{MC}^{2} + v_{MC}^{2}}}$' +
        ') / degrees')
    axs.set_ylim(-20.0,20.0)
    axs.set_xlim(-60.0, 60.0)
    
    # Plot the data.
    axs.scatter(ds['RELATIVE_WIND_DIR'].data,
                ds['tilt_angle'].data,
                s=0.1)
    
    # Save the figure.
    plot_filename = os.path.expanduser('~') + \
        '/Documents/plots/Saildrone_explore/MotionCorrection/' + \
        'vertical_windRaw_flow_tilt_angle_scatter_' +\
        yr + '_' + ID + '.png'
    plt.savefig(plot_filename, format='png', dpi=300)
    #
    return


def plot_angle_scatter_all(ds_l, yrs, ids, rows, cols, x_var, y_var, c_var):
    
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
        ax.set_xlim(-30.0, 30.0)
        ax.set_ylim(-20.0, 20.0)
        sc = ax.scatter(ds_l[i_sd][x_var].data,
                        ds_l[i_sd][y_var].data,
                        c=np.abs(ds_l[i_sd][c_var].data),
                        vmax=20.0, vmin=0.0, cmap='plasma',
                        s=0.2, marker='.', alpha=0.3)
        #bin_centers, bin_medians, bin_5p, bin_95p = \
        #    bin_tilts(ds_l[i_sd], x_var, bin_width=5.0, bin_min_count=20)
        #ax.errorbar(bin_centers, bin_medians, yerr=[bin_medians - bin_5p,
        #                                            bin_95p - bin_medians],
        #            fmt='ko', ecolor='black', markersize=2)
        ax.plot([-30.0, 30.0], [0.0, 0.0],
                c='gray', linewidth=0.5, alpha=0.5)
        ax.plot([0.0, 0.0], [-20.0, 20.0],
                c='gray', linewidth=0.5, alpha=0.5)
    
    # Other plot details.
    #for ax in axs[:,0].flatten():
    #    ax.set_ylabel(ds_l[-1][y_var].attrs['axis_label'])
    #for ax in axs[-1,:].flatten():
    #    ax.set_xlabel(ds_l[-1][x_var].attrs['axis_label'])
    axs[1,0].set_ylabel(ds_l[-1][y_var].attrs['axis_label'])
    axs[-1,1].set_xlabel(ds_l[-1][x_var].attrs['axis_label'])
    for ax in axs.flatten():
        if not ax.lines: ax.set_visible(False)
    handles, labels = sc.legend_elements(prop="colors", num=5, alpha=0.5)
    legend = axs[0,1].legend(handles, labels,
                             title=c_var + '  (' + 
                             ds_l[-1][c_var].attrs['units'] + ')',
                             loc="center left", bbox_to_anchor=(1, 0.5))
    
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_explore/MotionCorrection/' + \
        'vertical_windRaw_flow_tilt_angle_scatter_' + \
        y_var + '.'
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format,
                format=plot_file_format, dpi=500)
    #
    return
    
    
################################################################################

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


def load_1min_raw_wind(yr, ID):
    """Loads 1-minute raw wind data for year yr and Saildrone ID.

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_1min_from_10hz_1005_windRaw_20170902_20180505',
            '1006':'tpos_2017_1min_from_10hz_1006_windRaw_20170902_20180505'},
        '2018':{
            '1005':'tpos_2018_1min_from_10hz_1005_windRaw_20181003_20190127',
            '1006':'tpos_2018_1min_from_10hz_1006_windRaw_20181003_20190128',
            '1029':'tpos_2018_1min_from_10hz_1029_windRaw_20181004_20190128',
            '1030':'tpos_2018_1min_from_10hz_1030_windRaw_20181004_20190305'},
        '2019':{
            '1066':'tpos_2019_1min_from_20hz_1066_windRaw_20190626_20190916',
            '1067':'tpos_2019_1min_from_20hz_1067_windRaw_20190626_20191217',
            '1068':'tpos_2019_1min_from_20hz_1068_windRaw_20190625_20191212',
            '1069':'tpos_2019_1min_from_20hz_1069_windRaw_20190626_20191212'}
    }
    data_dir = config.data_dir + 'derived/lowfreq_from_highfreq/'
    file_suffix = '.nc'
    print('Loading file:')
    print(data_dir + filenames[yr][ID] + file_suffix)
    #
    result = xr.open_dataset(data_dir + filenames[yr][ID] + file_suffix)
    #
    if yr == '2019':
        result = result.rename(
            {'WIND_SENSOR_U_MEAN':'UWND_UNCOR_MEAN',
             'WIND_SENSOR_V_MEAN':'VWND_UNCOR_MEAN',
             'WIND_SENSOR_W_MEAN':'WWND_UNCOR_MEAN'}
        )
    #
    return result


def load_1min_recalc_data(yr, ID):
    """Loads recalculated 1-minute wind data for year yr and Saildrone ID.

    Both inputs should be strings.
    Output is an xarray dataset.
    """
    filenames = {
        '2017':{
            '1005':'tpos_2017_1min_from_10hz_1005_wind_20170904_20180430',
            '1006':'tpos_2017_1min_from_10hz_1006_wind_20170904_20180505'},
        '2018':{
            '1005':'tpos_2018_1min_from_10hz_1005_wind_20181006_20181106',
            '1006':'tpos_2018_1min_from_10hz_1006_wind_20181006_20190114',
            '1029':'tpos_2018_1min_from_10hz_1029_wind_20181009_20190108',
            '1030':'tpos_2018_1min_from_10hz_1030_wind_20181009_20190303'},
        '2019':{
            '1066':'tpos_2019_1min_from_20hz_1066_wind_20190626_20190916',
            '1067':'tpos_2019_1min_from_20hz_1067_wind_20190626_20191217',
            '1068':'tpos_2019_1min_from_20hz_1068_wind_20190625_20191212',
            '1069':'tpos_2019_1min_from_20hz_1069_wind_20190626_20191212'}
    }
    data_dir = config.data_dir + 'derived/lowfreq_from_highfreq/'
    file_suffix = '.nc'
    print('Loading file:')
    print(data_dir + filenames[yr][ID] + file_suffix)
    #
    result = xr.open_dataset(data_dir + filenames[yr][ID] + file_suffix)
    return result


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
