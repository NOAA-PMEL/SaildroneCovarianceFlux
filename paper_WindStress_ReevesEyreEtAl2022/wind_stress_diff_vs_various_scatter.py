"""Make scatter plots of wind stress (DC-bulk) diff vs. several explanatory vars

Usage:
    $ python wind_stress_diff_vs_various_scatter.py

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
    years = np.array([2017, 2017,
                      2019, 2019, 2019, 2019])
    drones = np.array([1005, 1006,
                       1066, 1067, 1068, 1069])
    ds_list = []
    
    # Loop over saildrones and get data for each.
    for yr, sd in zip(years, drones):
        
        # Get some details.
        dir_cov_period = 10 # minutes
        resample_period = str(dir_cov_period) + 'min'
        half_resample_period = str(int(dir_cov_period/2)) + 'min'
        freq_hz = SD_mission_details.\
            saildrone_frequency[str(yr)][str(sd)]['freq_hertz']
        
        # Combine bulk and DC datasets.
        ds_bulk = load_bulk_data(str(yr), str(sd))
        for v in ds_bulk.data_vars.keys():
            ds_bulk = ds_bulk.rename({v:v+'_bulk'})
        ds_DC = load_L2_data(str(yr), str(sd),
                             downsample=False, av_period=dir_cov_period)
        ds_DC = exclude_json(ds_DC, str(yr), str(sd))
        ds_DC = ds_DC.assign_coords(time=ds_DC.time + np.timedelta64(5, 'm'))
        for v in ['taux', 'tauy', 'taus', 'tauc']:
            ds_DC[v] = ds_DC[v].where(ds_DC[v + '_count']
                                      >= 0.99*freq_hz*60*dir_cov_period,
                                      other=np.nan)
        ds_both = xr.merge([ds_bulk, ds_DC], join='inner')
        
        # Add derived variables. 
        ds_both['tau_xy_bulk_diff'] = np.sqrt((ds_both['taux']**2) +
                                              (ds_both['tauy']**2)) \
                                      - np.abs(ds_both['TAU_bulk'])
        ds_both['tau_xy_bulk_diff'].attrs['plot_name'] = \
            'Wind stress difference: direct covariance minus bulk'
        ds_both['tau_xy_bulk_diff'].attrs['axis_name'] = \
            r'$|\tau_{DC,xy}| - |\tau_{bulk}|\ /\ N~m^{-2}$'
        ds_both['tau_xy_bulk_fracdiff'] = ds_both['tau_xy_bulk_diff']/\
            np.abs(ds_both['TAU_bulk'])
        ds_both['tau_xy_bulk_fracdiff'].attrs['plot_name'] = \
            'Wind stress fractional difference: direct covariance minus bulk'
        ds_both['tau_xy_bulk_fracdiff'].attrs['axis_name'] = \
            r'$|\tau_{DC,xy}| - |\tau_{bulk}|\ /\ |\tau_{bulk}|$'
        #
        ds_both['rel_wind_speed'] = np.sqrt(
            ((ds_both['UWND_bulk'] - ds_both['UCUR10MIN_bulk'])**2) +
            ((ds_both['VWND_bulk'] - ds_both['VCUR10MIN_bulk'])**2)
        )
        ds_both['rel_wind_speed'].attrs['plot_name'] = 'Relative wind speed'
        ds_both['rel_wind_speed'].attrs['axis_name'] = \
            r'$|\mathbf{U_{a}} - \mathbf{U_{s}}|\ /\ m~s^{-1}$'
        ds_both['zeta'] = 5.0/ds_both['MO_LENGTH_bulk']
        ds_both['zeta'].attrs['plot_name'] = 'dimensionless height'
        ds_both['zeta'].attrs['axis_name'] = \
            r'$\zeta = \frac{z}{L}$'
        
        # Add variables from 1-minute data.
        ds_1min = load_1min_data(str(yr), str(sd))
        if str(yr) == '2019':
            ds_both['sig_wave_height'] = ds_1min['WAVE_SIGNIFICANT_HEIGHT']\
                .dropna(dim='time', how='all')\
                .resample(time=resample_period)\
                .nearest(tolerance='20min')
            ds_both['sig_wave_height'].attrs['plot_name'] = \
                'Signicant wave height'
            ds_both['sig_wave_height'].attrs['axis_name'] = \
                r'$H_{sig}\ /\ m$'
        #
        if False:
            ds_both['heading_range'] = \
                ds_1min['HDG'].resample(time=resample_period,
                                        base=-1*int(dir_cov_period/2),
                                        loffset=half_resample_period).max() -\
                ds_1min['HDG'].resample(time=resample_period,
                                        base=-1*int(dir_cov_period/2),
                                        loffset=half_resample_period).min()
            ds_both['heading_range'].attrs['plot_name'] = \
                'Range of saildrone heading over flux averaging period'
            ds_both['heading_range'].attrs['axis_name'] = \
                'heading range (max - min) / degrees'
        if True:
            ds_both['wind_speed_range'] = \
                ds_1min['wind_speed'].resample(time=resample_period,
                                               base=-1*int(dir_cov_period/2),
                                               loffset=half_resample_period)\
                                     .max() -\
                ds_1min['wind_speed'].resample(time=resample_period,
                                               base=-1*int(dir_cov_period/2),
                                               loffset=half_resample_period)\
                                     .min()
            ds_both['wind_speed_range'].attrs['plot_name'] = \
                'Range of 1-minute wind speed \n over flux averaging period'
            ds_both['wind_speed_range'].attrs['axis_name'] = \
                'wind speed range (max - min) / ' + r'$m~s^{-1}$'
        if True:
            ds_both['temp_air_range'] = \
                ds_1min['TEMP_AIR_MEAN'].resample(time=resample_period,
                                               base=-1*int(dir_cov_period/2),
                                               loffset=half_resample_period)\
                                     .max() -\
                ds_1min['TEMP_AIR_MEAN'].resample(time=resample_period,
                                               base=-1*int(dir_cov_period/2),
                                               loffset=half_resample_period)\
                                     .min()
            ds_both['temp_air_range'].attrs['plot_name'] = \
                'Range of 1-minute air temp. \n over flux averaging period'
            ds_both['temp_air_range'].attrs['axis_name'] = \
                'air temp. range (max - min) / K'
        if True:
            SST_varname = {'2017':'TEMP_CTD_MEAN',
                           '2019':'TEMP_CTD_RBR_MEAN'}[str(yr)]
            ds_both['SST_range'] = \
                ds_1min[SST_varname].resample(time=resample_period,
                                              base=-1*int(dir_cov_period/2),
                                              loffset=half_resample_period)\
                                     .max() -\
                ds_1min[SST_varname].resample(time=resample_period,
                                              base=-1*int(dir_cov_period/2),
                                              loffset=half_resample_period)\
                                     .min()
            ds_both['SST_range'].attrs['plot_name'] = \
                'Range of 1-minute SST \n over flux averaging period'
            ds_both['SST_range'].attrs['axis_name'] = \
                'SST range (max - min) / K'
            
        # Subset dataset to where everything is available.
        subset_vars = ['tau_xy_bulk_diff', 'tau_xy_bulk_fracdiff',
                       'rel_wind_speed', 'SST_range',
                       'wind_speed_range', 'temp_air_range',
                       'zeta']
        if str(yr) == '2019':
            subset_vars.append('sig_wave_height')
        ds_both = ds_both.dropna('time', how='any', subset=subset_vars)
        
        # Add dataset to list.
        ds_list.append(ds_both)
        
    # Plot scatter plots.
    plot_diff_scatter(ds_list, years, drones,
                      'tau_xy_bulk_fracdiff',
                      ['rel_wind_speed', 'sig_wave_height',
                       'SST_range', 'temp_air_range',
                       'wind_speed_range', 'zeta'],
                      3, 2)
    
    #
    return


def plot_diff_scatter(ds_l, yr_l, sd_l,
                      y_varname, x_varname_l,
                      n_rows, n_cols):
    
    # Set up figure.
    fig, axs = plt.subplots(n_rows, n_cols,
                            sharex=False, sharey=True,
                            squeeze=False,
                            gridspec_kw={'hspace':0.5},
                            figsize=(10,10))
    
    # Loop over variables and set up axes.
    for iax, ax in enumerate(axs.flatten()):
        ax.tick_params(axis='x', which='both', bottom=True, top=True)
        ax.tick_params(axis='y', which='both', left=True, right=True)
        ax.set_xlabel(ds_l[-1][x_varname_l[iax]].attrs['axis_name'])
        ax.set_title(ds_l[-1][x_varname_l[iax]].attrs['plot_name'], loc='left')
        ax.set_ylim(-10, 10.0)
        
        # Loop over missions.
        N = 0
        for ids, ds in enumerate(ds_l):
            if str(yr_l[ids]) != '2019':
                continue
            ax.scatter(ds[x_varname_l[iax]].data,
                       ds[y_varname].data,
                       c='blue', s=0.4, marker='.', alpha=0.2)
            N = N + np.sum((~np.isnan(ds[x_varname_l[iax]].data)) &
                           (~np.isnan(ds[y_varname].data)))
            
        # Add a label for data count.
        ax.text(0.1, 0.9, 'N = ' + str(N),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        
    # Add ylabels for left hand figures only.
    for ax in axs[:,0].flatten():
        ax.set_ylabel(ds_l[-1][y_varname].attrs['axis_name'])
        
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_DirCov/DC_vs_bulk/' + \
        'wind_stress_diff_vs_various_scatter.'
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format, format=plot_file_format)
    #
    return
    


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
