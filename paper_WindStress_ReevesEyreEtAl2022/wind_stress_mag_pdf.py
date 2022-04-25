"""Make PDF plots of wind stress magnitude ratios between different  methods.

Usage:
    $ python wind_stress_mag_pdf.py

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
        ds_bulk = ds_bulk.assign_coords(time=ds_bulk.time
                                        - np.timedelta64(5, 'm'))
        ds_DC = load_L2_data(str(yr), str(sd),
                             downsample=False, av_period=dir_cov_period)
        ds_DC = exclude_json(ds_DC, str(yr), str(sd))
        #ds_DC = ds_DC.assign_coords(time=ds_DC.time + np.timedelta64(5, 'm'))
        ds_DC_rel = load_L2_relative_data(str(yr), str(sd),
                                          downsample=False,
                                          av_period=dir_cov_period)
        ds_DC_rel = exclude_json(ds_DC_rel, str(yr), str(sd))
        for v in ds_DC_rel.data_vars.keys():
            ds_DC_rel = ds_DC_rel.rename({v:v+'_rel'})
        ds_both = xr.merge([ds_DC, ds_DC_rel, ds_bulk], join='inner')
        for v in ['taux', 'tauy', 'taus', 'tauc']:
            ds_both[v] = ds_both[v].where(
                ds_both[v + '_count'] >= 0.99*freq_hz*60*dir_cov_period,
                other=np.nan
            )
            ds_both[v+'_rel'] = ds_both[v+'_rel'].where(
                ds_both[v + '_count_rel'] >= 0.99*freq_hz*60*dir_cov_period,
                other=np.nan
            )
            """
            ds_both[v] = ds_both[v].where(
                np.abs(ds_both['latitude']) <= 2.0,
                other=np.nan
            )
            ds_both[v+'_rel'] = ds_both[v+'_rel'].where(
                np.abs(ds_both['latitude']) <= 2.0,
                other=np.nan
            )
            """
        
        # Add derived variables.
        ds_both['wind_speed'] = np.sqrt((ds_both['UWND']**2) +
                                        (ds_both['VWND']**2))
        ds_both['current_speed'] = np.sqrt((ds_both['UCUR10MIN_bulk']**2) +
                                           (ds_both['VCUR10MIN_bulk']**2))
        ds_both['tau_xy'] = np.sqrt((ds_both['taux']**2) +
                                    (ds_both['tauy']**2))
        ds_both['r_xy'] = np.abs(ds_both['taus_rel']/ds_both['tau_xy'])
        ds_both['w_xy'] = np.abs(ds_both['taus']/ds_both['tau_xy'])
        ds_both['r_xy_m_w_xy'] = ds_both['r_xy'] - ds_both['w_xy']
        
        # Subset dataset to where everything is available.
        subset_vars = ['tau_xy', 'taus', 'taus_rel']
        ds_both = ds_both.dropna('time', how='any', subset=subset_vars)
        
        # Add dataset to list.
        ds_list.append(ds_both)
        
    # Plot scatter plots.
    plot_mag_pdf(ds_list, years, drones,
                 4.0, 'wind_speed',
                 1.0, 'current_speed',
                 0.5, thresh_type='ratio')
    
    #
    return


def plot_mag_pdf(ds_l, yr_l, sd_l,
                 ws_max, ws_var, curr_min, curr_var,
                 cu_ratio_thresh, thresh_type='ratio'):
    
    # Gather all data up into single numpy arrays for PDFs.
    r_xy = np.nan*np.zeros(1)
    w_xy = np.nan*np.zeros(1)
    r_xy_m_w_xy = np.nan*np.zeros(1)
    for ds in ds_l:
        if thresh_type == 'ratio':
            thresh_bool = (((ds[curr_var]/ds[ws_var]) >= cu_ratio_thresh) &
                           (ds[curr_var] >= 0.25))
            leg_title = r'$\frac{|\mathbf{U_{curr}}|}{|\mathbf{U}|} \geq \ $' \
            + str(cu_ratio_thresh) \
            + ' & ' + r'$|\mathbf{U_{curr}}| \geq \mathrm{0.25\ m~s^{-1}}$' \
            + '\nN = '
        else:
            thresh_bool = ((ds[ws_var] <= ws_max) &
                           (ds[curr_var] >= curr_min))
            leg_title = r'$|\mathbf{U}| \leq \ $' + str(ws_max) \
            + r'$\ \mathrm{m~s^{-1}}$' \
            + '\n' \
            + r'$|\mathbf{U_{curr}}| \geq \ $' + str(curr_min) \
            + r'$\ \mathrm{m~s^{-1}}$' \
            + '\nN = ' 
        r_xy = np.concatenate((r_xy, ds['r_xy'].data[thresh_bool]),
                              axis=None)
        w_xy = np.concatenate((w_xy, ds['w_xy'].data[thresh_bool]),
                              axis=None)
        r_xy_m_w_xy = np.concatenate((r_xy_m_w_xy,
                                      ds['r_xy_m_w_xy'].data[thresh_bool]),
                                     axis=None)
    
    # Set up figure.
    fig, axs = plt.subplots(1, 1,
                            squeeze=False,
                            figsize=(5,5))
    
    # Plot details.
    axs[0,0].tick_params(axis='x', which='both', bottom=True, top=True)
    axs[0,0].tick_params(axis='y', which='both', left=True, right=True)
    axs[0,0].set_ylabel('count')
    axs[0,0].set_xlabel('ratio')
    axs[0,0].grid(alpha=0.5, linewidth=0.5)
        
    # Make the plot.
    mag_bins = np.arange(-1.0, 1.01, 0.05)
    axs[0,0].hist(r_xy, bins=mag_bins,
                  density=False, histtype='step',
                  color='#268bd2',
                  label=r'$\frac{|\tau_{x,rel}|}{|\mathbf{\tau}|}$' +
                  '     (mean = ' + "{:.3f}".format(np.nanmean(r_xy)) + ')')
    axs[0,0].hist(w_xy, bins=mag_bins,
                  density=False, histtype='step',
                  color='#cb4b16',
                  label=r'$\frac{|\tau_{x}|}{|\mathbf{\tau}|}$' +
                  '     (mean = ' + "{:.3f}".format(np.nanmean(w_xy)) + ')')
    axs[0,0].hist(r_xy_m_w_xy, bins=mag_bins,
                  density=False, histtype='stepfilled',
                  color='#859900', alpha=0.5,
                  label=r'$\frac{|\tau_{x,rel}|}{|\mathbf{\tau}|}| ' +
                  r'- \frac{|\tau_{x}|}{|\mathbf{\tau}|}$' +
                  '     (mean = ' +
                  "{:.3f}".format(np.nanmean(r_xy_m_w_xy)) + ')')
    axs[0,0].legend(loc='upper left',
                    title=leg_title + str(np.sum(~np.isnan(r_xy_m_w_xy))))
        
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_DirCov/low_wind_checks/' + \
        'wind_stress_mag_pdf.'
    plot_file_format = 'pdf'
    plt.savefig(plot_filename + plot_file_format, format=plot_file_format)
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format, format=plot_file_format)
    #
    return
    


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
