"""Make scatter plots of drag coefficient vs. wind speed.

Usage:
    $ python wind_speed_drag_coeff_scatter.py

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
    
    # Drag coefficient type -- 'stream' or 'mag'
    cd_type = 'mag'
    
    # List of saildrones.
    years = np.array([2017, 2017,
                      2019, 2019, 2019, 2019])
    drones = np.array([1005, 1006,
                       1066, 1067, 1068, 1069])
    row_list = np.array([0, 0, 0, 1, 1, 1])
    col_list = np.array([0, 1, 2, 0, 1, 2])
    ds_list = []
    
    # Loop over saildrones and get data for each.
    for yr, sd in zip(years, drones):
        # Combine datasets.
        ds_bulk = load_bulk_data(str(yr), str(sd))
        
        for v in ds_bulk.data_vars.keys():
            ds_bulk = ds_bulk.rename({v:v+'_bulk'})
        freq_hz = SD_mission_details.\
            saildrone_frequency[str(yr)][str(sd)]['freq_hertz']
        dir_cov_period = 10 # minutes
        ds_DC = load_L2_relative_data(str(yr), str(sd),
                                      downsample=False,
                                      av_period=dir_cov_period)
        ds_DC = exclude_json(ds_DC, str(yr), str(sd))
        ds_DC = ds_DC.assign_coords(time=ds_DC.time + np.timedelta64(5, 'm'))
        for v in ['taux', 'tauy', 'taus']:
            ds_DC[v] = ds_DC[v].where(ds_DC[v + '_count']
                                      >= 0.99*freq_hz*60*10,
                                      other=np.nan)
        ds_both = xr.merge([ds_bulk, ds_DC], join='inner')
        # Deal with NaNs.
        any_nan = (np.isnan(ds_both['UWND_bulk']))
        for v in ['taux', 'tauy', 'taus', 'UWND', 'VWND', 'air_density',
                  'USR_bulk', 'UWND_bulk', 'VWND_bulk',
                  'UCUR10MIN_bulk', 'VCUR10MIN_bulk']:
            any_nan = (any_nan | np.isnan(ds_both[v]))
        for v in ['taux', 'tauy', 'taus', 'UWND', 'VWND', 'air_density',
                  'USR_bulk', 'UWND_bulk', 'VWND_bulk',
                  'UCUR10MIN_bulk', 'VCUR10MIN_bulk']:
            ds_both[v].data[any_nan] = np.nan
        print('Available data: ' + str(np.sum(~np.isnan(ds_both['taux'].data))))
        # Calculate some new variables.
        ds_both['wind_speed_bulk'] = np.sqrt(
            ((ds_both['UWND_bulk'] - ds_both['UCUR10MIN_bulk'])**2) +
            ((ds_both['VWND_bulk'] - ds_both['VCUR10MIN_bulk'])**2)
        )
        ds_both['drag_coefficient_bulk'] = (ds_both['USR_bulk']**2)/\
            (ds_both['wind_speed_bulk']**2)
        ds_both['wind_speed'] = np.sqrt(
            ((ds_both['UWND'] - ds_both['UCUR10MIN_bulk'])**2) +
            ((ds_both['VWND'] - ds_both['VCUR10MIN_bulk'])**2)
        )
        if  cd_type == 'mag':
            ds_both['drag_coefficient'] = \
                np.sqrt((ds_both['taux']**2) + (ds_both['tauy']**2))/\
                        (ds_both['air_density']*(ds_both['wind_speed']**2))
        elif cd_type == 'stream':
            ds_both['drag_coefficient'] = \
                ds_both['taus']/\
                (ds_both['air_density']*(ds_both['wind_speed']**2))
        else:
            sys.exit('cd_type must be one of {stream, mag}.')
        # Add dataset to list.
        ds_list.append(ds_both)
    
    # Plot scatter.
    plot_drag_scatter(ds_list, years, drones,
                      row_list, col_list, cd_type)
    # 
    return


def plot_drag_scatter(ds_l, yrs, ids, rows, cols, cdt):
    
    # Set up figure.
    fig, axs = plt.subplots(1, 1,
                            sharex=True, sharey=True,
                            figsize=(7,6))
    
    # Join all the datasets together for plotting on a single plot.
    CD_DC = ds_l[0]['drag_coefficient']
    CD_bulk = ds_l[0]['drag_coefficient_bulk']
    U_DC = ds_l[0]['wind_speed']
    U_bulk = ds_l[0]['wind_speed_bulk']
    for i in range(1,len(yrs)):
        CD_DC = np.concatenate((CD_DC, ds_l[i]['drag_coefficient']),
                               axis=None)
        CD_bulk = np.concatenate((CD_bulk, ds_l[i]['drag_coefficient_bulk']),
                                 axis=None)
        U_DC = np.concatenate((U_DC, ds_l[i]['wind_speed']),
                              axis=None)
        U_bulk = np.concatenate((U_bulk, ds_l[i]['wind_speed_bulk']),
                                axis=None)
    bulk_centers, bulk_medians, bulk_5p, bulk_95p = \
        bin_stats(U_bulk, CD_bulk, bin_width=1.0, bin_min_count=20)
    DC_centers, DC_medians, DC_5p, DC_95p = \
        bin_stats(U_DC, CD_DC, bin_width=1.0, bin_min_count=20)
    
    # Make the plot.
    if cdt == 'mag':
        axs.set_ylim(-0.001, 0.01)
    elif cdt == 'stream':
        axs.set_ylim(-0.005, 0.01)
    else:
        sys.exit('cdt must be one of {stream, mag}.')
    axs.set_xlim(0.0, 15.0)
    axs.tick_params(axis='x', which='both', bottom=True, top=True)
    axs.tick_params(axis='y', which='both', left=True, right=True)
    axs.set_xlabel(r'$|\mathbf{U_{rel}}|$' + ' (' +
                   r'$m~s^{-1}$' + ')')
    if cdt == 'mag':
        axs.set_ylabel(r'$\mathit{C_{D,mag}}$')
    elif cdt == 'stream':
        axs.set_ylabel(r'$\mathit{C_{D,stream}}$')
    else:
        sys.exit('cdt must be one of {stream, mag}.')
    #
    # Plot data.
    sc = axs.scatter(U_DC, CD_DC,
                     c='blue',
                     s=0.4, marker='.', alpha=0.2,
                     label='N = ' + str(np.sum(~np.isnan(CD_DC))))
    axs.errorbar(bulk_centers, bulk_medians, yerr=[bulk_medians - bulk_5p,
                                                   bulk_95p - bulk_medians],
                 fmt='kx', ecolor='black', label='bulk')
    axs.errorbar(DC_centers+0.2, DC_medians, yerr=[DC_medians - DC_5p,
                                                   DC_95p - DC_medians],
                 fmt='rx', ecolor='red', label='direct cov.')
    axs.legend(loc='upper right', frameon=False)
    
    # Save out figure.
    plot_filename = config.plot_dir + \
        'Saildrone_DirCov/' + \
        'wind_speed_drag_coeff_scatter_' + cdt + '.'
    #plot_file_format = 'pdf'
    #plt.savefig(plot_filename + plot_file_format, format=plot_file_format)
    plot_file_format = 'png'
    plt.savefig(plot_filename + plot_file_format, format=plot_file_format,
                dpi=500)
    #
    return


def bin_stats(x, y, bin_width=1.0, bin_min_count=20):
    bin_edges = np.arange(1.0, np.ceil(np.nanmax(x)) + 0.0001,
                          bin_width)
    bin_c = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_m = np.nan*np.zeros(len(bin_c))
    bin_5 = np.nan*np.zeros(len(bin_c))
    bin_95 = np.nan*np.zeros(len(bin_c))
    for i_b in range(len(bin_c)):
        quants = np.nanquantile(
            y[(x >= bin_edges[i_b]) &
              (x < bin_edges[i_b + 1])],
            np.array([0.05, 0.5, 0.95])
        )
        count = np.sum(~np.isnan(
            y[(x >= bin_edges[i_b]) &
              (x < bin_edges[i_b + 1])]
        ))
        if count >= bin_min_count:
            bin_5[i_b] = quants[0]
            bin_m[i_b] = quants[1]
            bin_95[i_b] = quants[2]
    #
    return bin_c, bin_m, bin_5, bin_95
        
    
def nan_merge(ds1, ds2):
    ds_both = xr.merge([ds1, ds2],
                       join='inner')
    any_nan = ((ds_both['drag_coefficient_bulk'] == np.nan) |
               (ds_both['drag_coefficient'] == np.nan) |
               (ds_both['wind_speed_bulk'] == np.nan) |
               (ds_both['wind_speed'] == np.nan))
    ds_both['drag_coefficient'].data[any_nan] = np.nan
    ds_both['drag_coefficient_bulk'].data[any_nan] = np.nan
    ds_both['wind_speed'].data[any_nan] = np.nan
    ds_both['wind_speed_bulk'].data[any_nan] = np.nan
    #
    return ds_both
                       
                       

################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
