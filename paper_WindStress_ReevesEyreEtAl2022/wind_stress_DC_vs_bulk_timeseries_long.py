"""Plot spectra of winds (motion-corrected and raw) and saildrone motion.

Usage:
    $ python wind_stress_DC_vs_bulk_timeseries_long.py year drone

where:
    year is the mission year [2017, 2018, 2019]
    drone is the saildrone ID [1066, 1067, ...]
"""

#------------------------------------------------------------------------------
import sys
import os
import warnings
#---- Analysis tools:
import numpy as np
import numpy.fft as fft
import scipy.stats
from scipy.signal import detrend
from scipy.signal.windows import hann, tukey
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
    
    # Load data.
    argv = sys.argv
    if len(argv) != 3:
        sys.exit("Incorrect number of command line arguments (2 required)")
    else:
        if str(argv[1]) == '2017':
            plot_date_range = np.arange('2017-09', '2018-05',
                                        dtype='datetime64[M]')
        elif str(argv[1]) == '2018':
            plot_date_range = np.arange('2018-09', '2019-04',
                                        dtype='datetime64[M]')
        elif str(argv[1]) == '2019':
            plot_date_range = np.arange('2019-07', '2020-01',
                                        dtype='datetime64[M]')
        else:
            sys.exit("year must be in {2017,2018,2019}")
    av_per = 10 # minutes
    ds_DC = load_L2_data(argv[1], argv[2], downsample=False, av_period=av_per)
    ds_bulk = load_bulk_data(argv[1], argv[2])
    
    # Postprocess the data.
    ds_DC = ds_DC.assign_coords(time=ds_DC.time +
                                np.timedelta64(int(av_per/2), 'm'))
    freq_hz = SD_mission_details.\
        saildrone_frequency[argv[1]][argv[2]]['freq_hertz']
    ds_DC['tau_mag'] = np.sqrt((ds_DC['taux']**2) +
                               (ds_DC['tauy']**2)).\
                               where(ds_DC['taux_count'] >=
                                     0.99*freq_hz*60*av_per,
                                     other=np.nan)
    
    # Plot the time series.
    plot_monthly_timeseries(ds_DC['tau_mag'], ds_bulk['TAU'],
                            plot_date_range,
                            argv[1],  argv[2])
    #
    return


def plot_monthly_timeseries(da_DC, da_bulk,
                            months,
                            yr, ID,
                            custom_title=''):
    """Plots timeseries for months - one plot per month.

    """
    # Increase all font sizes.
    matplotlib.rcParams.update({'font.size': 14})
    
    # Get variable-specific plot details.
    plot_dets = plot_details()
    
    # Set up axes.
    fig, axs = plt.subplots(len(months),1, sharex=False,
                            gridspec_kw={'hspace':0},
                            figsize=(10,1.5*len(months)))
    
    # Loop over months.
    for im, mm in enumerate(months):
        
        # Subset data.
        da_DC_m = da_DC.sel(time=slice(mm, mm+np.timedelta64(1,'M')))
        da_bulk_m = da_bulk.sel(time=slice(mm, mm+np.timedelta64(1,'M')))
        if ((len(da_DC_m.dropna(dim='time')) == 0) &
            (len(da_bulk_m.dropna(dim='time')) == 0)):
            continue
        
        # Plot the data.
        custom_cols = ['#002b36',
                       '#dc322f']
        axs[im].plot(da_DC_m.time.data,
                     da_DC_m.data,
                     color=custom_cols[0], linewidth=0.2,
                     label=r'$|\mathbf{\tau}|$')
        axs[im].plot(da_bulk_m.time.data,
                     da_bulk_m.data,
                     color=custom_cols[1], linewidth=0.2,
                     label=r'$\mathit{\tau_{bulk}}$')
        
        # Format axes.
        ylims = plot_dets['ylims']
        axs[im].set_ylim(ylims)
        axs[im].set_xlim(mm, mm+np.timedelta64(31,'D'))
        
        # Format tickmarks
        axs[im].tick_params(axis='x', which='both', bottom=True, top=True)
        axs[im].tick_params(axis='y', which='both', left=True, right=True)
        axs[im].xaxis.set_major_locator(
            mdates.DayLocator(bymonthday=range(1,32,5), tz=None)
        )
        axs[im].xaxis.set_minor_locator(
            mdates.DayLocator(bymonthday=range(1,32), tz=None)
        )
        axs[im].set_yticks(plot_dets['yticks_major'], minor=False)
        axs[im].set_yticks(plot_dets['yticks_minor'], minor=True)
        
        # Set tick labels.
        if im == len(months)-1:
            axs[im].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        else:
            axs[im].tick_params(axis='x', labelbottom=False, labeltop=False)
        if im%2 == 0:
            axs[im].tick_params(axis='y', labelleft=True, labelright=False)
        else:
             axs[im].tick_params(axis='y', labelleft=False, labelright=True)
        
        # Set titles.
        if im == len(months)-1:
            axs[im].set_xlabel('Day of month')
        if im == 2:
            axs[im].set_ylabel('wind stress magnitude (' +
                               r'$N~m^{-2}$' + ')')
        axs[im].text(mm+np.timedelta64(12,'h'),
                     ylims[0] + 0.8*(ylims[1]-ylims[0]),
                     str(mm),
                     bbox=dict(boxstyle='round',
                               edgecolor='black',
                               facecolor='white', alpha=0.5))
    # Overall plot details.
    axs[0].legend(loc='upper right')
    plt.suptitle(custom_title + '\n' +
                 yr + ', SD ' + ID,
                 x=0.1, ha='left')
    
    # Save out plot.
    plot_dir = os.path.expanduser('~/Documents/plots/Saildrone_DirCov/' +
                                  'DC_vs_bulk/')
    if custom_title != '':
        custom_title = '_' + custom_title
    plot_file_name = plot_dir + 'wind_stress_DC_vs_bulk_timeseries_long_' + \
        str(yr) + '_' + str(ID) + custom_title + '.'
    plot_format = 'pdf'
    plt.savefig(plot_file_name + plot_format,
                format=plot_format)
    plot_format = 'png'
    plt.savefig(plot_file_name + plot_format,
                format=plot_format)
    #
    return


def plot_details():
    if 1:
        out_dict = {
            'ylims':(0,0.8),
            'yticks_major':np.arange(0.0, 0.81, 0.2),
            'yticks_minor':np.arange(0.0, 0.81, 0.1)
        }
    #
    return out_dict


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
