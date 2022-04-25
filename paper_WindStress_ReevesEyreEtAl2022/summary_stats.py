""" Prints out summary stats of DC and bulk comparisons.
"""

#---------------------------------------------------------------------
import sys
import os
import math
import numpy as np
import scipy.stats
import xarray as xr
import pandas as pd
import datetime
import cf_units
import metpy
from metpy import calc
from metpy.units import units
#---- Plotting tools:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import cm, gridspec, rcParams, colors
#---- My code:
sys.path.append( os.path.expanduser('~') +
                 '/Documents/code/SaildroneCovarianceFlux/src')
import config
import SD_mission_details
from SD_IO import *
from SD_QC import *
from utils_time_chunk import *
from utils_timeseries import *
#---------------------------------------------------------------------

def main():
    
    missions = [
        ('2017', '1005'),
        ('2017', '1006'),
        ('2019', '1066'),
        ('2019', '1067'),
        ('2019', '1068'),
        ('2019', '1069')
    ]
    #    ('2018', '1005'),
    #    ('2018', '1006'),
    #    ('2018', '1029'),
    #    ('2018', '1030'),
    
    count_thresh = 0.99
    stress_types = ['tauxy', 'taus', 'taus_rel']
    count_total = 0.0
    total_sum_diff = np.zeros(len(stress_types), dtype=np.float64)
    total_sum_abs_diff = np.zeros(len(stress_types), dtype=np.float64)
    total_sum_sq_diff = np.zeros(len(stress_types), dtype=np.float64)
    
    for m in missions:
        
        print('\n' + m[0] + ', ' + m[1])
        
        dir_cov_period = 10 # minutes
        freq_hz = SD_mission_details.saildrone_frequency\
             [str(m[0])][str(m[1])]['freq_hertz']
        ds_dir_cov = load_L2_data(str(m[0]), str(m[1]),
                                  downsample=False, av_period=dir_cov_period)
        ds_dir_cov = exclude_json(ds_dir_cov, str(m[0]), str(m[1]))
        ds_dir_cov = ds_dir_cov.assign_coords(time=ds_dir_cov.time +
                                              np.timedelta64(5, 'm'))
        ds_dir_cov_rel = load_L2_relative_data(str(m[0]), str(m[1]),
                                               downsample=False,
                                               av_period=dir_cov_period)
        ds_dir_cov_rel = exclude_json(ds_dir_cov_rel, str(m[0]), str(m[1]))
        ds_dir_cov_rel = ds_dir_cov_rel.assign_coords(time=ds_dir_cov_rel.time
                                                      + np.timedelta64(5, 'm'))
        ds_dir_cov_rel = ds_dir_cov_rel.rename({'taus':'taus_rel',
                                                'tauc':'tauc_rel'})
        ds_bulk = load_bulk_data(str(m[0]), str(m[1]))
        ds_join = xr.merge([ds_dir_cov,
                            ds_dir_cov_rel[['taus_rel', 'tauc_rel']],
                            ds_bulk],
                           join='outer', compat='override')
        
        print('Number of ' + str(dir_cov_period) + '-minute periods: ' +
              str(len(ds_dir_cov['taux_count'].data)))
            
        # Count of valid data points. 
        min_count = count_thresh*dir_cov_period*60*freq_hz
        subset_bool = ((ds_join['taux_count'] >= min_count) &
                       (ds_join['tauy_count'] >= min_count) &
                       (~np.isnan(ds_join['taux'])) &
                       (~np.isnan(ds_join['tauy'])) &
                       (~np.isnan(ds_join['taus'])) &
                       (~np.isnan(ds_join['taus_rel'])) &
                       (~np.isnan(ds_join['TAU'])) &
                       (~np.isnan(ds_join['UCUR10MIN'])) &
                       (~np.isnan(ds_join['VCUR10MIN'])))
        count_complete = np.sum(subset_bool)
        print(str(100*count_thresh) + '% complete: ' + str(count_complete.data))
        count_total = count_total + count_complete.data
            
        # Calculate DC-bulk differences.
        ds_join['tauxy'] = np.sqrt((ds_join['taux']**2) + (ds_join['tauy']**2))
        for iv, v in enumerate(stress_types):
            print('-----' + v + '-----')
            ds_join[v + '_diff'] = ds_join[v] - ds_join['TAU']
            
            # Mean difference.
            diff_sum = ds_join[v + '_diff'].sel(time=subset_bool).sum()
            total_sum_diff[iv] = total_sum_diff[iv] + diff_sum
            print('sum/count = ' + str(diff_sum.data/count_complete.data))
            
            # RMS difference.
            diff_sq = (ds_join[v + '_diff'].sel(time=subset_bool)**2).sum()
            total_sum_sq_diff[iv] = total_sum_sq_diff[iv] + diff_sq
            print('RMSD = ' + str(np.sqrt(diff_sq.data/count_complete.data)))
            
            # Absolute difference.
            diff_abs = np.abs(ds_join[v + '_diff'].sel(time=subset_bool)).sum()
            total_sum_abs_diff[iv] = total_sum_abs_diff[iv] + diff_abs
            print('MAD = ' + str(diff_abs.data/count_complete.data))
            
    #
    print('\nTotal across missions:')
    for iv, v in enumerate(stress_types):
        print('-----' + v + '-----')
        print(str(100*count_thresh) + '% complete: ' + str(count_total)
              + ' = ' + str(count_total*dir_cov_period/(24*60)) + ' days')
        print('Mean diff = ' + str(total_sum_diff[iv]/count_total))
        print('RMSD = ' + str(np.sqrt(total_sum_sq_diff[iv]/count_total)))
        print('MAD = ' + str(total_sum_abs_diff[iv]/count_total))
    print('\n')
    #
    return
        

###########################################
# Now actually execute the script.
###########################################
if __name__ == '__main__':
    main()
