""" Prints out how much complete direct covariance data there is per mission.
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
    count_thresh = np.array([1.0, 0.99])
    count_total = np.zeros(len(count_thresh), dtype=np.int64)
    count_total_bulk = np.zeros(len(count_thresh), dtype=np.int64)
    count_total_curr = np.zeros(len(count_thresh), dtype=np.int64)
    count_total_wave = np.zeros(len(count_thresh), dtype=np.int64)
    
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
        ds_bulk = load_bulk_data(str(m[0]), str(m[1]))
        if str(m[0]) == '2019':
            ds_1min = load_1min_data(str(m[0]), str(m[1]))
            ds_wave = ds_1min[['WAVE_DOMINANT_PERIOD',
                               'WAVE_SIGNIFICANT_HEIGHT']]\
                               .dropna(dim='time', how='all')\
                               .resample(time='10min')\
                               .nearest(tolerance='20min')
            ds_join = xr.merge([ds_dir_cov, ds_bulk, ds_wave],
                               join='outer', compat='override')
        else:
            ds_join = xr.merge([ds_dir_cov, ds_bulk],
                               join='outer', compat='override')
        
        print('Number of ' + str(dir_cov_period) + '-minute periods: ' +
              str(len(ds_dir_cov['taux_count'].data)))
        
        for it, t in enumerate(count_thresh):
            
            min_count = t*dir_cov_period*60*freq_hz
            count_complete = np.sum((ds_dir_cov['taux_count'] >= min_count) &
                                    (ds_dir_cov['tauy_count'] >= min_count) &
                                    (~np.isnan(ds_dir_cov['taux'])) &
                                    (~np.isnan(ds_dir_cov['tauy'])))
            print(str(100*t) + '% complete: ' + str(count_complete.data))
            count_bulk = np.sum((ds_join['taux_count'] >= min_count) &
                                (ds_join['tauy_count'] >= min_count) &
                                (~np.isnan(ds_join['taux'])) &
                                (~np.isnan(ds_join['tauy'])) &
                                (~np.isnan(ds_join['TAU'])))
            print(str(100*t) + '% complete+bulk: ' + str(count_bulk.data))
            count_curr = np.sum((ds_join['taux_count'] >= min_count) &
                                (ds_join['tauy_count'] >= min_count) &
                                (~np.isnan(ds_join['taux'])) &
                                (~np.isnan(ds_join['tauy'])) &
                                (~np.isnan(ds_join['TAU'])) &
                                (~np.isnan(ds_join['UCUR10MIN'])) &
                                (~np.isnan(ds_join['VCUR10MIN'])))
            print(str(100*t) + '% complete+bulk+current: '
                  + str(count_curr.data))
            count_total[it] = count_total[it] + count_complete
            count_total_bulk[it] = count_total_bulk[it] + count_bulk
            count_total_curr[it] = count_total_curr[it] + count_curr
            if str(m[0]) == '2019':
                count_wave = np.sum(
                    (ds_join['taux_count'] >= min_count) &
                    (ds_join['tauy_count'] >= min_count) &
                    (~np.isnan(ds_join['taux'])) &
                    (~np.isnan(ds_join['tauy'])) &
                    (~np.isnan(ds_join['TAU'])) &
                    (~np.isnan(ds_join['UCUR10MIN'])) &
                    (~np.isnan(ds_join['VCUR10MIN'])) &
                    (~np.isnan(ds_join['WAVE_DOMINANT_PERIOD'])) &
                    (~np.isnan(ds_join['WAVE_SIGNIFICANT_HEIGHT']))
                )
                print(str(100*t) + '% complete+bulk+current+wave: '
                      + str(count_wave.data))
                count_total_wave[it] = count_total_wave[it] + count_wave
            
    
    print('\nTotal across missions:')
    for it, t in enumerate(count_thresh):
        print(str(100*t) + '% complete: ' + str(count_total[it])
              + ' = ' + str(count_total[it]*dir_cov_period/(24*60)) + ' days')
        print(str(100*t) + '% complete+bulk: ' + str(count_total_bulk[it])
              + ' = ' + str(count_total_bulk[it]*dir_cov_period/(24*60))
              + ' days')
        print(str(100*t) + '% complete+bulk+curr: ' + str(count_total_curr[it])
              + ' = ' + str(count_total_curr[it]*dir_cov_period/(24*60))
              + ' days')
        print(str(100*t) + '% complete+bulk+curr+wave: '
              + str(count_total_wave[it])
              + ' = ' + str(count_total_wave[it]*dir_cov_period/(24*60))
              + ' days')
    print('\n')
    #
    return
        

###########################################
# Now actually execute the script.
###########################################
if __name__ == '__main__':
    main()
