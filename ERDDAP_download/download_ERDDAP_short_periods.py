"""Download short chunks of Saildrone high-frequency data form ERDDAP.

Usage:
    $ python download_ERDDAP_short_periods.py 

NB: configured for Python 3.
"""

############################################################################################
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
############################################################################################

def main():
    
    # Get the list of dates for this year and drone.
    chunk_details = SD_chunks(0)
    
    # Loop over times and do the downloads.
    for cc in chunk_details:
        SD_download(cc['year'], cc['saildrone'],
                    cc['start_time'], cc['end_time'],
                    cc['var_list'])
    
    #
    return


############################################################################################
# Helper functions.

def SD_download(yr, SD, t1, t2, var_ls):
    
    # Define website and location to save files at.
    base_url = 'https://data.pmel.noaa.gov/pmel/erddap/tabledap/'
    data_dir = os.path.expanduser('~/Documents/Data/Saildrone/10Hz/')
    
    # Define query prefix.
    if str(yr) == '2017':
        query_ID = 'tpos_2017_hf_10hz'
    elif str(yr) == '2018':
        query_ID = 'tpos_2018_hf_10hz'
    elif str(yr) == '2019':
        query_ID = 'sd' + str(SD) + '_2019_20hz'
    else:
        sys.exit('ACHTUNG: year must be one of (2017, 2018, 2019)')
    
    # Define variables to download.
    # -----------------------------
    # Wind components in Earth coordinates.
    query_var ='%2C'.join(var_ls)
    
    # Format trajectory (Saildrone number) query.
    query_traj = 'trajectory=%22' + str(SD) + '.0%22'
    
    # Format times for the query.
    query_t1 = format_query_dt64(t1, 'ge')
    query_t2 = format_query_dt64(t2, 'le')
    
    # Construct query.
    query = base_url + query_ID + '.nc?' + \
        query_var + '&' + query_traj + '&' + \
        query_t1 + '&' + query_t2
    
    # Construct file name to save as.
    new_file_name = query_ID + '_' + str(SD) + \
        '_windMotion_forSpectra_' + \
        format_timestamp(t1,t2) + '.nc'
    
    # Download it.
    print('Downloading with query:')
    print(query)
    print('Saving as:')
    print(data_dir + new_file_name)
    #
    urllib.request.urlretrieve(query, data_dir + new_file_name)
    #
    print('--------------------------------------------------------')
    #
    return


def format_query_dt64(t, ineq):
    d1 = pd.to_datetime(t)
    ineq_dict = {'ge':'%3E=',
                 'le':'%3C=',
                 'gt':'%3E',
                 'lt':'%3C'}
    query_t1 = \
        'time' + ineq_dict[ineq] + \
        '{:04d}'.format(d1.year) + '-' + \
        '{:02d}'.format(d1.month) + '-' + \
        '{:02d}'.format(d1.day) + 'T' + \
        '{:02d}'.format(d1.hour) + '%3A' + \
        '{:02d}'.format(d1.minute) + '%3A' + \
        '{:04.1f}'.format(d1.second + (d1.microsecond/1000000))
    #
    return query_t1

def format_timestamp(t1, t2):
    d1 = pd.to_datetime(t1)
    d2 = pd.to_datetime(t2)
    timestamp = '{:04d}'.format(d1.year) + \
        '{:02d}'.format(d1.month) + \
        '{:02d}'.format(d1.day) + 'T' + \
        '{:02d}'.format(d1.hour) + \
        '{:02d}'.format(d1.minute) + \
        '_' + \
        '{:04d}'.format(d2.year) + \
        '{:02d}'.format(d2.month) + \
        '{:02d}'.format(d2.day) + 'T' + \
        '{:02d}'.format(d2.hour) + \
        '{:02d}'.format(d2.minute)
    #
    return timestamp

def save(url, filename):
    urllib.urlretrieve(url, filename, reporthook)

def SD_chunks(group):
    #
    if group == 0:
        common_vars_1718 = ['time', 'trajectory', 'latitude', 'longitude',
                            'UWND', 'VWND', 'WWND',
                            'UWND_UNCOR', 'VWND_UNCOR', 'WWND_UNCOR',
                            'ROLL', 'PITCH', 'HDG',
                            'ROLL_WING', 'PITCH_WING', 'HDG_WING', 'WING_ANGLE',
                            'INS_HULL_VEL_U', 'INS_WING_VEL_U']
        common_vars_19 = ['time', 'trajectory', 'latitude', 'longitude',
                          'UWND', 'VWND', 'WWND',
                          'WIND_SENSOR_U', 'WIND_SENSOR_V', 'WIND_SENSOR_W',
                          'ROLL', 'PITCH', 'HDG',
                          'ROLL_WING', 'PITCH_WING', 'HDG_WING', 'WING_ANGLE',
                          'INS_HULL_VEL_D', 'INS_WING_VEL_D']
        chunks = [
            {'year':'2019',
             'saildrone':'1069',
             'start_time':np.datetime64('2019-10-26T23:45:00.0','ns'),
             'end_time':np.datetime64('2019-10-27T06:15:00.0','ns'),
             'var_list':common_vars_19},
            {'year':'2019',
             'saildrone':'1068',
             'start_time':np.datetime64('2019-11-28T23:45:00.0','ns'),
             'end_time':np.datetime64('2019-11-29T06:15:00.0','ns'),
             'var_list':common_vars_19},
            {'year':'2019',
             'saildrone':'1067',
             'start_time':np.datetime64('2019-09-25T23:45:00.0','ns'),
             'end_time':np.datetime64('2019-09-26T06:15:00.0','ns'),
             'var_list':common_vars_19},
            {'year':'2019',
             'saildrone':'1068',
             'start_time':np.datetime64('2019-08-31T23:45:00.0','ns'),
             'end_time':np.datetime64('2019-09-01T06:15:00.0','ns'),
             'var_list':common_vars_19},
            {'year':'2019',
             'saildrone':'1066',
             'start_time':np.datetime64('2019-09-10T17:45:00.0','ns'),
             'end_time':np.datetime64('2019-09-11T00:15:00.0','ns'),
             'var_list':common_vars_19},
            {'year':'2019',
             'saildrone':'1069',
             'start_time':np.datetime64('2019-08-15T23:45:00.0','ns'),
             'end_time':np.datetime64('2019-08-16T06:15:00.0','ns'),
             'var_list':common_vars_19},
            {'year':'2017',
             'saildrone':'1006',
             'start_time':np.datetime64('2018-01-05T17:45:00.0','ns'),
             'end_time':np.datetime64('2018-01-06T00:15:00.0','ns'),
             'var_list':common_vars_1718},
            {'year':'2017',
             'saildrone':'1005',
             'start_time':np.datetime64('2017-12-14T17:45:00.0','ns'),
             'end_time':np.datetime64('2017-12-15T00:15:00.0','ns'),
             'var_list':common_vars_1718}
        ]
    else:
        sys.exit('ACHTUNG: group must be one of (0)')
    #
    return chunks



###########################################
# Now actually execute the script.
###########################################
if __name__ == '__main__':
    main()
