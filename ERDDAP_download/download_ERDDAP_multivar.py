"""Download Saildrone high-frequency shortwave data form ERDDAP.

Usage:
    $ python download_ERDDAP_multivar.py year drone vars
where
    year is the mission year (2018, 2019)
    drone is the Saildrone ID (1005, 1006, etc.)
    vars is the descriptor of the required variables; one of:
        'uvw', 'uvwRaw', 'Temp_WindSt', 'SWrad'

NB: configured for Python 3.
"""

############################################################################################
import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
############################################################################################

def main():
    
    # Get the list of dates for this year and drone.
    argv = sys.argv
    assert len(argv)==4,\
        'Incorrect number of command line arguments (3 required)'
    saildrone_yr = argv[1]
    saildrone_ID = argv[2]
    var_name = argv[3]
    file_dates = SD_dates_monthly_chunks(saildrone_yr, saildrone_ID)
    
    # Loop over times and do the downloads.
    for d in file_dates:
        SD_download(saildrone_yr, saildrone_ID, d[0], d[1], var_name)
    
    #
    return


############################################################################################
# Helper functions.

def SD_download(yr, SD, t1, t2, var_nm):
    
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
    if (var_nm == 'uvw'):
        query_var = 'trajectory%2Ctime%2Clatitude%2Clongitude' + \
            '%2CUWND%2CVWND%2CWWND'
    # -----------------------------
    # Wind components in sensor-relative coordinates.
    elif (var_nm == 'uvwRaw'):
        if str(yr) == '2019':
            query_var = 'trajectory%2Ctime%2Clatitude%2Clongitude' + \
                '%2CWIND_SENSOR_U%2CWIND_SENSOR_V%2CWIND_SENSOR_W'
        else:
            query_var = 'trajectory%2Ctime%2Clatitude%2Clongitude' + \
                '%2CUWND_UNCOR%2CVWND_UNCOR%2CWWND_UNCOR'
    # -----------------------------
    # Sonic temperature and anemometer wind_status
    elif (var_nm == 'Temp_WindSt'):
        query_var = 'trajectory%2Ctime' + \
            '%2CWIND_STATUS%2CWIND_SONICTEMP'
    # -----------------------------
    # Shortwave radiation.
    elif (var_nm == 'SWrad'):
        if str(yr) == '2018':
            query_var = 'trajectory%2Ctime%2Clatitude%2Clongitude' + \
                '%2CSW_IRRAD_TOTAL%2CSW_IRRAD_DIFFUSE' + \
                '%2CSW_UNMASKED_IRRAD_CENTER%2CSW_UNMASKED_IRRAD_6DET'
        elif str(yr) == '2019':
            query_var = 'trajectory%2Ctime%2Clatitude%2Clongitude' + \
                '%2CSW_IRRAD_TOTAL%2CSW_IRRAD_DIFFUSE'
        else:
            sys.exit('ACHTUNG: year must be one of (2018, 2019)')
    # -----------------------------
    else:
        sys.exit('ACHTUNG: variable descriptor not recognized.')
    
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
        '_' + var_nm + '_' + \
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
    if (d1.hour == 23) & (d1.minute == 59):
        d1 = pd.to_datetime(t1 + np.timedelta64(1, 'h'))
    if (d2.hour == 0) & (d2.minute == 0):
        d2 = pd.to_datetime(t2 - np.timedelta64(1, 'h'))
    timestamp = '{:04d}'.format(d1.year) + \
        '{:02d}'.format(d1.month) + \
        '{:02d}'.format(d1.day) + '_' + \
        '{:04d}'.format(d2.year) + \
        '{:02d}'.format(d2.month) + \
        '{:02d}'.format(d2.day)
    #
    return timestamp

def save(url, filename):
    urllib.urlretrieve(url, filename, reporthook)

def SD_dates_monthly_chunks(yr, SD):
    #
    if str(yr) == '2017':
        file_dates = [
            [np.datetime64('2017-09-02T00:00:00.0','ns'),
             np.datetime64('2017-09-30T23:59:59.9','ns')],
            [np.datetime64('2017-10-01T00:00:00.0','ns'),
             np.datetime64('2017-10-31T23:59:59.9','ns')],
            [np.datetime64('2017-11-01T00:00:00.0','ns'),
             np.datetime64('2017-11-30T23:59:59.9','ns')],
            [np.datetime64('2017-12-01T00:00:00.0','ns'),
             np.datetime64('2017-12-31T23:59:59.9','ns')],
            [np.datetime64('2018-01-01T00:00:00.0','ns'),
             np.datetime64('2018-01-31T23:59:59.9','ns')],
            [np.datetime64('2018-02-01T00:00:00.0','ns'),
             np.datetime64('2018-02-28T23:59:59.9','ns')],
            [np.datetime64('2018-03-01T00:00:00.0','ns'),
             np.datetime64('2018-03-31T23:59:59.9','ns')],
            [np.datetime64('2018-04-01T00:00:00.0','ns'),
             np.datetime64('2018-05-05T23:59:59.9','ns')]
        ]
    elif str(yr) == '2018':
        file_dates = [
            [np.datetime64('2018-10-03T00:00:00.0','ns'),
             np.datetime64('2018-10-31T23:59:59.9','ns')],
            [np.datetime64('2018-11-01T00:00:00.0','ns'),
             np.datetime64('2018-11-30T23:59:59.9','ns')],
            [np.datetime64('2018-12-01T00:00:00.0','ns'),
             np.datetime64('2018-12-31T23:59:59.9','ns')],
            [np.datetime64('2019-01-01T00:00:00.0','ns'),
             np.datetime64('2019-01-31T23:59:59.9','ns')],
            [np.datetime64('2019-02-01T00:00:00.0','ns'),
             np.datetime64('2019-03-05T23:59:59.9','ns')]
        ]
    elif str(yr) == '2019':
        all_dates = {
            '1066':[
                [np.datetime64('2019-06-08T00:00:00.0', 'ns'),
                 np.datetime64('2019-06-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-07-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-07-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-08-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-08-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-09-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-09-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-10-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-10-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-11-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-11-06T23:59:59.9', 'ns')]
            ],
            '1067':[
                [np.datetime64('2019-06-18T00:00:00.0', 'ns'),
                 np.datetime64('2019-06-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-07-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-07-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-08-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-08-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-09-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-09-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-10-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-10-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-11-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-11-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-12-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-12-31T23:59:59.9', 'ns')],
                [np.datetime64('2020-01-01T00:00:00.0', 'ns'),
                 np.datetime64('2020-01-03T23:59:59.9', 'ns')]
            ],
            '1068':[
                [np.datetime64('2019-06-08T00:00:00.0', 'ns'),
                 np.datetime64('2019-06-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-07-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-07-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-08-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-08-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-09-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-09-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-10-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-10-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-11-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-11-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-12-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-12-31T23:59:59.9', 'ns')],
                [np.datetime64('2020-01-01T00:00:00.0', 'ns'),
                 np.datetime64('2020-01-07T23:59:59.9', 'ns')]
            ],
            '1069':[
                [np.datetime64('2019-06-08T00:00:00.0', 'ns'),
                 np.datetime64('2019-06-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-07-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-07-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-08-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-08-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-09-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-09-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-10-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-10-31T23:59:59.9', 'ns')],
                [np.datetime64('2019-11-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-11-30T23:59:59.9', 'ns')],
                [np.datetime64('2019-12-01T00:00:00.0', 'ns'),
                 np.datetime64('2019-12-28T23:59:59.9', 'ns')]
            ]
        }
        file_dates = all_dates[SD]
    else:
        sys.exit('ACHTUNG: year must be one of (2018, 2019)')
    #
    return file_dates



###########################################
# Now actually execute the script.
###########################################
if __name__ == '__main__':
    main()
