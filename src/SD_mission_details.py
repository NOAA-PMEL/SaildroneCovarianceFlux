"""Details of Saildrone missions.

Includes:
Frequency of data collection.
Dates to mark start and end times of suitable Saildrone mission data.
"""

import numpy as np
import datetime
import cf_units

saildrone_frequency = {
    '2017':{
        '1005':{
            'freq_hertz':10,
            'dt_ms':100
        },
        '1006':{
            'freq_hertz':10,
            'dt_ms':100
        }
    },
    '2018':{
        '1005':{
            'freq_hertz':10,
            'dt_ms':100
        },
        '1006':{
            'freq_hertz':10,
            'dt_ms':100
        },
        '1029':{
            'freq_hertz':10,
            'dt_ms':100
        },
        '1030':{
            'freq_hertz':10,
            'dt_ms':100
        }
    },
    '2019':{
        '1066':{
            'freq_hertz':20,
            'dt_ms':50
        },
        '1067':{
            'freq_hertz':20,
            'dt_ms':50
        },
        '1068':{
            'freq_hertz':20,
            'dt_ms':50
        },
        '1069':{
            'freq_hertz':20,
            'dt_ms':50
        }
    }
}

saildrone_dates = {
    '2017':{'1005':{'start':np.datetime64('2017-09-04T03:00:00'),
                    'end':np.datetime64('2018-04-30T00:00:00'),
                    'start_continuous':np.datetime64('2017-12-02T16:58:24.6'),
                    'end_continuous':np.datetime64('2018-03-19T19:34:01.2')},
            '1006':{'start':np.datetime64('2017-09-04T06:00:00'),
                    'end':np.datetime64('2018-05-13T00:00:00'),
                    'start_continuous':np.datetime64('2017-12-02T16:59:30.1'),
                    'end_continuous':np.datetime64('2018-03-19T19:34:10.2')}},
    '2018':{'1005':{'start':np.datetime64('2018-10-06T00:00:00'),
                    'end_anemometer':np.datetime64('2018-11-06T06:00:00'),
                    'end':np.datetime64('2019-01-07T15:00:00')},
            '1006':{'start':np.datetime64('2018-10-06T06:00:00'),
                    'end':np.datetime64('2019-01-14T00:00:00')},
            '1029':{'start':np.datetime64('2018-10-09T02:00:00'),
                    'end':np.datetime64('2019-01-08T00:00:00')},
            '1030':{'start':np.datetime64('2018-10-09T02:00:00'),
                    'end':np.datetime64('2019-03-03T07:00:00')}},
    '2019':{'1066':{'start':np.datetime64('2019-06-26T19:00:00'),
                    'end_anemometer':np.datetime64('2019-09-16T09:00:00'),
                    'end':np.datetime64('2019-12-19T00:00:00'),
                    'start_continuous':np.datetime64('2019-06-09T00:00:00'),
                    'end_continuous':np.datetime64('2019-09-16T09:04:58')},
            '1067':{'start':np.datetime64('2019-06-26T06:00:00'),
                    'end':np.datetime64('2019-12-17T15:00:00'),
                    'start_continuous':np.datetime64('2019-06-18T00:59:30'),
                    'end_continuous':np.datetime64('2020-01-01T00:00:00')},
            '1068':{'start':np.datetime64('2019-06-25T18:00:00'),
                    'end':np.datetime64('2019-12-12T06:00:00'),
                    'start_continuous':np.datetime64('2019-06-08T00:00:00'),
                    'end_continuous':np.datetime64('2019-12-07T00:00:00')},
            '1069':{'start':np.datetime64('2019-06-26T09:00:00'),
                    'end':np.datetime64('2019-12-12T06:00:00'),
                    'start_continuous':np.datetime64('2019-06-08T00:00:00'),
                    'end_continuous':np.datetime64('2019-12-27T00:00:00')}}
}


"""
Manually recovered mission dates:
ALL TIMES UTC 

---------- 2017 ----------

-  1005  -
Left dock:  2017-09-01 17:00:00
Clear of SF bay:  2017-09-01 23:00:00
Left continental shelf:  2017-09-04 03:00:00
Returned to continental shelf:  2018-04-30 00:00:00
Returned to dock at San Luis Obispo: 2018-05-06 17:00:00

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2017_hf_10hz_1005_Uwind_20170902_20180505.nc ---------------
2017-09-05T17:34:00.000000000 to 2017-09-05T18:54:03.800000000
2017-09-13T00:25:49.800000000 to 2017-09-13T02:40:36.700000000
2017-10-05T17:28:05.200000000 to 2017-10-05T18:15:30.300000000
2017-10-12T19:54:30.200000000 to 2017-10-12T21:12:41.600000000
2017-10-22T22:16:26.300000000 to 2017-10-23T13:43:41.400000000
2017-10-26T18:54:30.100000000 to 2017-10-27T22:53:42.600000000
2017-10-31T03:04:21.900000000 to 2017-10-31T03:34:09.900000000
2017-11-01T17:07:01.200000000 to 2017-11-01T23:59:26.100000000
2017-12-02T16:58:24.600000000 to 2018-01-23T17:40:07.600000000
2018-01-23T17:47:28.600000000 to 2018-01-29T10:54:46.700000000
2018-01-29T10:54:47.900000000 to 2018-01-29T21:24:30.100000000
2018-01-29T21:37:11.700000000 to 2018-02-04T23:27:46.700000000
2018-02-04T23:27:48.000000000 to 2018-02-21T07:23:05.400000000
2018-02-21T07:23:16.500000000 to 2018-03-13T00:20:47.400000000
2018-03-13T00:20:52.500000000 to 2018-03-18T12:33:05.500000000
2018-03-18T12:33:06.600000000 to 2018-03-19T19:34:01.200000000
2018-03-25T10:31:13.100000000 to 2018-03-26T00:00:30.000000000

-  1006  -
Left dock:   2017-09-01 17:00:00
Clear of SF Bay:  2019-09-01 23:00:00
Left continental shelf:  2017-09-04 06:00:00
Returned to continental shelf:  2018-05-13 00:00:00
Returned to SF Bay:  2018-05-18 19:00:00

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2017_hf_10hz_1006_Uwind_20170902_20180505.nc ---------------
2017-09-05T17:34:10.100000000 to 2017-09-05T18:52:23.100000000
2017-09-13T00:24:30.100000000 to 2017-09-13T02:41:28.500000000
2017-10-05T17:33:11.700000000 to 2017-10-05T18:15:30.000000000
2017-10-12T19:53:19.900000000 to 2017-10-12T21:14:16.000000000
2017-10-22T22:14:30.100000000 to 2017-10-23T13:45:51.300000000
2017-10-26T19:02:34.900000000 to 2017-10-27T22:48:14.900000000
2017-10-31T03:04:05.700000000 to 2017-10-31T03:34:21.100000000
2017-11-01T17:07:47.100000000 to 2017-11-02T00:05:30.000000000
2017-11-06T14:53:18.900000000 to 2017-11-06T19:14:24.300000000
2017-11-08T21:39:30.200000000 to 2017-11-09T00:54:12.500000000
2017-12-02T16:59:30.100000000 to 2018-01-23T18:56:13.400000000
2018-01-23T18:57:43.500000000 to 2018-02-20T17:23:51.600000000
2018-02-20T17:23:52.700000000 to 2018-03-10T18:46:05.500000000
2018-03-10T18:46:06.800000000 to 2018-03-19T19:34:10.200000000


---------- 2018 ----------

-  1005  -
Left dock:  2018-10-04 00:00:00
Clear of Hawaii continental shelf:  2018-10-06 00:00:00
Anemometer stopped:  2018-11-06 06:00:00
Back to Hawaii continental shelf:  2019-01-07 15:00:00
Maneuvers in coastal waters:  2019-01-14 00:00:00 through 2019-01-27 10:00:00

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2018_hf_10hz_1005_Uwind_20181003_20190131.nc ---------------
2018-10-04T06:59:30.100000000 to 2018-10-04T22:30:30.000000000

-  1006  -
Left dock:  2018-10-04 00:00:00
Clear of Hawaii continental shelf:  2018-10-06 06:00:00
Back to Hawaii continental shelf: 2019-01-14 00:00:00
Maneuvers in coastal waters:  2019-01-22 15:00:00
Back at Dock in Hawaii:  2019-01-27 23:00:00

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2018_hf_10hz_1006_Uwind_20181003_20190131.nc ---------------
2018-10-04T06:59:27.400000000 to 2018-10-04T22:30:30.000000000

-  1029  -
Start of record:  2018-10-04 00:00:00
Clear of Hawaii continental shelf:  2018-10-09 02:00:00
Back to Hawaii continental shelf:  2019-01-08 00:00:00
Maneuvers in coastal waters:  2019-01-17 09:00:00
Back at Dock in Hawaii:  2019-01-27 01:00:00

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2018_hf_10hz_1029_Uwind_20181003_20190131.nc ---------------
2018-10-04T06:56:15.400000000 to 2018-10-04T22:25:45.000000000

-  1030  -
Start of record:  2018-10-04 00:00:00
Clear of Hawaii continental shelf:  2018-10-09 02:00:00
Back to Hawaii continental shelf:  2019-03-03 07:00:00
End of record:  2019-03-06 00:00:00 

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2018_hf_10hz_1030_Uwind_20181003_20190305.nc ---------------
2018-10-04T06:55:34.700000000 to 2018-10-04T22:26:28.900000000

---------- 2019 ----------

-  1066  -
Start of record (maneuvers off Hawaii):  2019-06-08 01:00:00
Clear of Hawaii continental shelf:  2019-06-26 19:00:00
Anemometer switched off:  2019-09-16 09:00:00
Back to Hawaii continental shelf:  2019-12-19 00:00:00
End of record (maneuvers off Hawaii):  2020-01-07 00:00:00

Continuous high-res dates (> 10 minutes):
--------------- ~/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1066_Uwind_20190608_20191106.nc ---------------
2019-06-08T01:39:30.049999872 to 2019-06-28T22:35:30.750000128
2019-06-28T22:35:34.900000000 to 2019-07-09T12:14:24.650000128
2019-07-09T12:14:27.150000128 to 2019-07-12T18:28:53.450000128
2019-07-12T18:28:57.500000000 to 2019-07-27T11:44:37.750000128
2019-07-27T11:44:40.700000000 to 2019-07-29T02:58:35.000000000
2019-07-29T02:58:39.049999872 to 2019-07-31T09:57:48.349999872
2019-07-31T09:57:51.049999872 to 2019-08-22T03:10:44.300000000
2019-08-22T03:10:46.700000000 to 2019-09-03T08:21:44.249999872
2019-09-03T08:21:47.950000128 to 2019-09-16T09:04:58.150000128
Alternative:
-------------------------- File list: -------------------------- 
['/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1066_uvw_20190609_20190701.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1066_uvw_20190701_20190801.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1066_uvw_20190801_20190901.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1066_uvw_20190901_20191001.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1066_uvw_20191001_20191101.nc']
2019-06-09T00:00:00.000000000 to 2019-06-28T22:35:30.750000128
2019-06-28T22:35:34.900000000 to 2019-07-09T12:14:24.650000128
2019-07-09T12:14:27.150000128 to 2019-07-12T18:28:53.450000128
2019-07-12T18:28:57.500000000 to 2019-07-27T11:44:37.750000128
2019-07-27T11:44:40.700000000 to 2019-07-29T02:58:35.000000000
2019-07-29T02:58:39.049999872 to 2019-07-31T09:57:48.349999872
2019-07-31T09:57:51.049999872 to 2019-08-22T03:10:44.300000000
2019-08-22T03:10:46.700000000 to 2019-09-03T08:21:44.249999872
2019-09-03T08:21:47.950000128 to 2019-09-16T09:04:58.150000128


-  1067  -
Start of record (maneuvers off Hawaii):  2019-06-08 05:00:00
Clear of Hawaii continental shelf:  2019-06-26 06:00:00
Back to Hawaii continental shelf:  2019-12-17 15:00:00
End of record (maneuvers off Hawaii):  2020-01-03 19:00:00
-------------------------- File list: -------------------------- 
['/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20190618_20170701.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20190701_20190801.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20190801_20190901.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20190901_20191001.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20191001_20191101.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20191101_20191201.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1067_uvw_20191201_20200101.nc']
2019-06-18T00:59:30.049999872 to 2019-07-11T02:30:46.200000000
2019-07-11T02:30:49.650000128 to 2019-07-18T07:45:24.049999872
2019-07-18T07:45:27.150000128 to 2019-08-02T04:29:09.000000000
2019-08-02T04:29:12.600000000 to 2019-10-04T06:57:36.700000000
2019-10-04T06:57:39.549999872 to 2019-10-07T01:09:53.450000128
2019-10-07T01:09:56.700000000 to 2019-10-24T15:05:57.200000000
2019-10-24T15:06:00.750000128 to 2019-11-23T12:22:55.249999872
2019-11-23T12:22:58.950000128 to 2019-12-05T16:19:32.100000000
2019-12-06T21:52:43.500000000 to 2019-12-23T10:33:46.349999872
2019-12-23T10:33:48.650000128 to 2020-01-01T00:00:00.000000000


-  1068  -
Start of record (maneuvers off Hawaii):  2019-06-07 22:00:00
Clear of Hawaii continental shelf:  2019-06-25 18:00:00
Back to Hawaii continental shelf:  2019-12-12 06:00:00
End of record (maneuvers off Hawaii):  2020-01-07 18:00:00
-------------------------- File list: -------------------------- 
['/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20190608_20190701.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20190701_20190801.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20190801_20190901.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20190901_20191001.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20191001_20191101.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20191101_20191201.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1068_uvw_20191201_20200101.nc']
2019-06-08T00:00:00.000000000 to 2019-08-22T00:44:52.650000128
2019-08-22T00:44:55.400000000 to 2019-09-12T12:49:28.950000128
2019-09-12T12:49:31.549999872 to 2019-10-30T14:22:46.400000000
2019-10-30T14:22:50.000000000 to 2019-11-10T23:14:02.800000000
2019-11-10T23:14:06.349999872 to 2019-12-07T00:00:00.000000000
2019-12-17T06:47:58.800000000 to 2019-12-24T13:41:23.450000128
2019-12-24T13:41:25.650000128 to 2020-01-01T00:00:00.000000000


-  1069  -
Start of record (maneuvers off Hawaii):  2019-06-07 00:00:00
Clear of Hawaii continental shelf:  2019-06-26 09:00:00
Back to Hawaii continental shelf:  2019-12-12 06:00:00
End of record (maneuvers off Hawaii):  2019-12-28 18:00:00
-------------------------- File list: -------------------------- 
['/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20190608_20190701.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20190701_20190801.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20190801_20190901.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20190901_20191001.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20191001_20191101.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20191101_20191201.nc', '/Users/reeveseyre/Documents/Data/Saildrone/10Hz/tpos_2019_hf_20hz_1069_uvw_20191201_20191227.nc']
2019-06-08T00:00:00.000000000 to 2019-06-17T20:26:29.150000128
2019-06-17T20:26:33.100000000 to 2019-08-06T00:00:00.000000000
2019-08-06T15:59:30.049999872 to 2019-08-14T22:33:16.349999872
2019-08-14T22:40:50.950000128 to 2019-10-20T23:22:53.800000000
2019-10-20T23:22:56.049999872 to 2019-10-21T01:07:47.549999872
2019-10-21T01:07:49.600000000 to 2019-10-21T01:59:29.450000128
2019-10-21T02:02:01.800000000 to 2019-11-08T13:57:35.650000128
2019-11-08T13:57:38.049999872 to 2019-12-24T05:26:19.950000128
2019-12-24T05:26:24.000000000 to 2019-12-27T00:00:00.000000000

"""


