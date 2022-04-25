"""Plot spectra of winds (motion-corrected and raw) and saildrone motion.

Usage:
    $ python spectra_winds_motion.py
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
    """Control function:
    1. Define events, details (period, variables, etc.).
    2. Per event: 
        a. load data
        b. pre-process
        c. calculate spectra
    3. Plot - either all events or single.
    """
    
    # Define details.
    spectra_period = 10 # minutes
    motion_var = 'INS_WING_VEL_U'
    wind_comp = 'W'
    
    # Select events.
    events = get_event_dict(0)
    
    # Loop over events and get spectra (xarray dataset) for each.
    for ev in events:
        ev['spectra'] = wind_motion_spectra(ev, spectra_period,
                                            motion_var, wind_comp)
    
    # Plot.
    #import pdb; pdb.set_trace()
    plot_all(events, spectra_period, motion_var, wind_comp)
    for ev in events:
        plot_single(ev, spectra_period, motion_var, wind_comp)
    #
    return


def get_event_dict(group):
    #
    if group == 0:
        chunks = [
            {'year':'2019',
             'saildrone':'1069',
             'start_time':np.datetime64('2019-10-26T23:55:00.02','ns'),
             'end_time':np.datetime64('2019-10-27T06:05:00.02','ns'),
             'filename':'sd1069_2019_20hz_1069_windMotion_forSpectra_20191026T2345_20191027T0615.nc',
             'title_details':'2019 SD1069 2019-10-27 0000     4.93 m/s     11.28 s     1.23 m'},
            {'year':'2019',
             'saildrone':'1068',
             'start_time':np.datetime64('2019-11-28T23:55:00.02','ns'),
             'end_time':np.datetime64('2019-11-29T06:05:00.02','ns'),
             'filename':'sd1068_2019_20hz_1068_windMotion_forSpectra_20191128T2345_20191129T0615.nc',
             'title_details':'2019 SD1068 2019-11-29 0000     9.16 m/s     9.74 s     2.97 m'},
            {'year':'2019',
             'saildrone':'1067',
             'start_time':np.datetime64('2019-09-25T23:55:00.02','ns'),
             'end_time':np.datetime64('2019-09-26T06:05:00.02','ns'),
             'filename':'sd1067_2019_20hz_1067_windMotion_forSpectra_20190925T2345_20190926T0615.nc',
             'title_details':'2019 SD1067 2019-09-26 0000     6.87 m/s     15.49 s     2.12 m'},
            {'year':'2019',
             'saildrone':'1068',
             'start_time':np.datetime64('2019-08-31T23:55:00.02','ns'),
             'end_time':np.datetime64('2019-09-01T06:05:00.02','ns'),
             'filename':'sd1068_2019_20hz_1068_windMotion_forSpectra_20190831T2345_20190901T0615.nc',
             'title_details':'2019 SD1068 2019-09-01 0000     4.22 m/s     9.35 s     1.84 m'},
            {'year':'2019',
             'saildrone':'1066',
             'start_time':np.datetime64('2019-09-10T17:55:00.02','ns'),
             'end_time':np.datetime64('2019-09-11T00:05:00.02','ns'),
             'filename':'sd1066_2019_20hz_1066_windMotion_forSpectra_20190910T1745_20190911T0015.nc',
             'title_details':'2019 SD1066 2019-09-10 1800     7.44 m/s     7.84 s     2.15 m'},
            {'year':'2019',
             'saildrone':'1069',
             'start_time':np.datetime64('2019-08-15T23:55:00.02','ns'),
             'end_time':np.datetime64('2019-08-16T06:05:00.02','ns'),
             'filename':'sd1069_2019_20hz_1069_windMotion_forSpectra_20190815T2345_20190816T0615.nc',
             'title_details':'2019 SD1069 2019-08-16 0000     9.13 m/s     11.45 s     1.95 m'},
            {'year':'2017',
             'saildrone':'1006',
             'start_time':np.datetime64('2018-01-05T17:55:00.02','ns'),
             'end_time':np.datetime64('2018-01-06T00:05:00.02','ns'),
             'filename':'tpos_2017_hf_10hz_1006_windMotion_forSpectra_20180105T1745_20180106T0015.nc',
             'title_details':'2017 SD1006 2018-01-05 1800     7.17 m/s'},
            {'year':'2017',
             'saildrone':'1005',
             'start_time':np.datetime64('2017-12-14T17:55:00.02','ns'),
             'end_time':np.datetime64('2017-12-15T00:05:00.02','ns'),
             'filename':'tpos_2017_hf_10hz_1005_windMotion_forSpectra_20171214T1745_20171215T0015.nc',
             'title_details':'2017 SD1005 2017-12-14 1800     6.69 m/s'}
        ]
    else:
        sys.exit('ACHTUNG: group must be one of (0)')
    #
    return chunks


def wind_motion_spectra(event, PD, varname_motion, varname_wind):
    
    # Define variable names as in files.
    vn1 = varname_wind + 'WND'
    if event['year'] == '2019':
        vn2 = 'WIND_SENSOR_' + varname_wind
    else:
        vn2 = varname_wind + 'WND_UNCOR'
    if (event['year'] == '2019') & (varname_motion == 'INS_WING_VEL_U'):
        vn3 = 'INS_WING_VEL_D'
    elif (event['year'] == '2019') & (varname_motion == 'INS_HULL_VEL_U'):
        vn3 = 'INS_HULL_VEL_D'
    else:
        vn3 = varname_motion
    var_labels = {vn1:r'$w$', vn2:r'$w_{raw}$', vn3:r'$\dot{z}_{SD}$',
                  vn1 + '_decorr':r'$w_{decorr}$',
                  vn2 + '_decorr':r'$w_{raw,decorr}$'}
    var_colors = {vn1: 'blue', vn2: 'orange', vn3: 'green',
                  vn1 + '_decorr':'blue', vn2 + '_decorr': 'orange'}
    var_lstyle = {vn1: '-', vn2: '-', vn3: '-',
                  vn1 + '_decorr':'--', vn2 + '_decorr': '--'}
    var_lwidth = {vn1: '1', vn2: '1', vn3: '1',
                  vn1 + '_decorr':'0.3', vn2 + '_decorr': '0.3'}
    
    # Load data.
    freq_hertz = SD_mission_details.saildrone_frequency\
        [event['year']][event['saildrone']]['freq_hertz']
    delta_t_ms = SD_mission_details.saildrone_frequency\
        [event['year']][event['saildrone']]['dt_ms']
    delta_t_64 = np.timedelta64(delta_t_ms, 'ms')
    ds_ev = load_swap_time(config.data_dir + '10Hz/' +
                           event['filename'])
    if not qc_times_evenly_spaced(ds_ev, tol=np.timedelta64(25, 'ms')):
        ds_ev = qc_times_make_even(ds_ev, delta_t_ms)
    ds_ev = ds_ev.sel(time=((ds_ev.time >= event['start_time']) &
                            (ds_ev.time <= event['end_time'])))
    ds_L1 = load_L1_data(event['year'], event['saildrone'])
    ds_L1 = ds_L1.sel(time=((ds_L1.time >= event['start_time']) &
                            (ds_L1.time <= event['end_time'])))
    if (len(ds_L1.time) != len(ds_ev.time)):
        print('Achtung: Event time series should be same length as L1 time series.')
        #import pdb; pdb.set_trace()
    print(vn1 + ' count of L1 != event data:')
    print(np.sum(~np.isclose(ds_L1[vn1].data, ds_ev[vn1].data,
                             rtol=0.0, atol=1e-14, equal_nan=True)))
    
    # Add some metadata that is missing for a few variables.
    for v in [vn1, vn2, vn3]:
        if 'long_name' not in ds_ev[v].attrs.keys():
            ds_ev[v].attrs['long_name'] = v
        if 'standard_name' not in ds_ev[v].attrs.keys():
            ds_ev[v].attrs['standard_name'] = v
        if 'units' not in ds_ev[v].attrs.keys():
            ds_ev[v].attrs['units'] = 'm s-1'
    
    # Apply QC flags.
    print('Number missing before flags:')
    for v in [vn1, vn2, vn3]:
        print(np.sum(np.isnan(ds_ev[v].data)))
    all_wind_ok = (
        ((ds_L1['UWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_L1['UWND_FLAG_PRIMARY'] == get_flag1('not_evaluated'))) &
        ((ds_L1['VWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_L1['VWND_FLAG_PRIMARY'] == get_flag1('not_evaluated'))) &
        ((ds_L1['WWND_FLAG_PRIMARY'] == get_flag1('good')) |
         (ds_L1['WWND_FLAG_PRIMARY'] == get_flag1('not_evaluated')))
    )
    print('Number missing after flags:')
    for v in [vn1, vn2, vn3]:
        ds_ev[v] = ds_ev[v].where(all_wind_ok, other=np.nan)
        print(np.sum(np.isnan(ds_ev[v].data)))
    
    # Interpolate gaps.
    ds_ev = ds_ev.interpolate_na(dim='time',
                                 method='linear',
                                 use_coordinate=True,
                                 max_gap=np.timedelta64(1000,'ms'))
    print('Number missing after interpolation:')
    for v in [vn1, vn2, vn3]:
        print(np.sum(np.isnan(ds_ev[v].data)))
        
    # Reshape time series to 2D.
    period_len = freq_hertz*60*PD
    ds_2D = reshape_1D_nD_numpy(
        ds_ev, [np.int64(len(ds_ev.time)/period_len), period_len]
    )
    ds_2D['time_mid'] = np.array(
        np.array(
            ds_2D.time_first
            + 0.5*(ds_2D.time_last - ds_2D.time_first),
            dtype='datetime64[s]'
        ),
        dtype='datetime64[ns]'
    )
    ds_2D[vn1 + '_decorr'], mu0_vn1, mu1_vn1 = \
        decorr_single(ds_2D[vn1], ds_2D[vn3])
    ds_2D[vn2 + '_decorr'], mu0_vn2, mu1_vn2 = \
        decorr_single(ds_2D[vn2], ds_2D[vn3])
    #import pdb; pdb.set_trace()
    
    # Detrend and apply tapering.
    tukey_weights = np.expand_dims(tukey(PD*60*freq_hertz,
                                         alpha=0.2, sym=True),
                                   0)
    ds_detrend = ds_2D.copy()
    for v in [vn1, vn2, vn3]:
        ds_detrend[v].data = detrend(ds_2D[v], axis=1, type='linear')
        ds_detrend[v].data = ds_detrend[v]*tukey_weights
    
    # Calculate spectra.
    data_vars = {}
    for v in [vn1, vn2, vn3]: #, vn1 + '_decorr', vn2 + '_decorr']:
        spec_np, f = \
            discrete_spectral_intensity_2D(
                ds_detrend[v].data,
                period_len, delta_t_64, PD*60
            )
        spec_xr = xr.DataArray(
            spec_np,
            coords=[('time', ds_detrend.time_mid),
                    ('frequency', f)],
            attrs={'long_name':'Discrete spectral intensity of ' +
                   ds_detrend[v].attrs['long_name'],
                   'standard_name':'spectral_intensity_' +
                   ds_detrend[v].attrs['standard_name'],
                   'units':'s (' + ds_detrend[v].attrs['units'] + ')2',
                   'plot_label':var_labels[v],
                   'plot_color':var_colors[v],
                   'plot_linestyle':var_lstyle[v],
                   'plot_linewidth':var_lwidth[v]})
        data_vars[v + '_discrete_spectral_intensity'] = spec_xr
    
    # Put into dataset.
    ds_spectra = xr.Dataset(data_vars)
    ds_spectra['time'].attrs['long_name'] = 'Date-time of nominal center of spectral analysis period.'
    ds_spectra['frequency'].attrs['standard_name'] = 'frequency'
    ds_spectra['frequency'].attrs['units'] = 'Hz'
    ds_spectra['frequency'].attrs['limits'] = 'Minimum: 1 oscillation per spectral analysis period. Maximum: Nyquist frequency.'
    ds_spectra.coords['n_bounds'] = np.array([1,2])
    ds_spectra['time_bounds'] = (('time', 'n_bounds'),
                                 np.stack([ds_detrend.time_first.data,
                                           ds_detrend.time_last.data],
                                          axis=1))
    #
    return ds_spectra


def decorr_single(Y_in, X_in):
    """Pyhon-ised version of Byron Blomquist's script; but one predictor only!

    Decorrelates high-rate response Y from predictor variable in X.
    Y and X should be the same length, and simultaneous measurements.
    

    inputs:    Y_in    - N x M signal response vector (to be decorrelated)
               X_in    - N x M array for one predictor variable,
                   N is number of time windows, M is number of high-frequency
                   observations within each time window (e.g, N=37, M=12000).
    outputs:   Y_out   - decorrelated response signal
               mu0     - correlation coefficients w/respect to Y_in (raw)
               mu1     - correlation coefficients w/respect to Y_out 
                         (after decorr)

    BWB Feb 2021
    """
    # Create outputs (Y_out is an xarray data array to match inputs).
    Y_out = Y_in.copy()
    mu0 = np.nan*np.zeros((Y_in.data.shape[0], 1))
    mu1 = np.nan*np.zeros((Y_in.data.shape[0], 1))
    
    # Detrend inputs (these are now numpy arrays).
    Y_in_dt = detrend(Y_in, axis=1, type='linear')
    X_in_dt = detrend(X_in, axis=1, type='linear')
    
    # Loop over time periods.
    for ip in range(Y_in.data.shape[0]):
        
        # Calculate correlations coefficients before decorr.
        mu0[ip, 0] = np.corrcoef(Y_in_dt[ip,:],
                                 X_in_dt[ip,:])[0,1]
        
        # Calculate decorrelated time series.
        linreg = scipy.stats.linregress(X_in.data[ip,:], Y_in_dt[ip,:])
        Y_out.data[ip,:] = Y_in.data[ip,:] - linreg.slope*X_in.data[ip,:]
        
        # Calculate correlations coefficients after decorr.
        Y_out_dt = detrend(Y_out.data[ip,:], axis=0, type='linear')
        mu1[ip, 0] = np.corrcoef(Y_out_dt,
                                 X_in_dt[ip,:])[0,1]
    
    # Add some metadata.
    Y_out.attrs['long_name'] = 'decorrelated ' + Y_out.attrs['long_name']
    Y_out.attrs['standard_name'] = 'decorrelated_' \
        + Y_out.attrs['standard_name']
    #
    return Y_out, mu0, mu1


def discrete_spectral_intensity_2D(A, len_A, dt, PD):
    """Calculates discrete spectral intensity of A, which is 
       of length len_A data points, has period PD and sample spacing dt.
    """
    
    F_A = (1/len_A)*fft.fft(A, axis=1)
    n = fft.fftfreq(len_A, d=(1.0/len_A))
    n[len_A//2] = -1.0*n[len_A//2]
    f = fft.fftfreq(len_A, d=(dt/np.timedelta64(1, 's')))
    f[len_A//2] = -1.0*f[len_A//2]
    
    G_A = F_A*np.conjugate(F_A)
    E_factor = np.concatenate((2.0*np.ones(np.int64(len_A/2)-1), np.ones(1)))
    E_A = G_A[:, 1:np.int64((len_A/2) + 1)]*np.expand_dims(E_factor, 0)
    delta_f = 1.0/PD
    S_A_f = np.real(E_A/delta_f)
    
    return S_A_f, f[1:np.int64((len_A/2) + 1)]


def discrete_spectral_intensity(A, len_A, dt, PD):
    """Calculates discrete spectral intensity of A, which is 
       of length len_A data points, has period PD and sample spacing dt.
    """
    
    F_A = (1/len_A)*fft.fft(A)
    n = fft.fftfreq(len_A, d=(1.0/len_A))
    n[len_A//2] = -1.0*n[len_A//2]
    f = fft.fftfreq(len_A, d=(dt/np.timedelta64(1, 's')))
    f[len_A//2] = -1.0*f[len_A//2]
    
    G_A = F_A*np.conjugate(F_A)
    E_factor = np.concatenate((2.0*np.ones(np.int64(len_A/2)-1), np.ones(1)))
    E_A = G_A[1:np.int64((len_A/2) + 1)]*E_factor
    delta_f = 1.0/PD
    S_A_f = np.real(E_A/delta_f)
    
    return S_A_f, f


def plot_single(ev, PD, varname_motion, varname_wind):
    
    # Set up axes.
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    
    # Plot of all averaged spectra.
    ip = (0, 0)
    axs[ip].set_ylabel(r'$f~S_{x}(f)$')
    axs[ip].set_title('Averages of ' +
                      str(len(ev['spectra'].coords['time'])) +
                      ' spectra')
    axs[ip].set_xscale('log')
    axs[ip].set_yscale('log')
    #axs[ip].set_ylim(ylims)
    axs[ip].grid()
    ind_plot_list = []
    for v in ev['spectra'].keys():
        if 'plot_label' in ev['spectra'][v].attrs.keys():
            ind_plot_list.append(v)
            axs[ip].plot(ev['spectra'].coords['frequency'],
                         ev['spectra'].coords['frequency']\
                         *ev['spectra'][v].mean(dim='time'),
                         label=ev['spectra'][v].attrs['plot_label'])
    axs[ip].legend()
    
    # Plots of individual spectra.
    ip = (0, 1)
    v = ind_plot_list[0]
    axs[ip].set_ylabel(r'$f~S_{x}(f)$')
    axs[ip].set_title(ev['spectra'][v].attrs['plot_label'])
    axs[ip].set_xscale('log')
    axs[ip].set_yscale('log')
    #axs[ip].set_ylim(ylims)
    axs[ip].grid()
    for it in range(len(ev['spectra'].coords['time'])):
        axs[ip].plot(ev['spectra'].coords['frequency'],
                     ev['spectra'].coords['frequency']\
                     *ev['spectra'][v].isel(time=it),
                     color='gray', linewidth=0.1, alpha=0.5)
    axs[ip].plot(ev['spectra'].coords['frequency'],
                 ev['spectra'].coords['frequency']\
                 *ev['spectra'][v].mean(dim='time'))
    
    # Plots of individual spectra.
    ip = (1, 0)
    v = ind_plot_list[1]
    axs[ip].set_xlabel('Frequency / Hz')
    axs[ip].set_ylabel(r'$f~S_{x}(f)$')
    axs[ip].set_title(ev['spectra'][v].attrs['plot_label'])
    axs[ip].set_xscale('log')
    axs[ip].set_yscale('log')
    #axs[ip].set_ylim(ylims)
    axs[ip].grid()
    for it in range(len(ev['spectra'].coords['time'])):
        axs[ip].plot(ev['spectra'].coords['frequency'],
                     ev['spectra'].coords['frequency']\
                     *ev['spectra'][v].isel(time=it),
                     color='gray', linewidth=0.1, alpha=0.5)
    axs[ip].plot(ev['spectra'].coords['frequency'],
                 ev['spectra'].coords['frequency']\
                 *ev['spectra'][v].mean(dim='time'))
    
    # Plots of individual spectra.
    ip = (1, 1)
    v = ind_plot_list[2]
    axs[ip].set_xlabel('Frequency / Hz')
    axs[ip].set_ylabel(r'$f~S_{x}(f)$')
    axs[ip].set_title(ev['spectra'][v].attrs['plot_label'])
    axs[ip].set_xscale('log')
    axs[ip].set_yscale('log')
    #axs[ip].set_ylim(ylims)
    axs[ip].grid()
    for it in range(len(ev['spectra'].coords['time'])):
        axs[ip].plot(ev['spectra'].coords['frequency'],
                     ev['spectra'].coords['frequency']\
                     *ev['spectra'][v].isel(time=it),
                     color='gray', linewidth=0.1, alpha=0.5)
    axs[ip].plot(ev['spectra'].coords['frequency'],
                 ev['spectra'].coords['frequency']\
                 *ev['spectra'][v].mean(dim='time'))
    
    # Overall details.
    fig.suptitle(ev['title_details'])
    
    # Save out or display figure.
    # plt.show()
    file_name = config.plot_dir + \
        'Saildrone_explore/MotionCorrection/spectra_winds_motion_' + \
        ev['year'] + '_' + ev['saildrone'] + '_' + \
        pd.to_datetime(str(ev['spectra'].coords['time'].data[0]))\
        .strftime('%Y%m%dT%H%M') + '.'
    file_format = 'pdf'
    plt.savefig(file_name + file_format,
                format=file_format,
                bbox_inches='tight')
    file_format = 'png'
    plt.savefig(file_name + file_format,
                format=file_format,
                bbox_inches='tight')
    #
    return


def plot_all(evs, PD, varname_motion, varname_wind):
    
    # Set up axes.
    fig, axs = plt.subplots(int(np.ceil(len(evs)/2)), 2,
                            figsize=(10, 10),
                            sharex=True, sharey=True)
    
    # Loop over events.
    for iev, ev in enumerate(evs):
        ax = axs.flatten()[iev]
        if iev == 0:
            ax.set_title('Year Drone Start time           ' +
                         r'$|\mathbf{U}|$' + '       ' +
                         r'$\mathit{T_{D}}$' + '      ' +
                         r'$\mathit{H_{s}}$' +
                         '\n' + ev['title_details'],
                         fontsize='small', loc='left')
        else:
            ax.set_title(ev['title_details'], fontsize='small', loc='left')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        for v in ev['spectra'].keys():
            if 'plot_label' in ev['spectra'][v].attrs.keys():
                ax.plot(ev['spectra'].coords['frequency'],
                        ev['spectra'][v].mean(dim='time'),
                        color=ev['spectra'][v].attrs['plot_color'],
                        linewidth=ev['spectra'][v].attrs['plot_linewidth'],
                        linestyle=ev['spectra'][v].attrs['plot_linestyle'],
                        label=ev['spectra'][v].attrs['plot_label'])
    
    # Sort out common details.
    for ax in axs[:,0].flatten():
        ax.set_ylabel(r'$\mathit{S_{x}(f)}$' + ' (' + r'$m^{2}~s^{-1}$' + ')')
    for ax in axs[-1,:].flatten():
        ax.set_xlabel(r'$\mathit{f}$' + ' (Hz)')
    axs[0,0].legend(
        title=str(int((len(evs[0]['spectra'].coords['time'])-1)/6)) +
        '-hour mean'
    )
    
    # Save out or display figure.
    # plt.show()
    file_name = config.plot_dir + \
        'Saildrone_explore/MotionCorrection/' + \
        'spectra_winds_motion_all.'
    file_format = 'pdf'
    plt.savefig(file_name + file_format,
                format=file_format,
                bbox_inches='tight')
    file_format = 'png'
    plt.savefig(file_name + file_format,
                format=file_format,
                bbox_inches='tight')
    #
    return


################################################################################
# Now actually execute the script.
################################################################################
if __name__ == '__main__':
    main()
