"""
Last modified: 27 Oct 2025
"""


import numpy as np
from numpy import ma

import xarray as xr
from eofs.xarray import Eof

import pandas as pd

import scipy.stats as ss
import scipy as sp
from scipy.stats.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator as rginterp

import matplotlib.dates as mdates

import sys
import os

from area import area

import gsw

# from PhD.PhD_scripts.ch2_altimetry.A_altimetry.B_offsets.B2_OL_plots_full import total_area


def decode_cf_time(var, units):
  """
  Use xr.decode_cf to convert time as a number with a reference
  to datetime64[ns]. Uses a units attribute like 'days since 0001-01-01'

  > returns an array of datetime64[ns]
  """
  attrs = {"units": units}
  t = xr.Dataset({'time':('time', var, attrs)})
  t_cf = xr.decode_cf(t)
  return t_cf.time.values

def detrend_DataArray(arr, confidence):
    """
    Calculate a linear least-squares regression and confidence
    interval for a given xarray.DataArray time series.
    The time array is converted to number of days from a datetime.
    The linear trend hence has units of array_units/day.

    arr : xarray.DataArray, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95

    returns: de-trended xarray.DataArray 
    """

    # If input is a DataArray, proceed with the original logic
    ndays = mdates.date2num(list(arr.time.values))
    dt = ndays - ndays[0]  # for computing linear trend

    # mask nans if present
    arr_data = ma.masked_invalid(arr.data)
    # number of valid observations
    obs = ma.count(arr_data)

    # extract only valid data and their time indices from masked arrays
    arr_vals = arr_data[~arr_data.mask].squeeze()
    tim = dt[~arr_data.mask].squeeze()

    if obs > 2:  # time series must have at least 2 points
        slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
        # confidence interval
        ci = stderr * sp.stats.t.ppf((1 + confidence) / 2., obs - 1)
    else:
        slope = interc = r = pp = ci = np.nan
        arr_det = arr

    arr_det = arr - (dt * slope + interc)

    return arr_det


def detrend_Dataset(arr, confidence):
    """
    Calculate a linear least-squares regression and confidence
    interval for a given xarray time series.
    The time array is converted to number of days from a datetime.
    The linear trend hence has units of array_units/day.

    arr : xarray.Dataset or xarray.DataArray, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95

    returns: xarray.Dataset with intercept, slope, r-value, p-value, conf interval
            units of the array/day
    """

    # Check if input is a Dataset or DataArray
    if isinstance(arr, xr.Dataset):
        detrended = {}
        for var in arr.data_vars:
            var_det = detrend_DataArray(arr[var], confidence)
            detrended[var] = var_det

        # Combine results into a single Dataset
        arr_det = xr.Dataset(detrended)
    else:
        arr_det = detrend_dataArray(arr, confidence)

    return arr_det

def trend_ci(arr, confidence):
    """
    Calculate a linear least-squares regression and confidence 
    interval for a given xarray time series. 
    The time array is converted to number of dayas from a datetime.
    The linear trend hence has units of array_units/day.

    arr : array, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95
    
    returns: xarray.Dataset with intercept, slope, r-value, p-value, conf interval
            units of the array/day
    """
    ndays = mdates.date2num(list(arr.time.values))
    dt = ndays - ndays[0] # for computing linear trend, regularly sampled array

    # mask nans if present
    arr_data = ma.masked_invalid(arr)
    # number of valid observations
    obs = ma.count(arr_data)

    # extract only valid data and their time indices from masked arrays
    arr_vals = arr_data[~arr_data.mask].squeeze()
    tim = dt[~arr_data.mask].squeeze()

    if obs > 2: # time serie must have at least 2 points
        slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
        # confidence interval
        ci = stderr * sp.stats.t.ppf((1+confidence)/2., obs-1)
        
        # residuals
        #trend = interc + slope*tim
        #res = arr_vals - trend
        #mean_res, sem_res = ma.mean(res), res.std(ddof=1)/np.sqrt(obs-1)

    else:
        slope = interc = r = pp = ci = np.nan
        arr_det = arr
    ds = xr.Dataset({'slope' : slope,
                    'intercept' : interc,
                    'r_coeff' : r,
                    'p_val' : pp,
                    'ci' : ci})

    ds.slope.attrs["long_name"] = 'slope of regression line'
    ds.r_coeff.attrs["long_name"] = 'correlation coefficient'
    ds.p_val.attrs["long_name"] = ("two-sided p-value for a hypothesis "
        "test whose null hypothesis is that the slope is zero," 
        "using Wald Test with t-distribution of the test statistic")
    ds.ci.attrs["long_name"] = 'confidence interval'

    arr_det = arr  - (dt * slope + interc)

    return ds, arr_det

def trend_ci_np(ndays, arr, confidence):
    """
    Calculate a linear least-squares regression and confidence 
    interval for a given xarray time series. 
    The time array is converted to number of dayas from a datetime.
    The linear trend hence has units of array_units/day.

    arr : array, 1D
          time series, must contain coordinate 'time'
    confidence : float
                 confidence interval, e.g. 0.95
    
    returns: xarray.Dataset with intercept, slope, r-value, p-value, conf interval
            units of the array/day
    """
    dt = ndays - ndays[0] # for computing linear trend, regularly sampled array

    # mask nans if present
    arr_data = ma.masked_invalid(arr)
    # number of valid observations
    obs = ma.count(arr_data)

    # extract only valid data and their time indices from masked arrays
    arr_vals = arr_data[~arr_data.mask].squeeze()
    tim = dt[~arr_data.mask].squeeze()

    if obs > 2: # time serie must have at least 2 points
        slope, interc, r, pp, stderr = ss.linregress(tim, arr_vals)
        # confidence interval
        ci = stderr * sp.stats.t.ppf((1+confidence)/2., obs-1)
        
        # residuals
        #trend = interc + slope*tim
        #res = arr_vals - trend
        #mean_res, sem_res = ma.mean(res), res.std(ddof=1)/np.sqrt(obs-1)

    else:
        slope = interc = r = pp = ci = np.nan
        arr_det = arr
    ds = xr.Dataset({'slope' : slope,
                    'intercept' : interc,
                    'r_coeff' : r,
                    'p_val' : pp,
                    'ci' : ci})

    ds.slope.attrs["long_name"] = 'slope of regression line'
    ds.r_coeff.attrs["long_name"] = 'correlation coefficient'
    ds.p_val.attrs["long_name"] = ("two-sided p-value for a hypothesis "
        "test whose null hypothesis is that the slope is zero," 
        "using Wald Test with t-distribution of the test statistic")
    ds.ci.attrs["long_name"] = 'confidence interval'

    arr_det = arr  - (dt * slope + interc)

    return ds, arr_det

def detrend2d(arr, tim):
    """
    remove linear trend from 2d array (time x depth)
    arr : shape(r, c)
    tim : shape(r)
    """
    t, d = arr.shape

    slope, interc, ci = [np.ones((d,)) for _ in range(3)]
    for i in range(d):
      arr_trend, _ = trend_ci(dt, arr[:, i], 0.95)
      slope[i] = arr_trend.slope.values
      interc[i] = arr_trend.intercept.values
      ci[i] = arr_trend.ci.values

    trend = tim[:, np.newaxis] * slope[np.newaxis, :] + interc[np.newaxis, :]
    arr_det = arr - trend

    return arr_det, slope, ci

def detrend3d(var):
    """
    remove linear trend (95% CI) from 3d array (lon, lat, tim)
    returns detrended array, slope, pval
    """
    
    date = var.time.dt.strftime("%m.%Y").values
    ndays = mdates.date2num(list(var.time.values))
    dt = ndays - ndays[0]

    # DIMENSIONS
    londim, latdim, timdim = var.shape

    # compute linear trend at every grid point
    interc, slope, ci, pval = [ma.zeros((londim, latdim)) for _ in range(4)]
    for r in range(londim):
        for c in range(latdim):
            arr_trend, _ = trend_ci(var[r,c,:], 0.95)
            interc[r, c] = arr_trend.intercept.values
            slope[r, c] = arr_trend.slope.values
            pval[r, c] = arr_trend.p_val.values

    var_det = var  - (dt[np.newaxis, np.newaxis, :] * slope[:, :, np.newaxis] + interc[:, :, np.newaxis])
            
    return var_det, slope, pval


def grid_area(glon, glat):
    """
    Compute area of every grid cell.
    
    glat, glon: lat/lon at bin edges
    returns an array of the same size as the grid
    """
    lo, la = glon.shape
    area_grid = np.ones((lo-1, la-1))

    for i in range(lo-1):
        for j in range(la-1):
            lon0, lon1 = glon[i, j], glon[i+1, j]
            lat0, lat1 = glat[i, j], glat[i, j+1] 
            obj = {'type':'Polygon', 'coordinates':[[[lon0, lat0], [lon0, lat1],
                                                     [lon1, lat1], [lon1, lat0],
                                                     [lon0, lat0]]]}
            area_grid[i, j] = area(obj)
    return area_grid

def area_weighted_avg(arr, lon, lat):
    """
    arr : 2d array, can be masked,
        shape(r, c)
    lon, lat : 1d array with coordinates at bin edges
    
    returns a 2d array of size (r, c)
    """
    var = ma.masked_invalid(arr)
    glat, glon = np.meshgrid(lat, lon) 
    area_var_tot = grid_area(glon, glat)
    #apply mask 
    area_var = area_var_tot[~var.mask]
    aw_avg = (var*area_var_tot).sum()/area_var.sum()
    return aw_avg

#-------------------------------------------------------------------------------
import scipy.interpolate as itp

def interp_nan(var,x,y):
    """
    interpolate non-nan values from neighbours at nan loci (e.g. to be used
    before Gaussian filtering)

    From Clement Vic (30/11/2018)
    """
    var  = np.ma.masked_invalid(var)
    xi   = x[~var.mask]
    yi   = y[~var.mask]
    vari = var[~var.mask]
    varnew = itp.griddata((xi,yi),vari.ravel(),(x,y),method='nearest')
    return varnew
#-------------------------------------------------------------------------------
# GAUSSIAN FILTERING
#-------------------------------------------------------------------------------
## 2D Gaussian filter
# Choose a sigma = 2 (i.e. 100km) (Armitage 2016, Dotto 2018)

from scipy.ndimage.filters import gaussian_filter as gfilt

def gaussian_filt(initial_arr, sigma, mode):
    """
    Multidimensional Gaussian filter from 
    scipy.ndimage.filters.

    initial_arr : array_like
        must not contain nans or be a masked array
    sigma : scalar
        standard deviation for the Gaussian kernel
    mode : str
        'reflect', 'wrap', 'constant', nearest', 'mirror'

    Returns:
        filt_arr_cut : the filtered array
    """    
    # append a few grid points at each longitude end
    # with data from the opposite end
    # similar to 'wrap' mode but avoiding wrapping
    # data at latitude ends
    n_append = 10*sigma
    extended_arr = np.vstack((initial_arr[-n_append:, :], initial_arr,
                              initial_arr[:n_append, :]))
    filt_arr = gfilt(extended_arr, sigma=sigma, order=0, 
                     mode=mode)
    filt_arr_cut = filt_arr[n_append:-n_append, :]
    
    return filt_arr_cut

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    # crop the ends to match the size of the initial signal
    yy = y[int(window_len/2):-int(window_len/2)]
    #print(yy.shape)
    return yy

def rotate_vec(u, v, alpha_deg, dir):
    """
    rotate a 2D vector clockwise by an angle
    parameters:
        u, v : velocity components
        alpha_deg : angle in degrees

   returns: rotated u, v 
    """
    alpha = np.radians(alpha_deg)
    if dir == 'clockwise':
        u_rot = u * np.cos(alpha) + v * np.sin(alpha)
        v_rot = (-1)*u * np.sin(alpha) + v * np.cos(alpha)
    elif dir == 'anticlockwise':
        u_rot = u * np.cos(alpha) - v * np.sin(alpha)
        v_rot = u * np.sin(alpha) + v * np.cos(alpha)        

    return u_rot, v_rot

def rotate_vec_xr(u, v, alpha_deg, dir):
    """
    rotate a 2D vector clockwise by an angle from a fixed frame
    parameters:
        u, v : velocity components
        alpha_deg : angle in degrees, +ve clockwise from zonal/meridional dir

   returns: rotated u, v in an xarray dataset
    """
    alpha = np.radians(alpha_deg)
    time = u.time
    u = u.values
    v = v.values

    if dir == 'clockwise':
        u_rot = u * np.cos(alpha) + v * np.sin(alpha)
        v_rot = (-1)*u * np.sin(alpha) + v * np.cos(alpha)
    elif dir == 'anticlockwise':
        u_rot = u * np.cos(alpha) - v * np.sin(alpha)
        v_rot = u * np.sin(alpha) + v * np.cos(alpha)   

    ds = xr.Dataset({"u" : ('time', u_rot),
        "v" : ("time", v_rot)},
        coords={"time" : time})
    return ds

def rotate_frame(u, v, alpha_deg, direction):
    """
    Return orthogonal vector components projected onto a rotated frame.
    (this is the correct function to use for obtaining alon/across trough components)
    
    Parameters:
        u, v : zonal/meridional velocity components (xarray)
        alpha_deg : angle in degrees
        direction : 'clockwise' or 'anticlockwise'

   returns: u, v projected onto the rotated coord frame
    """
    alpha = np.radians(alpha_deg)
    
    time = u.time
    u = u.values
    v = v.values

    if direction == 'clockwise':
        u_rot = u * np.cos(alpha) - v * np.sin(alpha)
        v_rot = u * np.sin(alpha) + v * np.cos(alpha)
    elif direction == 'anticlockwise':
        u_rot = u * np.cos(alpha) + v * np.sin(alpha)
        v_rot = (-1)* u * np.sin(alpha) + v * np.cos(alpha)        
        
    ds = xr.Dataset({"u" : ('time', u_rot),
        "v" : ("time", v_rot)},
        coords={"time" : time})
    return ds

def remove_nan_xr(arr1, arr2):
    """
    remove nans from two arrays. arrays can have different time start/end points
    but they should have some overlap.

    var1, var2: xarray datasets with time coordinate named 'time'; 
                monthly time series
    v1, v2: arrays with matching length and no nans

    returns v1, v2, time
    """
    var1 = arr1.copy()
    var2 = arr2.copy()

    v1_start, v1_end = var1.time[0], var1.time[-1]
    v2_start, v2_end = var2.time[0], var2.time[-1]
    
    t_start, t_end = v1_start, v1_end
    if v1_start < v2_start:
        t_start = v2_start
    if v1_end > v2_end:
        t_end = v2_end

    var1 = var1.sel(time=slice(t_start, t_end))
    var2 = var2.sel(time=slice(t_start, t_end))
    
    var1 = var1.values
    var2 = var2.values
    
    var1[np.isnan(var2)] = np.nan
    var2[np.isnan(var1)] = np.nan
    
    v1 = var1[~np.isnan(var1)]
    v2 = var2[~np.isnan(var2)]
    
    return v1, v2

def pearsonr_nan(arr1, arr2):
    """
    Compute the pearson correlation coefficient and significance 
    for two xarrays. arrays can have different time start/end points
    but they should have some overlap.

    var1, var2: xarray datasets with time coordinate named 'time'

    returns r, p
    """
    var1 = arr1.copy()
    var2 = arr2.copy()

    v1, v2 = remove_nan_xr(var1, var2)

    if np.count_nonzero(v1) < 2 or np.count_nonzero(v2) < 2:
        r, p = np.nan, np.nan 
        print("the arrays are either null or have less than 2 valid values")
    
    else:
        r, p = pearsonr(v1, v2)
    return r, p

def add_monthly_lag(var, lag):
    var_lag = var.shift(time=lag)    
    return var_lag

def add_daily_lag(var, lag):
    var_lag = var.shift(time=lag)    
    return var_lag

def mad(arr):
    """
    Compute the median absolute deviation of an array.
    """
    data = ma.masked_invalid(arr)
    data = data[~data.mask].squeeze()

    median = np.median(data)
    residuals = data - median
    mad = np.median(abs(residuals))
    return mad


def seasonal_avg(arr_months, arr, stats):
    # initialise arrays
    avg, spread = [ma.ones((12)) for _ in range(2)]
    arr = ma.masked_invalid(arr)
    if stats=='median':
        for i in range(12):
            mm = ma.masked_not_equal(arr_months, i+1)
            arr_m = arr[~mm.mask]
            avg[i] = np.median(arr_m)
            spread[i] = mad(arr)
    elif stats=='mean':
        for i in range(12):
            mm = ma.masked_not_equal(arr_months, i+1)
            arr_m = arr[~mm.mask]
            avg[i] = np.mean(arr_m)
            spread[i] = ma.std(arr_m, ddof=1)
    else:
        avg[:] = spread[:] = ma.masked
        print("typo")
    return avg, spread

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# def geos_vel(var, lat, lon):
#     glat, glon = np.meshgrid(lat, lon)
#     gmlat = 0.5*(glat[1:, 1:] + glat[:-1, :-1])
#     gmlon = 0.5*(glon[1:, 1:] + glon[:-1, :-1])

#     dx = gsw.distance(glon, glat, axis=0)
#     dy = gsw.distance(glon, glat, axis=1)

#     dsla_x = var.values[1:] - var.values[:-1]
#     dsla_y = var.values[:, 1:, :] - var.values[:, :-1, :]

#     # take differences along one axis ..
#     dsla_dx = dsla_x/dx[:, :, np.newaxis]
#     dsla_dy = dsla_y/dy[:, :, np.newaxis]

#     # average along the other direction
#     dsla_dx_mid = 0.5*(dsla_dx[:, 1:, :]+dsla_dx[:, :-1, :])
#     dsla_dy_mid = 0.5*(dsla_dy[1:, :, :]+dsla_dy[:-1, :, :])

#     # scale by f and g
#     f = gsw.f(gmlat)
#     g =  gsw.grav(gmlat, 0)

#     vg_rec = dsla_dx_mid * (g/f)[:, :, np.newaxis]
#     ug_rec = (-1) * dsla_dy_mid * (g/f)[:, :, np.newaxis]

#     return ug_rec, vg_rec, gmlon, gmlat



def geos_vel(dot, lat, lon):
  """
  dot: shape (lon, lat, time)
  """
  lon_ext = np.hstack((lon, lon[-1] + 1))
  dot_ext = np.concatenate((dot, dot[0:1,:,:]), axis=0)

  #since map is circumpolar append first col to the end
  glat, glon = np.meshgrid(lat, lon)
  glat_ext, glon_ext = np.meshgrid(lat, lon_ext)

  # scale by f and g
  f = gsw.f(glat)
  g =  gsw.grav(glat, 0)

  # distance between grid points
  # given the grid is uniform, no need to flip the arrays
  dx = gsw.distance(glon_ext, glat_ext, axis=0)  # lon
  dy = gsw.distance(glon, glat, axis=1)  # lat

  # ---------- u vel ------------------
  # a. forward diff
  dnx_forw = dot_ext[1:] - dot_ext[:-1]
  grad_x_forw = dnx_forw / dx[:, :, np.newaxis]

  grad_x_avg = 0.5 * (grad_x_forw[1:] + grad_x_forw[:-1])
  grad_x = np.concatenate((grad_x_forw[0:1, : , :], grad_x_avg), axis=0)

  vg = grad_x * (g/f)[:, :, np.newaxis]

  # ---------- v vel ------------------
  # a. forward diff
  dny_forw = dot[:, 1:, :] - dot[:, :-1, :]
  grad_y_forw = dny_forw / dy[:, :, np.newaxis]

  grad_y_avg = 0.5 * (grad_y_forw[:,1:,:] + grad_y_forw[:,:-1,:])
  grad_y = np.concatenate((grad_y_forw[:, 0:1, :], 
                          grad_y_avg, 
                          grad_y_forw[:, -1, :][:,np.newaxis,:]), axis=1)

  ug = (-1) * grad_y * (g/f)[:, :, np.newaxis]
  return ug, vg


def extract_eof(n_eof, dot, lat, lon):
    alt_var = dot

    glat, glon = np.meshgrid(lat, lon)

    # apply weighting to account for meridians converging at high lat
    coslat = np.cos(np.deg2rad(lat)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(alt_var.T, weights=wgts)

    #eof = solver.eofsAsCorrelation(neofs=15)
    eof = solver.eofsAsCovariance(neofs=10)
    pc = solver.pcs(npcs=10, pcscaling=1)
    variance_frac = solver.varianceFraction()

    # - - - - - - - - - - - - - - - - - 
    # > > reconstruct DOT using the first 3 modes
    rec_dot = solver.reconstructedField(n_eof)

    ug_rec, vg_rec = geos_vel(rec_dot.T.values,
                            rec_dot.latitude.values,
                            rec_dot.longitude.values)

    # - - - - - - - - - - - - - - - - - 
    rec = xr.Dataset({'dot' : (('longitude', 'latitude', 'time'), rec_dot.T.values),
                      'ug' : (('longitude', 'latitude', 'time'), ug_rec),
                      'vg' : (('longitude', 'latitude', 'time'), vg_rec)},
                      coords={'longitude' : rec_dot.longitude.values,
                              'latitude' : rec_dot.latitude.values,
                              'time' : rec_dot.time.values})
    return rec


def plot_sla_ts(sla_var, uc_var, clim_idx, clim_label, title):
    xtim1 = sla_var.time.values
    xtim2 = uc_var.time.values
    xtim3 = clim_idx.time.values
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(xtim1, sla_var.values, c='m', label='SLA')
    ax.plot(xtim2, uc_var.values, c='k', label='UC')
    ax.plot(xtim3, clim_idx.values, c='b', label=clim_label)
    ax.set_title(title, loc='left')
    ax.legend()
    # spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # date axis
    years = mdates.YearLocator(1, month=1, day=1)   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    ax.format_xdata = mdates.DateFormatter('%Y')
    fig.autofmt_xdate()

    ax.grid(True, which='major', lw=1., ls='-')
    ax.grid(True, which='minor', lw=1., ls=':')
    plt.tight_layout()

# -------------------------------
# Vectorized binning function
# -------------------------------
def bin_mean_count(lon, lat, grid, ssh):

    edges_lon = grid["edges_lon"]
    edges_lat = grid["edges_lat"]
    mid_lon = grid["mid_lon"]
    mid_lat = grid["mid_lat"]
    total_area = grid["area"]

    lat_idx = np.digitize(lat, edges_lat) - 1
    lon_idx = np.digitize(lon, edges_lon) - 1

    valid = (lat_idx >= 0) & (lat_idx < len(mid_lat)) & \
            (lon_idx >= 0) & (lon_idx < len(mid_lon))
    lat_idx = lat_idx[valid]
    lon_idx = lon_idx[valid]
    values = ssh[valid]

    ssh_sum = np.zeros((len(mid_lon), len(mid_lat)))
    ssh_count = np.zeros((len(mid_lon), len(mid_lat)))
    np.add.at(ssh_sum, (lon_idx, lat_idx), values)
    np.add.at(ssh_count, (lon_idx, lat_idx), 1)

    ssh_grid = np.full((len(mid_lon), len(mid_lat)), np.nan)
    mask = ssh_count > 0
    ssh_grid[mask] = ssh_sum[mask] / ssh_count[mask]

    return ssh_grid, ssh_count

# -------------------------------
# Compute offsets for one file
# -------------------------------
def compute_offsets_ocean_leads(fn, ncdir, grid, min_points=30, use_area_weights=True):
    edges_lon = grid["edges_lon"]
    edges_lat = grid["edges_lat"]
    mid_lon = grid["mid_lon"]
    mid_lat = grid["mid_lat"]
    total_area = grid["area"]

    try:
        ds = xr.open_dataset(ncdir + fn + ".nc", engine="netcdf4")
    except FileNotFoundError:
        print(f"MERGE {fn} not available")
        return np.nan, np.nan, np.nan, None

    # Check required variables
    required_vars = ['Latitude', 'Longitude', 'Elevation', 'SurfaceType', 'distance_m', 'MeanSSH', 'Time']
    if not all(var in ds.variables for var in required_vars) or ds.sizes == {}:
        print(f"MERGE {fn} is empty or missing required variables")
        return np.nan, np.nan, np.nan, None

    # Safe to extract now
    surface = ds['SurfaceType'].values
    mss = ds['MeanSSH'].values
    ssh = ds['Elevation'].values
    lon = ds['Longitude'].values
    lat = ds['Latitude'].values
    dist = ds['distance_m'].values


    ssh = ssh[dist>1e4]
    lon = lon[dist>1e4]
    lat = lat[dist>1e4]
    surface = surface[dist>1e4]
    mss = mss[dist>1e4]

    ocean_mask = surface ==1
    lead_mask = surface ==2

    ssh_ocean, count_ocean = bin_mean_count(
        lon[ocean_mask], lat[ocean_mask],
        grid,
        ssh[ocean_mask]
    )

    ssh_lead, count_lead = bin_mean_count(
        lon[lead_mask], lat[lead_mask],
        grid,
        ssh[lead_mask]
    )

    mss_ocean_binned, _ = bin_mean_count(
        lon[ocean_mask], lat[ocean_mask],
        grid,
        mss[ocean_mask]
    )

    mss_lead_binned, _ = bin_mean_count(
        lon[lead_mask], lat[lead_mask],
        grid,
        mss[lead_mask]
    )

    ssha_ocean = ssh_ocean - mss_ocean_binned
    ssha_lead = ssh_lead - mss_lead_binned

    #apply threshold to only keep bins with enough points
    valid_ocean = (count_ocean > min_points)
    valid_lead = (count_lead > min_points)

    #------------------------------------------------------------------
    # discard bins with fewer points than a certain threshold
    # ssh_ocean[~valid_ocean] = np.nan
    # ssh_lead[~valid_lead] = np.nan
    #offset = ssh_ocean - ssh_lead

    ssha_ocean[~valid_ocean] = np.nan
    ssha_lead[~valid_lead] = np.nan

    offset = ssha_ocean - ssha_lead

    if use_area_weights:
        weights = np.copy(total_area)
    else:
        weights = np.ones_like(offset)

    # weights[np.isnan(offset)] = 0
    # weights /= np.nansum(weights)
    #
    # mean_offset = np.nansum(offset * weights)
    # std_offset = np.sqrt(np.nansum(((offset - mean_offset)**2) * weights))

    mask = ~np.isnan(offset)
    values = offset[mask]
    w = weights[mask]

    if w.sum() ==0:
        return np.nan, np.nan, np.nan, offset

    w = w / np.sum(w)   # normalize

    #weighted mean
    mean_offset = np.sum(values * w)

    #weighted variance and std
    variance    = np.sum(w * (values - mean_offset) ** 2)
    std_offset  = np.sqrt(variance)

    median_offset = np.nanmedian(offset)

    return mean_offset, std_offset, median_offset, offset

def wrap_longitudes(ds, lon_name='longitude'):
    """
    Return a copy of the dataset with longitudes wrapped to 0–360
    and sorted so the grid is continuous and monotonic.

    Works for xarray Dataset or DataArray.
    """

    # Extract original longitudes
    lon = ds[lon_name].values

    # Convert to 0–360
    lon360 = np.where(lon < 0, lon + 360, lon)

    # Sorting index
    sort_idx = np.argsort(lon360)

    # Assign wrapped longitudes and reorder the dimension
    ds_wrap = ds.assign_coords({lon_name: lon360})
    ds_wrap = ds_wrap.isel({lon_name: sort_idx})

    return ds_wrap


def wrap_longitudes_for_sector(ds, lonmin, lonmax, lon_name='longitude'):

    lon = ds[lon_name].values
    lon360 = np.where(lon < 0, lon + 360, lon)

    sort_idx = np.argsort(lon360)
    lon_sorted = lon360[sort_idx]

    if lonmin > lonmax:
        start_idx = np.searchsorted(lon_sorted, lonmin)

        lon_rot = np.concatenate([lon_sorted[start_idx:], lon_sorted[:start_idx]])
        rot_idx = np.concatenate([sort_idx[start_idx:], sort_idx[:start_idx]])

        rotation_offset = lon_sorted[start_idx]

    else:
        lon_rot = lon_sorted
        rot_idx = sort_idx
        rotation_offset = 0.0

    ds_rot = ds.assign_coords({lon_name: lon_rot})
    ds_rot = ds_rot.isel({lon_name: rot_idx})

    return ds_rot, rotation_offset

def split_and_shift_gdf(gdf, lons_shifted):
    x_min, x_max = lons_shifted[0], lons_shifted[-1]
    domain = box(x_min, lat_min, x_max, lat_max)

    # Keep both original and +360 shifted versions of every geometry
    original = gdf.copy()
    shifted  = gdf.copy()
    shifted['geometry'] = shifted.geometry.apply(lambda g: translate(g, xoff=360))

    combined = pd.concat([original, shifted])

    # Clip to domain — only geometries that fall within the domain survive
    combined['geometry'] = combined.geometry.apply(
        lambda g: g.intersection(domain) if g is not None and not g.is_empty else g
    )
    combined = combined[~combined.geometry.is_empty]
    return combined

def clip_gdf_to_domain(gdf, lon_min, lon_max, lat_min, lat_max):
    domain = box(lon_min, lat_min, lon_max, lat_max)
    clipped = gdf.copy()
    clipped['geometry'] = clipped.geometry.apply(
        lambda g: g.intersection(domain) if g is not None and not g.is_empty else g
    )
    clipped = clipped[~clipped.geometry.is_empty]
    return clipped

def is_valid_contour(verts, max_aspect_ratio=0.5):
    """
    Returns False if the contour is suspiciously meridional.
    max_aspect_ratio: reject if lon_span / lat_span < this value
    e.g. 0.5 means contour must be at least half as wide as it is tall
    """
    lon_span = verts[:, 0].max() - verts[:, 0].min()
    lat_span = verts[:, 1].max() - verts[:, 1].min()

    if lat_span == 0:
        return True   # flat contour, not meridional

    aspect = lon_span / lat_span
    return aspect >= max_aspect_ratio

def mean_inside_contour(verts, field_2d, lon_grid, lat_grid):
    """Compute mean of field_2d inside the closed contour verts."""
    path   = Path(verts)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    mask   = path.contains_points(points).reshape(field_2d.shape)
    vals   = field_2d[mask]
    return np.nanmean(vals) if len(vals) > 0 else np.nan

def area_deg2_to_km2(area_deg2, verts):
    """
    Convert area from degrees² to km² accounting for latitude.
    Uses the mean latitude of the contour vertices.
    """
    mean_lat = np.mean(verts[:, 1])
    # 1 degree latitude ≈ 111.32 km
    # 1 degree longitude ≈ 111.32 * cos(lat) km
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(mean_lat))
    return area_deg2 * km_per_deg_lat * km_per_deg_lon

def points_inside_contour(verts, lon_grid, lat_grid):
    """
    Returns a boolean 2D array (same shape as lon_grid / lat_grid) that is
    True for every grid point whose (lon, lat) lies inside the polygon
    defined by verts.
    """
    from matplotlib.path import Path
    path   = Path(verts)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    mask   = path.contains_points(points)
    return mask.reshape(lon_grid.shape)

def sample_field_at_vertices(verts, var_2d, lon_grid, lat_grid):
    """
    Nearest-neighbour SSH sample at each contour vertex; returns the mean.
    """
    lons = lon_grid[0, :]
    lats = lat_grid[:, 0]
    vals = []
    for vx, vy in verts:
        ix = np.argmin(np.abs(lons - vx))
        iy = np.argmin(np.abs(lats - vy))
        v  = var_2d[iy, ix]
        if np.isfinite(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan

def load_climate_index(filepath):
    years, monthly_vals = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 13:
                try:
                    years.append(int(parts[0]))
                    monthly_vals.append([float(x) for x in parts[1:]])
                except ValueError:
                    continue
    yrs = np.array(years)
    ind_1d = np.array(monthly_vals).flatten()
    time = pd.date_range(start=f'{yrs[0]}-01', periods=len(ind_1d), freq='MS')
    da = xr.DataArray(ind_1d, dims=['time'], coords={'time': time})
    print(f'Loaded {filepath.split("/")[-1]}: {len(yrs)} years ({yrs[0]}-{yrs[-1]})')
    return da