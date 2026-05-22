"""
Script to calculate wsc and osc. Saves them to a file for easy access.
Can set start and end times to fit the period of interest (this won't cahnge the calcualtion so better to just
.sel (time = start, end) in a future script

Last modified: 22/05/2026
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import xeofs as xe
from pygments.modeline import modeline_re
import sys
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
import warnings
warnings.filterwarnings('ignore')

# path = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
# save_dir = path + 'Oana/test_figs/'
# aux_scripts = path + 'Scripts/aux_scripts/'
# print(path)

workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
mca_dir = data_dir + 'mca_processing/'
clim_dir = data_dir + 'climate_indices/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/'
vars_dir = mca_dir + 'preprocessed_vars/'


sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func

# load datasets (coordinates have been rotated / interpolated as needed)
dot_ds = xr.open_dataset('/Users/iw2g24/PycharmProjects/SLA_analysis/Data/dot_all_30bmedian_egm2008_sig3.nc')
era5_ds = xr.open_dataset(workdir + 'Data/ERA5/era5_regridded_2002_2024_monthly.nc')
sic_ds = xr.open_dataset(workdir + 'Data/NOAA_SIC_monthly/noaa_sic_2025_corrected.nc')
sid_ds  = xr.open_dataset(workdir + 'Data/NASA_SID_weekly/nasa_sid_2025_corrected.nc')

# select time range for vairable calculation
start_time = '2002-07-01'
end_time = '2024-12-01'

# constants
rho_a = 1.25 # air density kg m-3
cd_aw = 1.25e-3 # drag coeff
rho_w = 1028
cd_iw = 5.5e-3

# Oana's ds
# dot_ds = dot_ds.sel(time=slice('2002-07-01', '2018-10-01'))

# Phased response paper
# dot_ds = dot_ds.sel(time=slice('2011-01-01', '2016-01-01'))

# full ds time
dot_ds = dot_ds.sel(time=slice(start_time, end_time))

# select/extract variables
dot = dot_ds.dot.values
lon = dot_ds.longitude.values
lat = dot_ds.latitude.values
dot_time = dot_ds.time.values

# create a seamask
seamask = dot[:,:,0] / dot[:,:,0]
seamask[seamask == 0] = np.nan

# set other variable times to dot time (this manages the masked months)
era5_ds = era5_ds.sel(time=dot_time)
sic_ds = sic_ds.sel(time=dot_time)
sid_ds = sid_ds.sel(time=dot_time)

# create 2D array of lon and lat grid using meshgrid
# For plotting (matches (lon, lat) data arrays)
llon_ij, llat_ij = np.meshgrid(lon, lat, indexing='ij')  # shape (lon, lat)

# For grad_sphere (expects lat, lon) - grad_sphere calculates gradient at a lon/lat location on sphere (uses )
llon, llat = np.meshgrid(lon, lat)  # shape (lat, lon)
# print("llon edges:", llon[0, 0], llon[0, -1])  # -179.5, 179.5
# print("llat edges:", llat[0, 0], llat[-1, 0])  # lat min/max

print('selecting wind data...')

# select wind
u10 = era5_ds['u10'].values
v10 = era5_ds['v10'].values

print('applying seamask...')
u10 = u10*seamask[:,:, None]
v10 = v10*seamask[:,:, None]
U_air = np.sqrt(u10**2. + v10**2.)

print('calculating WS...')
# wind stress
tau_x = rho_a*cd_aw*U_air*u10 #zonal wind
tau_y = rho_a*cd_aw*U_air*v10 #meridional wind

# print('zonal wind:',  tau_x.shape)
# print('meridional wind:', tau_y.shape)

print('extracting SIC and SID data...')
# Sea ice concentration (α)
alpha = sic_ds['cdr_seaice_conc_monthly'].values
alpha = np.clip(np.where(np.isnan(alpha), np.nan, alpha), 0, 1)
# alpha_filled = np.where((np.isnan(alpha)) | (alpha < 0.15), 0.0, alpha) # cut off at alpha - 0.15 as this is the edge of the ice pack (ref this)
alpha_filled = alpha.copy()

# Ice drift — raw values, NaN→0
u_ice_raw = np.nan_to_num(sid_ds.u_rotated.values * 1e-2, nan=0.0)
v_ice_raw = np.nan_to_num(sid_ds.v_rotated.values * 1e-2, nan=0.0)

# # ── Smooth SID with same gaussian_filt as OSC (sig=3) ────────────── OPTIONAL ---
# u_ice = np.zeros_like(u_ice_raw)
# v_ice = np.zeros_like(v_ice_raw)
#
# for t in range(u_ice_raw.shape[2]):
#     u_t     = u_ice_raw[:, :, t]
#     v_t     = v_ice_raw[:, :, t]
#     mask_t  = (u_t != 0.0).astype(float)
#
#     u_smooth = ft.gaussian_filt(u_t,    sigma=3, mode='reflect')
#     v_smooth = ft.gaussian_filt(v_t,    sigma=3, mode='reflect')
#     weight   = ft.gaussian_filt(mask_t, sigma=3, mode='reflect')
#
#     with np.errstate(invalid='ignore'):
#         u_ice[:, :, t] = np.where(weight > 0.1, u_smooth / weight, 0.0)
#         v_ice[:, :, t] = np.where(weight > 0.1, v_smooth / weight, 0.0)

u_ice = u_ice_raw
v_ice = v_ice_raw

print('applying seamask ...')
u_ice = u_ice * seamask[:, :, None]
v_ice = v_ice * seamask[:, :, None]
U_ice = np.sqrt(u_ice**2 + v_ice**2)

print('calcualting OSS components...')
tau_iw_x = rho_w * cd_iw * U_ice * u_ice
tau_iw_y = rho_w * cd_iw * U_ice * v_ice
tau_aw_x = tau_x
tau_aw_y = tau_y

oss_x = alpha_filled * tau_iw_x + (1 - alpha_filled) * tau_aw_x
oss_y = alpha_filled * tau_iw_y + (1 - alpha_filled) * tau_aw_y

seamask_nan = np.where(seamask == 1, 1.0, np.nan)
oss_x = oss_x * seamask_nan[:, :, None]
oss_y = oss_y * seamask_nan[:, :, None]

oss_x[0, :, :] = oss_x[-1, :, :]
oss_y[0, :, :] = oss_y[-1, :, :]

oss_x_filled = (xr.DataArray(oss_x, dims=("lon", "lat", "time"))
                .ffill("lon").bfill("lon").values)
oss_y_filled = (xr.DataArray(oss_y, dims=("lon", "lat", "time"))
                .ffill("lon").bfill("lon").values)

oss_x_filled = oss_x_filled * seamask_nan[:, :, None]
oss_y_filled = oss_y_filled * seamask_nan[:, :, None]

oss_mag = np.sqrt(oss_x_filled**2 + oss_y_filled**2)
print("OSS ready, shape:", oss_x_filled.shape)
print("OSS NaN fraction:", np.isnan(oss_x_filled[:,:,0]).mean())
print("OSS magnitude:", np.nanmax(np.abs(oss_x_filled)))


print('============================')
print('calculating variable curls')
print('===========================')

print('calcuating WSC')
# WSC
ws_curl = np.zeros_like(tau_x)
for t in range(dot.shape[2]):
    dtaux_dx, dtaux_dy = grad_sphere(tau_x[:,:,t].T, llon, llat)
    dtauy_dx, dtauy_dy = grad_sphere(tau_y[:,:,t].T, llon, llat)
    ws_curl[:,:,t] = (dtauy_dx - dtaux_dy).T

ws_curl[~np.isfinite(ws_curl)] = np.nan
print("ws_curl shape:", ws_curl.shape)
print("ws_curl magnitude:", np.nanmax(np.abs(ws_curl)))

print('calcuating OSC')
# OSC
os_curl = np.zeros_like(oss_x_filled)
for t in range(dot.shape[2]):
    dtaux_dx, dtaux_dy = grad_sphere(oss_x_filled[:,:,t].T, llon, llat)
    dtauy_dx, dtauy_dy = grad_sphere(oss_y_filled[:,:,t].T, llon, llat)
    os_curl[:,:,t] = (dtauy_dx - dtaux_dy).T

os_curl[~np.isfinite(os_curl)] = np.nan
print("os_curl shape:", os_curl.shape)
print("os_curl magnitude:", np.nanmax(np.abs(os_curl)))

print('remove anything >5std for WSC and OSC')
# Clip extreme outliers before MCA — anything beyond 5 std is likely an artefact
wsc_std = np.nanstd(ws_curl)
ws_curl_clipped = np.clip(ws_curl, -5 * wsc_std, 5 * wsc_std)
print("WSC after clipping max:", np.nanmax(np.abs(ws_curl_clipped)))

osc_std = np.nanstd(os_curl)
os_curl_clipped = np.clip(os_curl, -5 * osc_std, 5 * osc_std)
print("OSC after clipping max:", np.nanmax(np.abs(os_curl_clipped)))

ds_out = xr.Dataset(
    {
    'wsc': (('lon', 'lat', 'time'), ws_curl_clipped),
    'osc': (('lon', 'lat', 'time'), os_curl_clipped),
    },
    coords ={
        "lon": lon,
        "lat": lat,
        "time": dot_time,
    },
    attrs = {
    'description': f'WSC and OSC variables calculated between {start_time} and {end_time} in A0_calc_wsc_osc.py ',
})


print('saving wsc and osc to nc file')
outfile = f'wsc_osc_{start_time}_{end_time}.nc'
ds_out.to_netcdf(vars_dir + outfile)
print(f'saved {outfile} to {vars_dir}{outfile}!')
