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
mca_dir = data_dir + 'mca_processing/full/'
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


# select time range for vairable calculation
start_time = '2002-07-01'
end_time = '2024-12-01'

# constants
rho_a = 1.25 # air density kg m-3
cd_aw = 1.25e-3 # drag coeff
rho_w = 1028
cd_iw = 5.5e-3

# load dot dataset
dot_ds = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
# full ds time
dot_ds = dot_ds.sel(time=slice(start_time, end_time))
dot = dot_ds.dot.values
dot_time = dot_ds.time.values

# load wsc and osc ds
vars_ds = xr.open_dataset(vars_dir + f'wsc_osc_{start_time}_{end_time}.nc')
ws_curl = vars_ds.wsc.values
os_curl = vars_ds.osc.values

months = (dot_time.astype('datetime64[M]').astype(int) % 12 + 1)
print(months)

# 1. Remove time mean at every grid cell
sla = dot - np.nanmean(dot, axis=2, keepdims=True)
wsc_demeaned = ws_curl - np.nanmean(ws_curl, axis=2, keepdims=True)
osc_demeaned = os_curl - np.nanmean(os_curl, axis=2, keepdims=True)

# 2. Remove linear trend at every grid cell (on de-meaned data)
import pandas as pd
time_years = (pd.DatetimeIndex(dot_time).year
              + pd.DatetimeIndex(dot_time).month / 12.0).values

for name, field_in in [('sla', sla),
                        ('wsc', wsc_demeaned),
                        ('osc', osc_demeaned)]:
    arr = field_in.transpose(2, 0, 1)  # (time, lon, lat)
    n, slope, intercept, _, _, _ = linregress_3D_spatial_time(arr, time_years)
    trend = (slope[np.newaxis,:,:] * time_years[:,np.newaxis,np.newaxis]
             + intercept[np.newaxis,:,:])
    # detrended = arr - trend
    detrended = (arr - trend)
    detrended = detrended.transpose(1, 2, 0)  # back to (lon, lat, time)

    if name == 'sla': sla_detrended = detrended
    elif name == 'wsc': wsc_detrended = detrended
    elif name == 'osc': osc_detrended = detrended

# 3. Retain seasonal cycle (remove_seas = False))

remove_seas = False  # keep seasonal cycle as interested in this

if remove_seas:
    # Preallocate seasonal arrays
    sla_seas = np.full((12, dot.shape[0], dot.shape[1]), np.nan)
    wsc_seas = np.full((12, dot.shape[0], dot.shape[1]), np.nan)
    osc_seas = np.full((12, dot.shape[0], dot.shape[1]), np.nan)

    # Compute monthly climatology
    for m in range(1, 13):
        mask = (months == m)
        if np.any(mask):
            sla_seas[m-1] = np.nanmean(sla_detrended[:,:,mask], axis=2)
            wsc_seas[m-1] = np.nanmean(wsc_detrended[:,:,mask], axis=2)
            osc_seas[m-1] = np.nanmean(osc_detrended[:,:,mask], axis=2)

    # Remove seasonality
    sla_final = np.zeros_like(sla_detrended)
    wsc_final = np.zeros_like(wsc_detrended)
    osc_final = np.zeros_like(osc_detrended)
    for i, m in enumerate(months):
        sla_final[:,:,i] = sla_detrended[:,:,i] - sla_seas[m-1]
        wsc_final[:,:,i] = wsc_detrended[:,:,i] - wsc_seas[m-1]
        osc_final[:,:,i] = osc_detrended[:,:,i] - osc_seas[m-1]

    print("Seasonal cycle removed.")

else:
    sla_final = sla_detrended.copy()
    wsc_final = wsc_detrended.copy()
    osc_final = osc_detrended.copy()

    print("Seasonal cycle retained.")


print('adding detrended variables...')
vars_ds = vars_ds.assign(
    sla_detrended = (('lon', 'lat', 'time'), sla_final),
    wsc_detrended = (('lon', 'lat', 'time'), wsc_final),
    osc_detrended = (('lon', 'lat', 'time'), osc_final),
)

vars_ds['sla_detrended'].attrs = {
    'long_name': 'Sea Level Anomaly detrended',
    'units': 'm'
}

vars_ds['wsc_detrended'].attrs = {
    'long_name': 'wind stress curl detrended',
    'units': 'N/m^3',
}
vars_ds['osc_detrended'].attrs = {
    'long_name': 'ocean stress curl detrended',
    'units': 'N/m^3',
}


print('save detrended variables to file...')
outfile = f"wsc_osc_sla_det_{start_time}_{end_time}.nc"
vars_ds.to_netcdf(vars_dir+ outfile)
print(f'detrended file saved to {vars_dir+ outfile}')