"""
0b_detrending_sla_sha_eha_ds.py

detrend sla (from my product) and sha/eha from jenny cocks' product
add to original datasets but save as a copy with _det to make sure the original files are not overwritten
"""

import sys
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import os
import xeofs as xe
from pygments.modeline import modeline_re
import sys
import cartopy.crs as ccrs

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
script_dir = workdir + 'Scripts/'
data_dir = workdir + 'Data/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/F_steric_height/'
os.makedirs(fig_dir, exist_ok=True)

sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func

sla_ds = xr.open_dataset(data_dir+'dot_all_30bmedian_egm2008_sig3.nc')
steric_height_ds_2024 = xr.open_dataset(data_dir + 'steric_height_cocks/steric_height_2002_2024.nc')
print(steric_height_ds_2024)
print(sla_ds)


sha_time_initial = steric_height_ds_2024.time

# align first so both datasets share an identical time axis
steric_height_ds_2024, sla_ds = xr.align(
    steric_height_ds_2024, sla_ds, join="inner"
)

# now build the mask from the (now-shared) time axis
bad_month_mask = pd.to_datetime([
    '2017-09-01', '2017-10-01', '2017-11-01', '2017-12-01',
    '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
    '2020-05-01', '2020-06-01', '2024-02-01'
])
mask = ~steric_height_ds_2024.time.isin(bad_month_mask)

steric_height_ds_2024 = steric_height_ds_2024.where(mask, drop=True)
sla_ds = sla_ds.where(mask, drop=True)

lon = steric_height_ds_2024.longitude
lat = steric_height_ds_2024.latitude
sha = steric_height_ds_2024.sha
eha = steric_height_ds_2024.eha
sha_time = steric_height_ds_2024.time
sla = sla_ds.sla





months = (sha_time.astype('datetime64[M]').astype(int) % 12 + 1)
print(months)

# 2. Remove linear trend at every grid cell (on de-meaned data)
time_years = (pd.DatetimeIndex(sha_time).year
              + pd.DatetimeIndex(sha_time).month / 12.0).values

for name, field_in in [('sla', sla),
                        ('sha', sha),
                       ('eha', eha),]:
    # Ensure consistent axis order
    arr = field_in.transpose("time", "latitude", "longitude").values

    # Regress each grid cell against time
    n, slope, intercept, _, _, _ = linregress_3D_spatial_time(arr, time_years)

    trend = slope[None,:,:] * time_years[:,None,None] + intercept[None,:,:]
    detrended = arr - trend

    # Back to (time, lat, lon)
    detrended = detrended.transpose(0,1,2)

    if name == 'sla':
        sla = detrended
    elif name == 'sha':
        sha = detrended
    elif name == 'eha':
        eha = detrended

# 3. Retain seasonal cycle (remove_seas = False))

remove_seas = False  # keep seasonal cycle as interested in this

if remove_seas:
    # Preallocate seasonal arrays
    sla_seas = np.full((12, sla.shape[0], sla.shape[1]), np.nan)
    sha_seas = np.full((12, sha.shape[0], sha.shape[1]), np.nan)
    eha_seas = np.full((12, eha.shape[0], eha.shape[1]), np.nan)

    # Compute monthly climatology
    for m in range(1, 13):
        mask = (months == m)
        if np.any(mask):
            sla_seas[m-1] = np.nanmean(sla[:,:,mask], axis=2)
            sha_seas[m-1] = np.nanmean(sha[:,:,mask], axis=2)
            eha_seas[m-1] = np.nanmean(eha[:,:,mask], axis=2)


    # Remove seasonality
    sla_final = np.zeros_like(sla)
    sha_final = np.zeros_like(sha)
    eha_final = np.zeros_like(eha)

    for i, m in enumerate(months):
        sla_final[:,:,i] = sla[:,:,i] - sla_seas[m-1]
        sha_final[:,:,i] = sha[:,:,i] - sha_seas[m-1]
        eha_final[:,:,i] = eha[:,:,i] - eha_seas[m-1]

    print("Seasonal cycle removed.")

else:
    sla_final = sla.copy()
    sha_final = sha.copy()
    eha_final = eha.copy()

    print("Seasonal cycle retained.")


print("Preparing detrended datasets...")

# SLA dataset
sla_ds_det = sla_ds.copy()
sla_ds_det["sla_det"] = (("time","lat","lon"), sla_final)

sla_outfile = "sla_det_2002_2024.nc"
sla_ds_det.to_netcdf(data_dir + sla_outfile)

print(f"SLA detrended file saved to {data_dir + sla_outfile}")


# SHA dataset
sha_ds_det = steric_height_ds_2024.copy()

sha_ds_det["sha_det"] = (("time","lat","lon"), sha_final)
sha_ds_det["eha_det"] = (("time","lat","lon"), eha_final)

sha_outfile = "steric_height_det_2002_2024.nc"
sha_ds_det.to_netcdf(data_dir + "steric_height_cocks/" + sha_outfile)

print(f"SHA detrended file saved to {data_dir + "steric_height_cocks/" + sha_outfile}")