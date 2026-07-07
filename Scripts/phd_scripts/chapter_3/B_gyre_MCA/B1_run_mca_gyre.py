"""
B1_run_mca_gyre.py
Load full grid OSC and WSC from A0.
Crop to selected region / gyre
Calculate MCA for this region / gyre
Save scores and comps

last modified 27/05/2026
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import xeofs as xe
import sys
import os
import warnings
warnings.filterwarnings('ignore')

workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
mca_dir = data_dir + 'mca_processing/sectors/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'

sys.path.append(auxscriptdir)

# -----------------------------
# 1. SETTINGS
# -----------------------------
start_time = '2002-07-01'
end_time   = '2024-12-01'

gyre_name = 'kerguelen'   # ross / weddell / kerguelen

if gyre_name == 'ross':
    lon_min = 140
    lon_max = 280     # NOTE: 280 instead of -80 (0–360 grid)
    lat_min = -90
    lat_max = -60
    sector_dir = mca_dir + 'ross/'
elif gyre_name == 'weddell':
    # Weddell spans 280–360 and 0–50
    lon_min1, lon_max1 = 280, 360
    lon_min2, lon_max2 = 0, 50
    sector_dir = mca_dir + 'weddell/'
elif gyre_name == 'kerguelen':
    lon_min = 60
    lon_max = 150
    sector_dir = mca_dir + 'kerguelen/'

vars_dir = sector_dir + 'preprocessed_vars/'
scores_comps_dir = sector_dir + 'scores_comps/'
os.makedirs(scores_comps_dir, exist_ok=True)

# -----------------------------
# 2. LOAD GLOBAL DETRENDED DATASET
# -----------------------------
full = xr.open_dataset(
    '/Users/iw2g24/PycharmProjects/SLA_analysis/Data/mca_processing/full/preprocessed_vars/wsc_osc_sla_det_2002-07-01_2024-12-01.nc'
)

# Convert to 0–360° longitude
full = full.assign_coords(lon=(full.lon % 360)).sortby('lon')

# -----------------------------
# 3. CROP TO GYRE SECTOR
# -----------------------------
if gyre_name == 'ross':
    sector = full.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
elif gyre_name == 'weddell':
    sec1 = full.sel(lon=slice(lon_min1, lon_max1))
    sec2 = full.sel(lon=slice(lon_min2, lon_max2))
    # Shift sec2 longitudes by +360 so they follow sec1
    sec2 = sec2.assign_coords(lon=sec2.lon + 360)
    sector = xr.concat([sec1, sec2], dim='lon')
elif gyre_name == 'kerguelen':
    sector = full.sel(lon=slice(lon_min, lon_max))

# Extract arrays
wsc_det = sector.wsc_detrended.values
osc_det = sector.osc_detrended.values
sla_det = sector.sla_detrended.values

# from scipy.ndimage import gaussian_filter
#
# def smooth_2d_time_nanaware(field, sigma=3):
#     """
#     field: (lon, lat, time), with NaNs
#     returns: same shape, smoothed in lon/lat for each time
#     """
#     out = np.full_like(field, np.nan)
#     T = field.shape[2]
#
#     for t in range(T):
#         f = field[:, :, t]
#         mask = np.isfinite(f).astype(float)
#
#         num = gaussian_filter(np.nan_to_num(f, nan=0.0), sigma=sigma, mode='reflect')
#         den = gaussian_filter(mask, sigma=sigma, mode='reflect')
#
#         with np.errstate(invalid='ignore', divide='ignore'):
#             out[:, :, t] = np.where(den > 0, num / den, np.nan)
#
#     return out
#
# # Optional: smooth OSC (and/or WSC) in space
# apply_smoothing = True
#
# if apply_smoothing:
#     print("Applying Gaussian smoothing (sigma=3) to OSC...")
#     osc_det = smooth_2d_time_nanaware(osc_det, sigma=1)
#     # If you also want WSC smoothed:
#     # print("Applying Gaussian smoothing (sigma=3) to WSC...")
#     # wsc_det = smooth_2d_time_nanaware(wsc_det, sigma=3)
#

from scipy.ndimage import gaussian_filter

def smooth_nanaware(field, sigma=3):
    """
    field: (lon, lat, time) with NaNs
    returns: smoothed field with NaNs preserved
    """
    out = np.full_like(field, np.nan)
    T = field.shape[2]

    for t in range(T):
        f = field[:, :, t]
        mask = np.isfinite(f).astype(float)

        # smooth numerator and denominator
        num = gaussian_filter(np.nan_to_num(f, nan=0.0), sigma=sigma, mode='reflect')
        den = gaussian_filter(mask, sigma=sigma, mode='reflect')

        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, :, t] = np.where(den > 0, num / den, np.nan)

    return out

# Extract arrays
wsc_det = sector.wsc_detrended.values
osc_det = sector.osc_detrended.values
sla_det = sector.sla_detrended.values

# Build seamask from SLA (or from original DOT)
seamask = np.isfinite(sla_det[:, :, 0]).astype(float)
seamask[seamask == 0] = np.nan
seamask_3d = seamask[:, :, None]  # broadcast over time

# Smooth OSC
print("Smoothing OSC with Gaussian sigma=1...")
osc_smooth = smooth_nanaware(osc_det, sigma=1)

# Reapply seamask so land matches SLA
osc_smooth = osc_smooth * seamask_3d
osc_det = osc_smooth

lon = sector.lon.values
lat = sector.lat.values
dot_time = sector.time.values

# -----------------------------
# 4. BUILD XARRAYS FOR MCA
# -----------------------------
sla_xa = xr.DataArray(
    sla_det.transpose(2,1,0),
    dims=("time","lat","lon"),
    coords={"time":dot_time, "lat":lat, "lon":lon},
    name="sla"
)

wsc_xa = xr.DataArray(
    wsc_det.transpose(2,1,0),
    dims=("time","lat","lon"),
    coords={"time":dot_time, "lat":lat, "lon":lon},
    name="wsc"
)

osc_xa = xr.DataArray(
    osc_det.transpose(2,1,0),
    dims=("time","lat","lon"),
    coords={"time":dot_time, "lat":lat, "lon":lon},
    name="osc"
)

# -----------------------------
# 5. CHOOSE VARIABLES FOR MCA
# -----------------------------
var_1 = sla_xa
var_2 = wsc_xa

var_1_name = var_1.name.upper()
var_2_name = var_2.name.upper()

print(f'Running MCA on: {var_1_name} vs {var_2_name}')

# -----------------------------
# 6. RUN MCA
# -----------------------------
model = xe.cross.MCA(n_modes=22, standardize=False, use_coslat=True)
model.fit(var_1, var_2, dim='time')

comps1, comps2 = model.components()
scores1, scores2 = model.scores()

# -----------------------------
# 7. SAVE OUTPUT
# -----------------------------
ds_out = xr.Dataset(
    {
        'scores_1': scores1,
        'scores_2': scores2,
        'comps_1': comps1,
        'comps_2': comps2,
    },
    coords={
        "lon": lon,
        "lat": lat,
        "time": dot_time,
    },
    attrs={
        'description': f'{var_1_name} and {var_2_name} MCA for {gyre_name}, using global detrended dataset.',
        'var_1_name': var_1_name,
        'var_2_name': var_2_name,
    },
)
print("Saving lat range:", lat.min(), lat.max())

outfile = f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}_{gyre_name}.nc'
ds_out.to_netcdf(scores_comps_dir + outfile)

print(f'Saved MCA results to {scores_comps_dir}{outfile}')
