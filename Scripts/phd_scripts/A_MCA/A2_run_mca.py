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
import os
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
scores_comps_dir = mca_dir + 'scores_comps/'


sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func


# select time range for vairable calculation
start_time = '2002-07-01'
end_time = '2024-12-01'


# load dot dataset
dot_ds = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
# full ds time
dot_ds = dot_ds.sel(time=slice(start_time, end_time))
dot = dot_ds.dot.values
dot_time = dot_ds.time.values
lon = dot_ds.longitude.values
lat = dot_ds.latitude.values

# load wsc and osc ds
det_vars_ds = xr.open_dataset(vars_dir + f'wsc_osc_sla_det_{start_time}_{end_time}.nc')
wsc_det = det_vars_ds.wsc_detrended.values
osc_det = det_vars_ds.osc_detrended.values
sla_det = det_vars_ds.sla_detrended.values

# transform to x-arrays
sla_xa = xr.DataArray(
    sla_det.transpose(2, 1, 0),          # (time, lat, lon)
    dims=("time", "lat", "lon"),
    coords={"time": dot_time, "lat": lat, "lon": lon},
    name = 'sla'
)

wsc_xa = xr.DataArray(
    wsc_det.transpose(2, 1, 0),          # (time, lat, lon)
    dims=("time", "lat", "lon"),
    coords={"time": dot_time, "lat": lat, "lon": lon},
    name = 'wsc'
)

osc_xa = xr.DataArray(
    osc_det.transpose(2,1,0),
    dims=("time", "lat", "lon"),
    coords={"time": dot_time, "lat": lat, "lon": lon},
    name = 'osc'
)

# choose mca variables
print('choosing mca variables...')

import inspect

# var_1 = osc_xa
# var_2 = sla_xa

var_1 = sla_xa
var_2 = osc_xa

var_1_name = var_1.name.upper()
var_2_name = var_2.name.upper()

print(f'var_1: {var_1_name}')
print(f'var_2: {var_2_name}')


model = xe.cross.MCA(n_modes=22, standardize=False, use_coslat = True)
model.fit(var_1, var_2, dim='time')

# print(sla_xa.coords)
# print(dir(model))
# print(var_1.dims)
# print(var_2.dims)

print('calculating comps and scores...')
comps1, comps2 = model.components()  # Singular vectors (spatial patterns)
scores1, scores2 = model.scores()  # Expansion coefficients (temporal patterns)

# s1 = scores1.values; s2 = scores2.values
# c1 = comps1.values; c2 = comps2.values

# Full path to the 'results' directory (scoreS_comps_dir)
# Check if the directory already exists
if os.path.exists(scores_comps_dir):
    print("Directory already exists")
else:
    # Create the directory if it doesn't exist
    os.makedirs(scores_comps_dir, exist_ok=True)
    print("Directory created!")

ds_out = xr.Dataset(
    {
    'scores_1': scores1,
    'scores_2': scores2,
    'comps_1': comps1,
    'comps_2': comps2,
    },
    coords ={
        "lon": lon,
        "lat": lat,
        "time": dot_time,
    },
    attrs = {
    'description': f'{var_1_name} and {var_2_name} scores and comps after mca between {start_time} and {end_time}. Created in A2_run_mca.py',
    'var_1_name': var_1_name,
    'var_2_name': var_2_name,
})

print('saving scores and comps to nc file')
outfile = f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}.nc'
ds_out.to_netcdf(scores_comps_dir + outfile)
print(f'saved {outfile} to {scores_comps_dir}{outfile}!')