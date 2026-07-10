"""
0c_mca_sla_sha_eha.py

calculate mca for sla_sha and sla_eha

sha = steric height anom (due to density changes)
eha = eustatic height anom (due to volume changes)
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
scores_comps_dir = data_dir + '/mca_processing/F_steric_height/scores_comps/'
os.makedirs(fig_dir, exist_ok=True)

mca_dir = data_dir + 'mca_processing/full/'
vars_dir = mca_dir + 'preprocessed_vars/'

sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func

start_time = '2002-07-01'
end_time = '2024-12-01'

sla_sha_eha_ds_det = xr.open_dataset(data_dir+'steric_height_cocks/sla_sha_eha_det.nc')
det_vars_ds = xr.open_dataset(vars_dir + f'wsc_osc_sla_det_{start_time}_{end_time}.nc')

time_ref = sla_sha_eha_ds_det['time'].values
lon = sla_sha_eha_ds_det.longitude
lat = sla_sha_eha_ds_det.latitude
sla = sla_sha_eha_ds_det["sla_det"].values #time, lat, lon
sha = sla_sha_eha_ds_det["sha_det"].values # time, lat, lon
eha = sla_sha_eha_ds_det["eha_det"].values #time, lat, lon


wsc_det = det_vars_ds['wsc_detrended']
osc_det = det_vars_ds['osc_detrended']
# Reindex to match SLA/SHA/EHA time axis
wsc_det = wsc_det.reindex(time=time_ref)
osc_det = osc_det.reindex(time=time_ref)

wsc_det = wsc_det.values
osc_det = osc_det.values

time = time_ref

# transform to x-arrays
sla_xa = xr.DataArray(
    sla,# (time, lat, lon)
    dims=("time","latitude","longitude"),
    coords={"time": time, "latitude": lat, "longitude": lon},
    name = 'sla'
)

sha_xa = xr.DataArray(
    sha,        # (time, lat, lon)
    dims=("time","latitude","longitude"),
    coords={"time": time, "latitude": lat, "longitude": lon},
    name = 'sha'
)

eha_xa = xr.DataArray(
    eha,
    dims=("time","latitude","longitude"),
    coords={"time": time, "latitude": lat, "longitude": lon},
    name = 'eha'
)

wsc_xa = xr.DataArray(
    wsc_det.transpose(2, 1, 0),          # (time, lat, lon)
    dims=("time","latitude","longitude"),
    coords={"time": time, "latitude": lat, "longitude": lon},
    name = 'wsc'
)

osc_xa = xr.DataArray(
    osc_det.transpose(2,1,0),
    dims=("time","latitude","longitude"),
    coords={"time": time, "latitude": lat, "longitude": lon},
    name = 'osc'
)

# choose mca variables
print('choosing mca variables...')


print(osc_xa)

# var_1 = sla_xa
# var_2 = sha_xa # sha_xa #eha_xa

var_1 = wsc_xa
var_2 = eha_xa # sha_xa #eha_xa

var_1_name = var_1.name.upper()
var_2_name = var_2.name.upper()

print(f'var_1: {var_1_name}')
print(f'var_2: {var_2_name}')


model = xe.cross.MCA(n_modes=22, standardize=False, use_coslat = True)
model.fit(var_1, var_2, dim='time')
sq_cov_frac = model.squared_covariance_fraction()
sq_cov_percent = sq_cov_frac * 100
print(sq_cov_percent)

print('plotting squared covariance fraction...')
fig_scf, ax_scf = plt.subplots(figsize=(8, 5))
modes = np.arange(1, len(sq_cov_percent) + 1)
ax_scf.plot(modes, sq_cov_percent, marker='o')
ax_scf.set_xlabel('Mode')
ax_scf.set_ylabel('Squared covariance fraction (%)')
ax_scf.set_title(f'Squared covariance fraction: {var_1_name} vs {var_2_name}')
ax_scf.set_xticks(modes)
ax_scf.grid(True, alpha=0.3)
fig_scf.tight_layout()

scf_dir = fig_dir + 'A_MCA/full/'
os.makedirs(scf_dir, exist_ok=True)
scf_path = scf_dir + f'squared_covariance_{var_1_name}_{var_2_name}_full_ts.png'
fig_scf.savefig(scf_path, dpi=300)
print(f'saved squared covariance plot to {scf_path}')

print('calculating comps and scores...')
comps1, comps2 = model.components()  # Singular vectors (spatial patterns)
scores1, scores2 = model.scores()  # Expansion coefficients (temporal patterns)

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
    'sq_cov_frac': sq_cov_percent,
    },
    coords ={
        "longitude": lon,
        "latitude": lat,
        "time": time,
        "mode": np.arange(1, len(sq_cov_percent)+1),
    },
    attrs = {
    'description': f'{var_1_name} and {var_2_name} scores and comps after mca between 2002-07 and 2024-12. Created in 0c_mca_sla_sha_eha.py',
    'var_1_name': var_1_name,
    'var_2_name': var_2_name,
})

print('saving scores and comps to nc file')
outfile = f'{var_1_name}_{var_2_name}_scores_comps_full_ts.nc'
ds_out.to_netcdf(scores_comps_dir + outfile)
print(f'saved {outfile} to {scores_comps_dir}{outfile}!')
