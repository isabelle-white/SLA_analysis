# A1_run_mca.py
"""
This script calculates the mca between two variables for the deseas + detrended or just detrended nc file from A0

Last modified: 14/04/2026
"""

import numpy as np
import xarray as xr
import os
import sys
import warnings
import xeofs as xe


# set paths
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
mca_dir = data_dir + 'mca_processing/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'

sys.path.append(auxscriptdir)
import mca_utils as mca

deseas = True
detrended = True

if deseas:
    suffix = 'deseas'
else:
    suffix = 'with_seasonal_cycle'

if detrended:
    suffix += '_det'
else:
    suffix += '_raw'

# load data from A0
filepath = mca_dir + f'mca_preprocessed_{suffix}.nc'
ds = xr.open_dataset(filepath)

print(f"Loaded: {filepath}")
print(f"  seasonal_removed = {ds.attrs.get('seasonal_removed', 'unknown')}")
print(f"  variables: {list(ds.data_vars)}")
print(f"  time: {ds.time.values[0]} to {ds.time.values[-1]} ({len(ds.time)} steps)")

# set sector
sector_name = 'full'
lon_min, lon_max = -180, 180
lat_min, lat_max = -90, -50
crosses_dateline = False

# Example for Ross:
# sector_name = 'Ross'
# lon_min, lon_max = 150, -80
# lat_min, lat_max = -90, -50
# crosses_dateline = True

#choose variables for mca
var_1_name = 'total_ws'# forcing variable
var_2_name = 'sla' # response variable

#prepare the data based on lon range
if not crosses_dateline:
    xa_1, xa_2 = mca.prepare_fields(
        ds, var_1_name, var_2_name,
        lon_slice=(lon_min, lon_max),
        lat_slice=(lat_min, lat_max)
    )
else:
    xa_1, xa_2 = mca.prepare_fields_dateline(
        ds, var_1_name, var_2_name,
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max
    )

#run mca
n_modes = 22
print(f"Running MCA (n_modes={n_modes})...")

model = xe.cross.MCA(n_modes=n_modes, standardize=True)
model.fit(xa_1, xa_2, dim='time')

comps1, comps2 = model.components()
scores1, scores2 = model.scores()

# Print variance
sq_cov = model.squared_covariance_fraction() * 100
cum = np.cumsum(sq_cov.values)
print("  Squared covariance fraction (%):")
for m in range(min(4, n_modes)):
    print(f"    Mode {m+1}: {sq_cov.isel(mode=m).values:.2f}%  "
          f"(cumulative: {cum[m]:.2f}%)")

# save the comps and scores
savedir = data_dir + 'mca_processing/'
os.makedirs(savedir, exist_ok=True)

scores_outfile = f"{savedir}scores_{var_1_name}_{var_2_name}_{sector_name}_{suffix}.nc"
comps_outfile  = f"{savedir}comps_{var_1_name}_{var_2_name}_{sector_name}_{suffix}.nc"

if os.path.exists(scores_outfile):
    print(f"Files already exist: {scores_outfile}, {comps_outfile}")
else:
    print(f"Writing scores and comps to {scores_outfile}, {comps_outfile}")
    scores_ds = xr.Dataset({
        'scores1': scores1,
        'scores2': scores2,
    })
    scores_ds.to_netcdf(scores_outfile)

    comps_ds = xr.Dataset({
        'comps1': comps1,
        'comps2': comps2,
    })

    comps_ds.to_netcdf(comps_outfile)
    print(f"File does not exist")


print(f"\nSaved MCA scores to {scores_outfile}")
print(f"Saved MCA components to {comps_outfile}")
print("Done.")