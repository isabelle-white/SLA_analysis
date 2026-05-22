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
fig_dir = workdir + 'Figures/A_MCA/'
vars_dir = mca_dir + 'preprocessed_vars/'
scores_comps_dir = mca_dir + 'scores_comps/'


sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func
import aux_stereoplot as st


# select time range for vairable calculation
start_time = '2002-07-01'
end_time = '2024-12-01'

# set variable names
var_1_name = 'SLA'
var_2_name = 'OSC'

# set climate index
clim_idx = 'ZW3_pc1'  # 'SAM', 'SOI', 'ASL', 'ZW3_pc1', 'ZW3_pc2'

if clim_idx == 'SAM':
    clim_ds = xr.open_dataset(clim_dir + 'SAM/sam_2000_2024' + '.nc')
    clim_da = clim_ds['SAM']
elif clim_idx == 'SOI':
    clim_ds = xr.open_dataset(clim_dir + 'SOI/soi_2000_2024.nc')
    clim_da = clim_ds["SOI"]
elif clim_idx == 'ASL':
    clim_ds = xr.open_dataset(clim_dir + 'ASL/ASL_2000_2025_data.nc')
    clim_da = clim_ds["RelCenPres"]
elif clim_idx == 'ZW3_pc1':
    clim_ds = xr.open_dataset(clim_dir + 'ZW3/zw3_2000_2024.nc')
    clim_da = clim_ds["pc1"]  # zw3index_magnitude # zw3index_phase #pc1 #pc2
elif clim_idx == 'ZW3_pc2':
    clim_ds = xr.open_dataset(clim_dir + 'ZW3/zw3_2000_2024.nc')
    clim_da = clim_ds["pc2"]

save_dir = f'{fig_dir}/climate_indices/{clim_idx}/'
os.makedirs(save_dir, exist_ok=True)

# set number of modes plotted
n_modes = 5


# load dot dataset
dot_ds = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
# full ds time
dot_ds = dot_ds.sel(time=slice(start_time, end_time))
dot = dot_ds.dot.values
dot_time = dot_ds.time.values
lon = dot_ds.longitude.values
lat = dot_ds.latitude.values

print('loading scores and comps....')
# load scores and comps
scores_comps_ds = xr.open_dataset(scores_comps_dir + f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}.nc')
scores1 = scores_comps_ds['scores_1']
scores2 = scores_comps_ds['scores_2']
comps1 = scores_comps_ds['comps_1']
comps2 = scores_comps_ds['comps_2']

r_pears = []; p_pears = []
r_spear = []; p_spear = []
for m in range(1,6):
    x = scores1.sel(mode=m)
    y = scores2.sel(mode=m)
    r, p = scipy.stats.pearsonr(x, y)
    r_pears.append(np.round(r, 2)); p_pears.append(np.round(p, 2))
    r, p = scipy.stats.spearmanr(x, y)
    r_spear.append(np.round(r, 2)); p_spear.append(np.round(p, 2))

common_time = np.intersect1d(clim_da.time.values, scores1.time.values)
common_time = common_time[common_time >= np.datetime64(start_time)]
common_time = common_time[common_time <= np.datetime64(end_time)]


# Select full period for plotting
index_da = clim_da.sel(time=common_time)

# Detrend over reference period first
index_ref = index_da.sel(time=slice(start_time, end_time))

t_ref   = np.arange(len(index_ref))
slope, intercept, _, _, _ = scipy.stats.linregress(t_ref, index_ref.values)

# Apply detrend fit to the FULL time series (using same slope/intercept)
t_full       = np.arange(len(index_da))
trend_full   = slope * t_full + intercept
index_detrended = index_da - xr.DataArray(trend_full, coords=[index_da.time], dims='time')

# Standardise using reference period mean and std
ref_mean = index_detrended.sel(time=slice(start_time, end_time)).mean()
ref_std  = index_detrended.sel(time=slice(start_time, end_time)).std()
index_norm = (index_detrended - ref_mean) / ref_std

# ── plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(n_modes, 1, figsize=(20 , 4 * n_modes), sharex=True)

for i, ax in enumerate(axes):
    m = i + 1

    s1 = scores1.sel(mode=m, time=common_time)
    s2 = scores2.sel(mode=m, time=common_time)
    s1_n = (s1 - s1.mean()) / s1.std()
    s2_n = (s2 - s2.mean()) / s2.std()

    r_12, _ = scipy.stats.pearsonr(s1_n.values, s2_n.values)
    r_i1, _ = scipy.stats.pearsonr(index_norm.values, s1_n.values)
    r_i2, _ = scipy.stats.pearsonr(index_norm.values, s2_n.values)

    ax.plot(s1_n.time, s1_n, '-', color = 'red',linewidth = '2.5', label=f'MCA{m} {var_1_name}')
    ax.plot(s2_n.time, s2_n, '-', color = 'blue', linewidth = '2.5', label=f'MCA{m} {var_2_name}')
    ax.plot(index_norm.time, index_norm, '--', color='black', label=f'{clim_idx}')

    ax.set_ylabel(f'Mode {m} (norm.)', fontsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15, loc = 'lower left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30, size = 15)

    corr_txt = (
        f'{var_1_name}–{var_2_name}  r={r_12:.2f}\n'
        f'{clim_idx}–{var_1_name}           r={r_i1:.2f}\n'
        f'{clim_idx}–{var_2_name}           r={r_i2:.2f}'
    )
    ax.text(0.98, 0.03, corr_txt, transform=ax.transAxes,
            fontsize=15, va='bottom', ha='right',
            bbox=dict(facecolor='white', edgecolor='grey', alpha=0.8))

axes[-1].set_xlabel('Year', fontsize=15)
fig.suptitle(f'{var_1_name} - {var_2_name} MCA modes vs {clim_idx}', fontsize=20, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.98])


save_name = f'timeseries_corr_{var_1_name}_{var_2_name}_{start_time}_{end_time}.png'
plt.savefig(save_dir + save_name, dpi=300)
plt.close(fig)