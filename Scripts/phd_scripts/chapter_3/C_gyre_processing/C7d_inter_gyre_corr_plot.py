"""
C7c_gyre_params_ts_plots.py

- Loads data from C1 (params file) for BOTH Weddell and Ross gyres
- Plots params against time, overlaying both gyres on each subplot
- Annotates each subplot legend with the Pearson correlation between
  the two gyres for that parameter

"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
import sys
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from pyproj import Geod
from shapely.geometry import Polygon
import warnings
import os


# Suppress only deprecation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
processing_dir = data_dir + 'C_gyre_processing/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
save_dir = data_dir + 'C_gyre_processing/'
fig_dir = workdir + 'Figures/C_gyre_processing/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st
from geometry_izzyv1 import grad_sphere
import aux_func as ft
clim_dir = data_dir + 'climate_indices/'

# user settings
gyre_names = ['weddell', 'ross']
variable = 'dot'  # 'dot' #'sla'

start_time = '2002-07-01'
end_time = '2024-12-01'

# --- shared output dir (not gyre-specific anymore) ---
combo_tag = f'{gyre_names[0]}_{gyre_names[1]}_{start_time}_{end_time}'
combo_fig_dir = fig_dir + 'comparison/timeseries/'
os.makedirs(combo_fig_dir, exist_ok=True)

# names/units/colours for params (one colour pair per gyre, defined below)
name_to_plot = ['dot_mean', 'area_km2', 'contour_level', 'strength_basic',
                 'strength_normalised', 'centre_lat', 'centroid_lat']
units_to_plot = ['m', 'km2', 'm', '-', '-', '-', '-']

# colours: one per gyre, used consistently across all subplots
gyre_colours = {'weddell': 'steelblue', 'ross': 'darkorange'}

# --- load data for each gyre into a dict ---
gyre_data = {}

for gyre_name in gyre_names:
    gyre_dir = processing_dir + gyre_name + '/'
    tag = f'{gyre_name}_{start_time}_{end_time}'

    ts_params_ds = xr.open_dataset(gyre_dir + f'{tag}_params.nc')
    ts_center_centroid_ds = xr.open_dataset(gyre_dir + tag + '_center_centroid.nc')
    ts_strength_ds = xr.open_dataset(gyre_dir + f'{tag}_strength.nc')

    time = ts_params_ds['time'].values

    gyre_data[gyre_name] = {
        'time': time,
        'dot_mean': ts_params_ds['dot_mean'].values,
        'area_km2': ts_params_ds['area_km2'].values,
        'contour_level': ts_params_ds['level'].values,
        'strength_basic': ts_strength_ds['strength_basic'].values,
        'strength_normalised': ts_strength_ds['strength_normalised'].values,
        'centre_lat': ts_center_centroid_ds['centre_lat'].values,
        'centroid_lat': ts_center_centroid_ds['centroid_lat'].values,
    }

print(f"Loaded data for: {list(gyre_data.keys())}")


def aligned_pearsonr(time_a, var_a, time_b, var_b):
    """
    Align two time series on common timestamps, drop NaNs pairwise,
    and return Pearson r and p. Returns (np.nan, np.nan) if too few
    overlapping points.
    """
    s_a = xr.DataArray(var_a, coords={'time': time_a}, dims='time')
    s_b = xr.DataArray(var_b, coords={'time': time_b}, dims='time')

    s_a, s_b = xr.align(s_a, s_b, join='inner')

    mask = (~np.isnan(s_a.values)) & (~np.isnan(s_b.values))
    if mask.sum() < 3:
        return np.nan, np.nan

    r, p = pearsonr(s_a.values[mask], s_b.values[mask])
    return r, p


# --- plot ---
fig, axes = plt.subplots(len(name_to_plot), 1, figsize=(25, 25),
                          sharex=False, sharey=False)

for i, param in enumerate(name_to_plot):
    ax = axes[i]

    # compute correlation between gyres for this param (assumes exactly 2 gyres)
    g1, g2 = gyre_names[0], gyre_names[1]
    r, p = aligned_pearsonr(
        gyre_data[g1]['time'], gyre_data[g1][param],
        gyre_data[g2]['time'], gyre_data[g2][param]
    )

    for gyre_name in gyre_names:
        ax.plot(
            gyre_data[gyre_name]['time'],
            gyre_data[gyre_name][param],
            linewidth=3,
            color=gyre_colours[gyre_name],
            label=f'{gyre_name.capitalize()}'
        )
        ax.axhline(np.nanmean(gyre_data[gyre_name][param]),
                   color=gyre_colours[gyre_name], linestyle='--',
                   linewidth=0.8, alpha=0.6)

    ax.set_title(f'{param}  |  r = {r:.2f}, p = {p:.3f}', fontsize=22)
    ax.set_ylabel(f'[{units_to_plot[i]}]', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14, loc='upper right')

plt.tight_layout()

save_fig_name = combo_tag + '_params_timeseries_comparison.png'
plt.savefig(combo_fig_dir + save_fig_name, dpi=300, bbox_inches='tight')

plt.show()