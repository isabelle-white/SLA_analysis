"""
B2_plot_mca_rectangular_gyre.py
Load sector saved scores and comps
Plot MCA on a rectangular grid for the gyre

last modified 27/05/2026
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
from matplotlib.gridspec import GridSpec


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
mca_dir = data_dir + 'mca_processing/sectors/'
clim_dir = data_dir + 'climate_indices/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/A_MCA/sectors/'


sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func


# select time range for vairable calculation
start_time = '2002-07-01'
end_time = '2024-12-01'

# Select gyre
gyre_name = 'weddell' #ross #weddell #kerguelen

# set variable names
var_1_name = 'SLA'
var_2_name = 'OSC'

if gyre_name == 'ross':
    lon_min  = 140        # western boundary of your sector
    lon_max  =  -80  # eastern boundary of your sector
    lat_min = -90
    lat_max = -60
    sector_dir = mca_dir + 'ross/'
    vars_dir = sector_dir + 'preprocessed_vars/'
    scores_comps_dir = sector_dir + 'scores_comps/'
    fig_dir = fig_dir + 'ross/rectangular/'
    crosses_dateline = True # set to False for normal sectors (<0 then >0)
elif gyre_name == 'weddell':
    lon_min = -80
    lon_max = 50
    sector_dir = mca_dir + 'weddell/'
    vars_dir = sector_dir + 'preprocessed_vars/'
    scores_comps_dir = sector_dir + 'scores_comps/'
    fig_dir = fig_dir + 'weddell/rectangular/'
    crosses_dateline = False # set to False for normal sectors (<0 then >0)
elif gyre_name == 'kerguelen':
    lon_min = 60
    lon_max = 150
    sector_dir = mca_dir + 'kerguelen/'
    vars_dir = sector_dir + 'preprocessed_vars/'
    scores_comps_dir = sector_dir + 'scores_comps/'
    fig_dir = fig_dir + 'kerguelen/rectangular/'
    crosses_dateline = False # set to False for normal sectors (<0 then >0)

# load dot dataset
dot_ds = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')

if crosses_dateline:
    # dot ds
    ds_A = dot_ds.sel(longitude=slice(lon_min, 180), latitude = slice(lat_min, lat_max))
    ds_B = dot_ds.sel(longitude=slice(-180, lon_max), latitude = slice(lat_min, lat_max))
    # Shift chunk B longitudes by +360 so the axis is monotonically increasing
    ds_B = ds_B.assign_coords(longitude=ds_B.longitude.values + 360)
    # Concatenate along longitude
    dot_ds = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')

    # extract dot coords
    dot = dot_ds.dot.values
    lon = dot_ds.longitude.values
    lat = dot_ds.latitude.values
    dot_time = dot_ds.time.values
else:
    dot_ds = dot_ds.sel(longitude=slice(lon_min, lon_max))
    # select/extract variables
    dot = dot_ds.dot.values
    lon = dot_ds.longitude.values
    lat = dot_ds.latitude.values
    dot_time = dot_ds.time.values

print('loading scores and comps....')
# load scores and comps
scores_comps_ds = xr.open_dataset(scores_comps_dir + f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}_{gyre_name}.nc')
print("Loaded lat range:", scores_comps_ds.lat.min().item(), scores_comps_ds.lat.max().item())
print('loaded path:', scores_comps_dir + f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}_{gyre_name}.nc')
scores1 = scores_comps_ds['scores_1']
scores2 = scores_comps_ds['scores_2']
comps1 = scores_comps_ds['comps_1']
comps2 = scores_comps_ds['comps_2']

# manual sign flip options --> to match reference papers
flip_both = []     # e.g. [1, 3]

# Flip both fields (var1 + var2)
for mode_num in flip_both:
    comps1.loc[dict(mode=mode_num)]  *= -1
    comps2.loc[dict(mode=mode_num)]  *= -1
    scores1.loc[dict(mode=mode_num)] *= -1
    scores2.loc[dict(mode=mode_num)] *= -1


print('calculating correlations....')
r_pears = []; p_pears = []
r_spear = []; p_spear = []
for m in range(1,6):
    x = scores1.sel(mode=m)
    y = scores2.sel(mode=m)
    r, p = scipy.stats.pearsonr(x, y)
    r_pears.append(np.round(r, 2)); p_pears.append(np.round(p, 2))
    r, p = scipy.stats.spearmanr(x, y)
    r_spear.append(np.round(r, 2)); p_spear.append(np.round(p, 2))

fig = plt.figure(figsize=(12,10))


print('plotting scores and comps....')
for i in range(0,5):
    j = 3*i+1

    mode_num = i+1
    # --- TIME SERIES ---
    ax = plt.subplot(5,3,j)

    # Normalise the scores for the time series plot
    s1_norm = scores1.sel(mode=mode_num) / scores1.sel(mode=mode_num).std()
    s2_norm = scores2.sel(mode=mode_num) / scores2.sel(mode=mode_num).std()

    # scores1.sel(mode=i+1).plot(label=f'{var_1_name}')
    # scores2.sel(mode=i+1).plot(label=f'{var_2_name}')

    s1_norm.plot(label=f'{var_1_name}')
    s2_norm.plot(label=f'{var_2_name}')

    ax.set_title(f'Mode {i+1}| r = {r_pears[i]:.2f}, p = {p_pears[i]}')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'PC{i+1}')
    ax.tick_params(axis = 'x', rotation=30)
    ax.legend()

    ax2 = plt.subplot(5, 3, j+1, projection=ccrs.PlateCarree())
    comps1.sel(mode=i+1).plot(ax=ax2, cmap='RdBu_r',
                               x='lon', y='lat',
                               transform=ccrs.PlateCarree(),
                               add_colorbar=True)
    ax2.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax2.set_extent([lon_min, lon_max,
                    comps1.lat.min().item()-3, comps1.lat.max().item()],
                   crs=ccrs.PlateCarree())
    ax2.set_aspect('auto')
    gl2 = ax2.gridlines(draw_labels=True, linewidth=0, color='grey', alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    ax2.set_title(f'{var_1_name} Mode {i+1}')

    ax3 = plt.subplot(5, 3, j+2, projection=ccrs.PlateCarree())
    comps2.sel(mode=i+1).plot(ax=ax3, cmap='RdBu_r',
                               x='lon', y='lat',
                               transform=ccrs.PlateCarree(),
                               add_colorbar=True)
    ax3.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax3.set_extent([lon_min, lon_max,
                    comps2.lat.min().item()-3, comps2.lat.max().item()],
                   crs=ccrs.PlateCarree())
    ax3.set_aspect('auto')
    gl3 = ax3.gridlines(draw_labels=True, linewidth=0, color='grey', alpha=0.5, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False
    ax3.set_title(f'{var_2_name} Mode {i+1}')

fig.tight_layout()
plt.show()
full_path = fig_dir + f'{var_1_name}_{var_2_name}_{start_time}_{end_time}_{gyre_name}_rectangular.png'
print('saving figure to :',full_path)
fig.savefig(full_path, dpi=300 )
print('figure saved !')