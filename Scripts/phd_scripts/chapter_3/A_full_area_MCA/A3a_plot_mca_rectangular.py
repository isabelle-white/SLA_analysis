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
mca_dir = data_dir + 'mca_processing/full/'
clim_dir = data_dir + 'climate_indices/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/A_MCA/full/rectangular/'
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

# set variable names
var_1_name = 'SLA'
var_2_name = 'OSC'


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

    # ax2 = plt.subplot(5,3,j+1)
    # comps1.sel(mode = i+1).plot(ax = ax2, cmap='RdBu_r')
    # ax2.set_title(f'{var_1_name} Mode {i+1}')
    #
    # ax3 = plt.subplot(5,3,j+2)
    # comps2.sel(mode = i+1).plot(ax = ax3, cmap='RdBu_r')
    # ax3.set_title(f'{var_2_name} Mode {i+1}')

    # get extent from the data itself
    lon_min_ext = comps1.lon.min().item()
    lon_max_ext = comps1.lon.max().item()
    lat_min_ext = comps1.lat.min().item() -3
    lat_max_ext = comps1.lat.max().item()

    ax2 = plt.subplot(5, 3, j + 1, projection=ccrs.PlateCarree())
    comps1.sel(mode=i + 1).plot(ax=ax2, cmap='RdBu_r',
                                x='lon', y='lat',
                                transform=ccrs.PlateCarree(),
                                add_colorbar=True)
    ax2.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax2.set_extent([lon_min_ext, lon_max_ext, lat_min_ext, lat_max_ext],
                   crs=ccrs.PlateCarree())
    ax2.set_aspect('auto')
    gl2 = ax2.gridlines(draw_labels=True, linewidth=0, color='grey', alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    ax2.set_title(f'{var_1_name} Mode {i + 1}')

    ax3 = plt.subplot(5, 3, j + 2, projection=ccrs.PlateCarree())
    comps2.sel(mode=i + 1).plot(ax=ax3, cmap='RdBu_r',
                                x='lon', y='lat',
                                transform=ccrs.PlateCarree(),
                                add_colorbar=True)
    ax3.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax3.set_extent([lon_min_ext, lon_max_ext, lat_min_ext, lat_max_ext],
                   crs=ccrs.PlateCarree())
    ax3.set_aspect('auto')
    gl3 = ax3.gridlines(draw_labels=True, linewidth=0, color='grey', alpha=0.5, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False
    ax3.set_title(f'{var_2_name} Mode {i + 1}')

fig.tight_layout()
plt.show()
full_path = fig_dir + f'{var_1_name}_{var_2_name}_{start_time}_{end_time}_rectangular.png'
print('saving figure to :',full_path)
fig.savefig(full_path, dpi=300 )
print('figure saved !')