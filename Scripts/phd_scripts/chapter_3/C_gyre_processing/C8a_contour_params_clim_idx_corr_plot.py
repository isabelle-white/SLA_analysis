"""
C8a_contour_params_clim_idx_corr_plot.py

-Loads data from  C1 (params file)
- Plots params against time

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
import scipy
import os


# Suppress only deprecation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
clim_dir = data_dir + 'climate_indices/'
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


# user settings gyre_name, variable, start_time ...
gyre_name = 'weddell' #ross #weddell #kerguelen
variable = 'dot' #'dot' #'sla'
start_time = '2002-09-01'
end_time = '2024-12-01'
# set climate index
clim_idx = 'ZW3_pc2'  # 'SAM', 'SOI', 'ASL', 'ZW3_pc1', 'ZW3_pc2'


gyre_dir = processing_dir + gyre_name + '/'
gyre_fig_dir = fig_dir + gyre_name + '/timeseries/climate_indices/'
os.makedirs(gyre_fig_dir, exist_ok=True)
tag = f'{gyre_name}_{start_time}_{end_time}'

# load dot ds
file_path = data_dir + 'dot_all_30bmedian_egm2008_sig3.nc'
ds = xr.open_dataset(file_path)
ds = ds.sel(time=slice(start_time, end_time))


# load params and verts
ts_verts = np.load(gyre_dir + f'{tag}_verts.npz', allow_pickle=True)
ts_params_ds = xr.open_dataset(gyre_dir + f'{tag}_params.nc')
mean_verts = np.load(gyre_dir + f'{tag}_verts_mdt.npz', allow_pickle=True)
mean_param_ds = xr.open_dataset(gyre_dir + f'{tag}_params_mdt.nc')
# load centers and centroids
ts_center_centroid_ds = xr.open_dataset(gyre_dir + tag +  '_center_centroid.nc')
mean_center_centroid_ds = xr.open_dataset(gyre_dir + tag +  '_center_centroid_mdt.nc')
#load strengths
ts_strength_ds = xr.open_dataset(gyre_dir + f'{tag}_strength.nc')
mean_strength_ds = xr.open_dataset(gyre_dir + f'{tag}_strength_mdt.nc')

# load climate index
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

# dot = ds['dot']
# MDT = dot.mean(dim='time')
# sla = dot - MDT
# ds['sla'] = sla

# get param time period (set in C1)
param_time = ts_params_ds['time'].values
common_time = np.intersect1d(clim_da.time.values, param_time)
common_time = common_time[common_time >= np.datetime64(start_time)]
common_time = common_time[common_time <= np.datetime64(end_time)]

# Select full period for plotting
index_da = clim_da.sel(time=common_time)
ts_params_ds = ts_params_ds.sel(time=common_time)

#normalise clim index
ref_mean = index_da.mean()
ref_std = index_da.std()
index_norm = (index_da - ref_mean)/ref_std

dot_mean = ts_params_ds['dot_mean'].values
area_km2 = ts_params_ds['area_km2'].values
contour_level = ts_params_ds['level'].values
strength_basic = ts_strength_ds['strength_basic'].values
strength_norm = ts_strength_ds['strength_normalised'].values
centre_lat = ts_center_centroid_ds['centre_lat'].values
centroid_lat = ts_center_centroid_ds['centroid_lat'].values

var_to_plot = [dot_mean, area_km2, contour_level, strength_basic, strength_norm, centre_lat, centroid_lat]
name_to_plot = ['dot_mean', 'area_km2', 'contour_level', 'strength_basic', 'strength_normalised', 'centre_lat', 'centroid_lat']
units_to_plot = ['m', 'km2', 'm', '-', '-', '-', '-']
colours_to_plot = ['steelblue', 'red', 'darkorange', 'purple', 'plum', 'seagreen', 'olive' ]

pearson_r = []
pearson_p = []
for var in var_to_plot:
    r, p  = scipy.stats.pearsonr(var, index_norm)
    pearson_r.append(r)
    pearson_p.append(p)


fig, axes = plt.subplots(len(var_to_plot), 1, figsize = (8, 15), sharex=False, sharey=False )
for i, var in enumerate(var_to_plot):
    axes[i].plot(param_time, var, linewidth = 1.5, color = colours_to_plot[i] )
    # axes[i].plot(param_time, index_norm, linewidth = 1.5, color = 'black', label = clim_idx, linestyle = '--' )
    axes[i].axhline(np.nanmean(var), color='gray', linestyle='--', linewidth=0.8)
    axes[i].set_ylabel(f'{name_to_plot[i]} [{units_to_plot[i]}]')
    axes[i].grid(alpha=0.3)
    axes[i].text(0.01, 0.87, f'r = {pearson_r[i]:.2f}, p = {pearson_p[i]:.2f}', transform=axes[i].transAxes)
    axes[i].set_title(f'{name_to_plot[i]}, climate index {clim_idx}')
plt.tight_layout()
#
save_fig_name = tag + f'_params_timeseries_{clim_idx}_corr.png'
plt.savefig(gyre_fig_dir + save_fig_name, dpi = 300, bbox_inches='tight')

plt.show()