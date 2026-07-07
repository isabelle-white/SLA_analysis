"""
C7c_contour_params_annual_plots.py

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


# user settings gyre_name, variable, start_time ...
gyre_name = 'weddell' #ross #weddell #kerguelen
variable = 'dot' #'dot' #'sla'
start_time = '2002-09-01'
end_time = '2024-12-01'


gyre_dir = processing_dir + gyre_name + '/'
gyre_fig_dir = fig_dir + gyre_name + '/timeseries/'
# os.makedirs(gyre_fig_dir, exist_ok=True)
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

print(ts_center_centroid_ds)

time = ts_params_ds['time']

dot_mean = ts_params_ds['dot_mean']
area_km2 = ts_params_ds['area_km2']
contour_level = ts_params_ds['level']
strength_basic = ts_strength_ds['strength_basic']
strength_norm = ts_strength_ds['strength_normalised']
centre_lat = ts_center_centroid_ds['centre_lat']
centroid_lat = ts_center_centroid_ds['centroid_lat']

g_mean = ts_params_ds.groupby('time.month').mean()
g_strength_mean = ts_strength_ds.groupby('time.month').mean()
g_centre_mean = ts_center_centroid_ds.groupby('time.month').mean()

g_std = ts_params_ds.groupby('time.month').std()
g_strength_std = ts_strength_ds.groupby('time.month').std()
g_centre_std = ts_center_centroid_ds.groupby('time.month').std()

g_count = ts_params_ds.groupby('time.month').count()
g_strength_count = ts_strength_ds.groupby('time.month').count()
g_centre_count = ts_center_centroid_ds.groupby('time.month').count()

#monthly means
dot_monthly_mean = g_mean['dot_mean']
area_km2_monthly_mean = g_mean['area_km2']
contour_level_month_mean = g_mean['level']
strength_basic_monthly_mean = g_strength_mean['strength_basic']
strength_normal_monthly_mean = g_strength_mean['strength_normalised']
centre_lat_monthly_mean = g_centre_mean['centre_lat']
centroid_lat_monthly_mean = g_centre_mean['centroid_lat']

#monthly stdvs
dot_monthly_std = g_std['dot_mean']
area_km2_monthly_std = g_std['area_km2']
contour_level_month_std = g_std['level']
strength_basic_monthly_std = g_strength_std['strength_basic']
strength_normal_monthly_std = g_strength_std['strength_normalised']
centre_lat_monthly_std = g_centre_std['centre_lat']
centroid_lat_monthly_std = g_centre_std['centroid_lat']

# monthyl standard error (stdv/sqrt(n))
dot_monthly_se          = dot_monthly_std          / np.sqrt(g_count['dot_mean'])
area_km2_monthly_se     = area_km2_monthly_std     / np.sqrt(g_count['area_km2'])
contour_level_monthly_se = contour_level_month_std / np.sqrt(g_count['level'])
strength_basic_monthly_se = strength_basic_monthly_std / np.sqrt(g_strength_count['strength_basic'])
strength_norm_monthly_se  = strength_normal_monthly_std / np.sqrt(g_strength_count['strength_normalised'])
centre_lat_monthly_se    = centre_lat_monthly_std   / np.sqrt(g_centre_count['centre_lat'])
centroid_lat_monthly_se  = centroid_lat_monthly_std / np.sqrt(g_centre_count['centroid_lat'])

var_to_plot = [dot_monthly_mean, area_km2_monthly_mean, contour_level_month_mean, strength_basic_monthly_mean, strength_normal_monthly_mean, centre_lat_monthly_mean, centroid_lat_monthly_mean]
se_to_plot = [dot_monthly_se, area_km2_monthly_se, contour_level_monthly_se, strength_basic_monthly_se, strength_norm_monthly_se, centre_lat_monthly_se, centroid_lat_monthly_se]
name_to_plot = ['dot_mean', 'area_km2', 'contour_level', 'strength_basic', 'strength_normalised', 'centre_lat', 'centroid_lat']
units_to_plot = ['m', 'km2', 'm', '-', '-', '-', '-']
colours_to_plot = ['steelblue', 'red', 'darkorange', 'purple', 'plum', 'seagreen', 'olive' ]


fig, axes = plt.subplots(len(var_to_plot), 1, figsize = (8, 15), sharex=False, sharey=False )
months = np.arange(1, 13)

for i, var in enumerate(var_to_plot):
    axes[i].plot(months, var, linewidth = 1.5, color = colours_to_plot[i] )
    axes[i].axhline(np.nanmean(var), color='gray', linestyle='--', linewidth=0.8)
    axes[i].fill_between(months,
                      np.array(var) - np.array(se_to_plot[i]),
                      np.array(var) + np.array(se_to_plot[i]),
                      color=colours_to_plot[i], alpha=0.1)
    axes[i].set_ylabel(f'{name_to_plot[i]} [{units_to_plot[i]}]')
    axes[i].grid(alpha=0.3)
    axes[i].set_title(name_to_plot[i])
plt.tight_layout()

save_fig_name = tag + '_params_annual_var.png'
plt.savefig(gyre_fig_dir + save_fig_name, dpi = 300, bbox_inches='tight')

plt.show()
