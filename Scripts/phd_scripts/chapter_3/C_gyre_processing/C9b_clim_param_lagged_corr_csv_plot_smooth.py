"""
C9b_clim_param_lagged_corr_csv_plot_smooth.py

-Loads data from  C1 (params file)
- calculates the correlation between all clims and params and saves to a csv
- Plots params against time (if to_plot == True)

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
import pandas as pd
import pprint


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
gyre_name = 'ross' #ross #weddell #kerguelen
variable = 'dot' #'dot' #'sla'
lag = 3 # number of months to shift climate index by


to_plot = False # False # True
clim_idx_to_plot = 'SAM'  # 'SAM', 'SOI', 'ASL', 'ZW3_pc1', 'ZW3_pc2'
param_to_plot = 'dot_mean'

start_time = '2002-09-01'
end_time = '2024-12-01'

# start_time = '2003-01-01'
# end_time = '2010-12-01'

# start_time = '2011-01-01'
# end_time = '2015-12-01'

# start_time = '2016-01-01'
# end_time = '2024-12-01'

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

clim_idx_list = ['SAM', 'SOI', 'ASL', 'ZW3_pc1', 'ZW3_pc2']
clim_idx_da_dict = {}

for clim_idx in clim_idx_list:
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

    clim_da = clim_da.shift(time = lag)

    clim_idx_da_dict[clim_idx] = clim_da

param_dict = {
    'dot_mean': ts_params_ds['dot_mean'],
    'area_km2': ts_params_ds['area_km2'],
    'contour_level': ts_params_ds['level'],
    'strength_basic': ts_strength_ds['strength_basic'],
    'strength_norm': ts_strength_ds['strength_normalised'],
    'centre_lat': ts_center_centroid_ds['centre_lat'],
    'centroid_lat': ts_center_centroid_ds['centroid_lat'],
}



# dot = ds['dot']
# MDT = dot.mean(dim='time')
# sla = dot - MDT
# ds['sla'] = sla

# get param time period (set in C1)
param_time = ts_params_ds['time'].values
common_time = np.intersect1d(clim_idx_da_dict[clim_idx_to_plot].time.values, param_time)
common_time = common_time[common_time >= np.datetime64(start_time)]
common_time = common_time[common_time <= np.datetime64(end_time)]

# Select full period for plotting
index_da = clim_da.sel(time=common_time)
ts_params_ds = ts_params_ds.sel(time=common_time)
ts_strength_ds = ts_strength_ds.sel(time=common_time)
ts_center_centroid_ds = ts_center_centroid_ds.sel(time=common_time)
param_time = ts_params_ds['time'].values
#normalise clim index
ref_mean = index_da.mean()
ref_std = index_da.std()
index_norm = (index_da - ref_mean)/ref_std
index_norm = index_norm.dropna('time')


results = {}
for clim_name , clim_da in clim_idx_da_dict.items():
    # common_time = np.intersect1d(clim_idx_da_dict[clim_idx_to_plot].time.values, param_time)
    common_time = np.intersect1d(clim_da.time.values, param_time)
    common_time = common_time[common_time >= np.datetime64(start_time)]
    common_time = common_time[common_time <= np.datetime64(end_time)]

    clim_sel = clim_da.sel(time=common_time)
    clim_sel_norm = (clim_sel - clim_sel.mean())/clim_sel.std()
    clim_sel_norm = clim_sel_norm.dropna('time')

    clim_results = {}

    for param_name, param_da in param_dict.items():
        param_sel = param_da.sel(time = common_time).dropna('time')

        valid = np.isfinite(param_sel.values) & np.isfinite(clim_sel_norm.values)
        if valid.sum() <3 :
            r, p = np.nan, np.nan
        else:
            r, p  = scipy.stats.pearsonr(param_sel, clim_sel_norm)

        clim_results[param_name] = {'r': r, 'p': p}

    results[clim_name] = clim_results

# Printign logic for results

# print(results.keys())
# print(results['SAM'].keys())
# print(results['SAM']['dot_mean'])
# print(len(results))
# print(len(results['SAM']))
#
# pprint.pprint(results) # pretty print

#conversion from results dict to savelable dict
rows = []
for clim_name, clim_results in results.items():
    for param_name, stats in clim_results.items():
        rows.append ({
            'gyre_name': gyre_name,
            'months_clim_lag': lag,
            'start_time': start_time,
            'end_time': end_time,
            'climate_index': clim_name,
            'gyre_param_name': param_name,
            'r': stats['r'],
            'p': stats['p'],
        })
results_df = pd.DataFrame(rows)

csv_path = gyre_dir + f'{tag}_clim_param_corr_{lag}_months_lag_smooth.csv'
results_df.to_csv(csv_path, index=False)
print('SAVED TO CSV in ', csv_path)


# toggled plotting code (use to_plot == True to plot figure with correlation and p value on time series)

if to_plot == True:
    var_to_plot = [
    ts_params_ds['dot_mean'].values,
    ts_params_ds['area_km2'].values,
    ts_params_ds['level'].values,
    ts_strength_ds['strength_basic'].values,
    ts_strength_ds['strength_normalised'].values,
    ts_center_centroid_ds['centre_lat'].values,
    ts_center_centroid_ds['centroid_lat'].values
    ]

    name_to_plot = ['dot_mean', 'area_km2', 'contour_level', 'strength_basic', 'strength_normalised', 'centre_lat', 'centroid_lat']
    units_to_plot = ['m', 'km2', 'm', '-', '-', '-', '-']
    colours_to_plot = ['steelblue', 'red', 'darkorange', 'purple', 'plum', 'seagreen', 'olive' ]

    pearson_r = []
    pearson_p = []
    for var in var_to_plot:
        r, p  = scipy.stats.pearsonr(var, index_norm)
        pearson_r.append(r)
        pearson_p.append(p)


    fig, axes = plt.subplots(len(var_to_plot), 1, figsize = (8, 15))

    for i, var in enumerate(var_to_plot):
        axes[i].plot(param_time, var, linewidth = 1.5, color = colours_to_plot[i] )
        # axes[i].plot(param_time, index_norm, linewidth = 1.5, color = 'black', label = clim_idx, linestyle = '--' )
        axes[i].axhline(np.nanmean(var), color='gray', linestyle='--', linewidth=0.8)
        axes[i].set_ylabel(f'{name_to_plot[i]} [{units_to_plot[i]}]')
        axes[i].grid(alpha=0.3)
        axes[i].text(0.01, 0.87, f'r = {pearson_r[i]:.2f}, p = {pearson_p[i]:.2f}', transform=axes[i].transAxes)
        axes[i].set_title(f'{name_to_plot[i]}, climate index {clim_idx}, lag = {lag}')
    plt.tight_layout()
    #
    save_fig_name = tag + f'_params_timeseries_{clim_idx}_corr_{lag}_months_lag.png'
    plt.savefig(gyre_fig_dir + save_fig_name, dpi = 300, bbox_inches='tight')

    plt.show()
else:
    print('not plotting at to_plot toggled off')