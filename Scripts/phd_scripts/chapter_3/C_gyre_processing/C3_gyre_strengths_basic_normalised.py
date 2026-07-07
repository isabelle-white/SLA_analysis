""""

C3_gyre_strengths_basic_normalised.py

- Loads dot param data from C1 and centre data from C2
- Calculates the strength of the gyre using the basic and normalised strength
- Saves: saves both strengths to the params file from C1 (separate files for full ts and mean gyre location - e.g. _mdt file)

"""""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from shapely.geometry import MultiPolygon
import geopandas as gpd
from shapely.geometry import box
import matplotlib.cm as cm
import pandas as pd
from shapely.affinity import translate
from matplotlib.path import Path
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import numpy.ma as ma
import matplotlib.pyplot as plt
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
sys.path.append(auxscriptdir)
import aux_stereoplot as st
from geometry_izzyv1 import grad_sphere
import aux_func as ft
clim_dir = data_dir + 'climate_indices/'

# user settings gyre_name, variable, start_time ...
gyre_name = 'ross' #ross #weddell #kerguelen
variable = 'dot' #'dot' #'sla'
start_time = '2002-09-01'
end_time = '2024-12-01'

gyre_dir = processing_dir + gyre_name + '/'
tag = f'{gyre_name}_{start_time}_{end_time}'

# load dot ds
file_path = data_dir + 'dot_all_30bmedian_egm2008_sig3.nc'
ds = xr.open_dataset(file_path)
ds = ds.sel(time=slice(start_time, end_time))


# load params and verts
ts_verts = np.load(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_verts.npz', allow_pickle=True)
ts_params_ds = xr.open_dataset(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_params.nc')
mean_verts = np.load(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_verts_mdt.npz', allow_pickle=True)
mean_param_ds = xr.open_dataset(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_params_mdt.nc')
# load centers and centroids
ts_center_centroid_ds = xr.open_dataset(gyre_dir + tag +  '_center_centroid.nc')
mean_center_centroid_ds = xr.open_dataset(gyre_dir + tag +  '_center_centroid_mdt.nc')

dot = ds['dot']
MDT = dot.mean(dim='time')
sla = dot - MDT
ds['sla'] = sla

# gyre specific settings (different masking lat limits and lat/lon areas)
if gyre_name == 'ross':
    lon_min  = 150        # western boundary of your sector
    lon_max  =  -120  # eastern boundary of your sector
    lat_min = -90
    lat_max = -55
    lat_mask_max = -70
    crosses_dateline = True # set to False for normal sectors (<0 then >0)
elif gyre_name == 'weddell':
    lon_min = -80
    lon_max = 50
    lat_min = -90
    lat_max = -50
    lat_mask_max = -75
    crosses_dateline = False # set to False for normal sectors (<0 then >0)

lat_sigma = 1
lon_sigma = 3

to_plot = []
for i, t in enumerate(ts_params_ds.time.values):
    tidx = int(ts_params_ds['time_idx'].values[i])
    to_plot.append({
        'time'    : t,
        'time_idx': tidx,
        'verts_max'   : ts_verts[f'{tidx}_best'],
        'verts_min' : ts_verts[f'{tidx}_min'],
    })


verts_max_mdt = mean_verts['mdt_contour_max_verts']
verts_min_mdt = mean_verts['mdt_contour_min_verts']

to_plot_mdt = []
to_plot_mdt.append({
    'verts_max_mdt' : verts_max_mdt,
    'verts_min_mdt' : verts_min_mdt,
})


gyre_strength =[]
R_earth = 6371
for result in to_plot:
    t = result['time']
    time_idx_i = result['time_idx']
    centre_lat = ts_center_centroid_ds.centre_lat.sel(time = t).item()
    centre_lon = (ts_center_centroid_ds.centre_lon.sel(time = t)).item()
    centre_var = (ts_center_centroid_ds.centre_var.sel(time = t)).item()
    verts_min = result['verts_min']
    verts_max = result['verts_max']

    if not crosses_dateline:
        ds_i = ds.sel(longitude=slice(lon_min, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)

        land_2d_i  = ds_i['land_mask'].values.T

        total_mask = (land_2d_i == 0)
        var_filled = np.where(total_mask, var_2d_i, 0.0)
        weights = np.where(total_mask, 1.0, 0.0)
        smoothed_vals  = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
        smoothed_weights = gaussian_filter(weights,    sigma=(lat_sigma, lon_sigma))
        smoothed  = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)
        var_2d_smoothed  = np.where(total_mask, smoothed, var_2d_i)
        var_2d_i  = np.where(land_2d_i == 1, np.nan, var_2d_smoothed)

        lat_mask   = lat_grid_i < lat_mask_max
        ocean_mask = ~np.isnan(var_2d_i)
        var_2d_i   = np.where(lat_mask & ocean_mask, smoothed, var_2d_i)
    else:
        ds_A = ds.sel(longitude=slice(lon_min, 180),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        ds_B = ds.sel(longitude=slice(-180, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        ds_B = ds_B.assign_coords(longitude=ds_B.longitude.values + 360)
        ds_i = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)


        land_2d_i  = ds_i['land_mask'].values.T

        total_mask = (land_2d_i == 0)
        var_filled = np.where(total_mask, var_2d_i, 0.0)
        weights  = np.where(total_mask, 1.0, 0.0)
        smoothed_vals  = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
        smoothed_weights = gaussian_filter(weights,    sigma=(lat_sigma, lon_sigma))
        smoothed   = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)
        var_2d_smoothed  = np.where(total_mask, smoothed, var_2d_i)
        var_2d_i  = np.where(land_2d_i == 1, np.nan, var_2d_smoothed)
        lat_mask  = lat_grid_i < lat_mask_max
        ocean_mask = ~np.isnan(var_2d_i)
        var_2d_i = np.where(lat_mask & ocean_mask, smoothed, var_2d_i)

    #CALCULATINGN BASIC GYRE STRENGTH
    # --- Boundary variable (dot or sla): mean var at contour vertices ---

    # print('calculating basic gyre strength')

    verts_i      = result['verts_max']
    boundary_var_basic = ft.sample_field_at_vertices(verts_i, var_2d_i, lon_grid_i, lat_grid_i)

    strength_basic = float(boundary_var_basic - centre_var)


    # print('calculating radius normalised gyre strength')

    # NORMALISED STRENGTH CALCULATION - full TS

    ## calc gyre strength (norm) using: S = (max dot contour height -center)/ Mean radius

    dot_max = ft.sample_field_at_vertices(verts_max, var_2d_i, lon_grid_i, lat_grid_i)
    dot_min = ft.sample_field_at_vertices(verts_min, var_2d_i, lon_grid_i, lat_grid_i)
    # --- Boundary variable (dot or sla): mean var at contour vertices ---
    boundary_var_norm = dot_max
    # use cKDTree to find the nearest neighbour between the min and max contours (this is needed to calculate R)
    ckdtree = cKDTree(verts_max)
    dists_deg, nearest_neighbour_index = ckdtree.query(verts_min)
    lon1 = np.deg2rad(verts_min[:,0])
    lat1 = np.deg2rad(verts_min[:,1])
    lon2 = np.deg2rad(verts_max[nearest_neighbour_index,0])
    lat2 = np.deg2rad(verts_max[nearest_neighbour_index,1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # haversine formula
    a = np.sin(dlat/2)**2
    b= np.cos(lat1)*np.cos(lat2)
    c = np.sin(dlon/2)**2

    dists_km = 2*R_earth*np.arcsin(np.sqrt(a+b*c))
    mean_R_km = np.mean(dists_km)

    if mean_R_km > 0.05:
        strength_normalised = float((dot_max - dot_min)/mean_R_km)
    else:
        strength_normalised = np.nan

    gyre_strength.append({
        'time'        : result['time'],
        'time_idx'    : time_idx_i,
        'boundary_var_basic': float(boundary_var_basic),
        'centre_var_basic'  : float(centre_var),
        'strength_basic'    : strength_basic,
        'boundary_var_norm' : float(boundary_var_norm),
        'centre_var_norm'   : float(centre_var),
        'strength_normalised': strength_normalised,
        'mean_R_km' : float(mean_R_km),
    })

print(f"Gyre strength (basic and normalised by radius) computed for {len(gyre_strength)} timesteps.")

#CALCULATING BOTH STRENGTHS FOR THE MEAN GYRE POSITION
# gyre_basic_strength_mdt = []
# gyre_normalised_strength_mdt = []
gyre_strength_mdt = []
# boundary_var_mdt = ft.sample_field_at_vertices(verts_i, var_2d_i, lon_grid_i, lat_grid_i)
for result_mdt in to_plot_mdt:
    centre_lat_mdt = mean_center_centroid_ds.centre_lat.item()
    centre_lon_mdt = mean_center_centroid_ds.centre_lon.item()
    centre_var_mdt = mean_center_centroid_ds.centre_var.item()
    verts_min_mdt = result_mdt['verts_min_mdt']
    verts_max_mdt = result_mdt['verts_max_mdt']

    if not crosses_dateline:
        ds_i = ds.sel(longitude=slice(lon_min, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[0])
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)

        land_2d_i  = ds_i['land_mask'].values.T

        total_mask = (land_2d_i == 0)
        var_filled = np.where(total_mask, var_2d_i, 0.0)
        weights = np.where(total_mask, 1.0, 0.0)
        smoothed_vals  = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
        smoothed_weights = gaussian_filter(weights,    sigma=(lat_sigma, lon_sigma))
        smoothed  = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)
        var_2d_smoothed  = np.where(total_mask, smoothed, var_2d_i)
        var_2d_i  = np.where(land_2d_i == 1, np.nan, var_2d_smoothed)

        lat_mask   = lat_grid_i < lat_mask_max
        ocean_mask = ~np.isnan(var_2d_i)
        var_2d_i   = np.where(lat_mask & ocean_mask, smoothed, var_2d_i)
    else:
        ds_A = ds.sel(longitude=slice(lon_min, 180),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[0])
        ds_B = ds.sel(longitude=slice(-180, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[0])
        ds_B = ds_B.assign_coords(longitude=ds_B.longitude.values + 360)
        ds_i = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)


        land_2d_i  = ds_i['land_mask'].values.T

        total_mask = (land_2d_i == 0)
        var_filled = np.where(total_mask, var_2d_i, 0.0)
        weights  = np.where(total_mask, 1.0, 0.0)
        smoothed_vals  = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
        smoothed_weights = gaussian_filter(weights,    sigma=(lat_sigma, lon_sigma))
        smoothed   = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)
        var_2d_smoothed  = np.where(total_mask, smoothed, var_2d_i)
        var_2d_i  = np.where(land_2d_i == 1, np.nan, var_2d_smoothed)
        lat_mask  = lat_grid_i < lat_mask_max
        ocean_mask = ~np.isnan(var_2d_i)
        var_2d_i = np.where(lat_mask & ocean_mask, smoothed, var_2d_i)

    # CALCULATING BASIC STRENGTH
    print('calculating basic stregnth for mean gyre location...')
    verts_i      = result_mdt['verts_max_mdt']
    boundary_var_mdt = ft.sample_field_at_vertices(verts_i, var_2d_i, lon_grid_i, lat_grid_i)

    strength_basic_mdt = float(boundary_var_mdt - centre_var_mdt)

    # gyre_strength_mdt.append({
    #     'boundary_var_mdt': float(boundary_var_mdt),
    #     'centre_var_mdt'  : float(centre_var_mdt),
    #     'strength_basic_mdt'    : strength_basic_mdt,
    # })

    # CALCULATING NORMALISED STRENGTH
    ## calc gyre strength (norm) using: S = (max dot contour height -center)/ Mean radius
    print('calculating normalised stregnth for mean gyre location...')
    dot_max = ft.sample_field_at_vertices(verts_max_mdt, var_2d_i, lon_grid_i, lat_grid_i)
    dot_min = ft.sample_field_at_vertices(verts_min_mdt, var_2d_i, lon_grid_i, lat_grid_i)
    # use cKDTree to find the nearest neighbour between the min and max contours (this is needed to calculate R)
    ckdtree = cKDTree(verts_max_mdt)
    dists_deg, nearest_neighbour_index = ckdtree.query(verts_min_mdt)
    lon1 = np.deg2rad(verts_min_mdt[:,0])
    lat1 = np.deg2rad(verts_min_mdt[:,1])
    lon2 = np.deg2rad(verts_max_mdt[nearest_neighbour_index,0])
    lat2 = np.deg2rad(verts_max_mdt[nearest_neighbour_index,1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # haversine formula
    a = np.sin(dlat/2)**2
    b= np.cos(lat1)*np.cos(lat2)
    c = np.sin(dlon/2)**2

    dists_km = 2*R_earth*np.arcsin(np.sqrt(a+b*c))
    mean_R_km_mdt = np.mean(dists_km)

    if mean_R_km_mdt > 0.05:
        strength_normalised_mdt = float((dot_max - dot_min)/mean_R_km_mdt)
    else:
        strength_normalised_mdt = np.nan

    gyre_strength_mdt.append({
        'boundary_var_mdt': float(boundary_var_mdt),
        'centre_var_mdt'  : float(centre_var_mdt),
        'strength_basic_mdt'    : strength_basic_mdt,
        'strength_normalised_mdt' : strength_normalised_mdt,
        'mean_R_km_mdt' : float(mean_R_km_mdt),
    })

print(f"Gyre 'basic' strength computed for mean gyre location.")
print(f"Gyre strength using mean R computed for mean gyre location: {mean_R_km_mdt}.")


print('saving datasets to files...')
ts_file_name = f'{tag}_strength.nc'
mean_file_name = f'{tag}_strength_mdt.nc'

df_file = pd.DataFrame(gyre_strength)
df_file_mdt = pd.DataFrame(gyre_strength_mdt)

ds_file = xr.Dataset.from_dataframe(df_file.set_index('time'))
ds_file_mdt = xr.Dataset.from_dataframe(df_file_mdt)

ds_file.to_netcdf(gyre_dir + ts_file_name)
ds_file_mdt.to_netcdf(gyre_dir + mean_file_name)
print(f'saved datasets to :\n {gyre_dir + ts_file_name} and \n {gyre_dir+mean_file_name}')