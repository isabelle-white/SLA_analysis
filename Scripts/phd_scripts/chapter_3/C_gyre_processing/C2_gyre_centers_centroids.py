""""

C2_gyre_centers.py

- Loads dot data from C1
- Calculates the center (min DOT) and centroid (min DOT distance vetweeb largest and smallest closed contours in the gyre)
- Calculate the center and centroids for the mean DOT
- Saves: adds centroid and center to parameter file from C 1 for the gyre over all TS, params for mdt also saved

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


# ---  cartopy directly to use local shapefiles ---
land_shp       = data_dir + 'ne_50m_land/ne_50m_land.shp'
coast_shp      = data_dir +'ne_50m_coastline/ne_50m_coastline.shp'
ice_poly_shp   = data_dir +'ne_50m_antarctic_ice_shelves_polys/ne_50m_antarctic_ice_shelves_polys.shp'
ice_lines_shp  = data_dir +'ne_50m_antarctic_ice_shelves_lines/ne_50m_antarctic_ice_shelves_lines.shp'

# --- Create cartopy features from your local files ---
land_feature      = cfeature.ShapelyFeature(shpreader.Reader(land_shp).geometries(),
                                             ccrs.PlateCarree(), facecolor='lightgray', edgecolor='none')
coast_feature     = cfeature.ShapelyFeature(shpreader.Reader(coast_shp).geometries(),
                                             ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)
ice_poly_feature  = cfeature.ShapelyFeature(shpreader.Reader(ice_poly_shp).geometries(),
                                             ccrs.PlateCarree(), facecolor='lightblue', edgecolor='none', alpha=0.5)
ice_lines_feature = cfeature.ShapelyFeature(shpreader.Reader(ice_lines_shp).geometries(),
                                             ccrs.PlateCarree(), facecolor='none', edgecolor='steelblue', linewidth=0.8)

# Load shapefiles as geodataframes
land_gdf = gpd.read_file(land_shp)
coast_gdf = gpd.read_file(coast_shp)
ice_poly_gdf = gpd.read_file(ice_poly_shp)
ice_lines_gdf = gpd.read_file(ice_lines_shp)

# load params and verts
ts_verts = np.load(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_verts.npz', allow_pickle=True)
ts_params_ds = xr.open_dataset(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_params.nc')
mean_verts = np.load(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_verts_mdt.npz', allow_pickle=True)
mean_param_ds = xr.open_dataset(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_params_mdt.nc')


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
        'verts'   : ts_verts[f'{tidx}_best'],
    })

gyre_centres = []

for result in to_plot:
    time_idx_i = result['time_idx']

    # --- Dateline-aware slice ---
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

        _land  = ft.clip_gdf_to_domain(land_gdf, lon_min, lon_max, lat_min, lat_max)
        _coast   = ft.clip_gdf_to_domain(coast_gdf, lon_min, lon_max, lat_min, lat_max)
        _ice_poly  = ft.clip_gdf_to_domain(ice_poly_gdf, lon_min, lon_max, lat_min, lat_max)
        _ice_lines = ft.clip_gdf_to_domain(ice_lines_gdf, lon_min, lon_max, lat_min, lat_max)

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

        _land  = ft.split_and_shift_gdf(land_gdf, lons_i, lat_min, lat_max)
        _coast = ft.split_and_shift_gdf(coast_gdf, lons_i, lat_min, lat_max)
        _ice_poly = ft.split_and_shift_gdf(ice_poly_gdf, lons_i, lat_min, lat_max)
        _ice_lines = ft.split_and_shift_gdf(ice_lines_gdf, lons_i, lat_min, lat_max)


    # --- Minimum DOT inside the contour only ---
    verts_i  = result['verts']
    mask_i, points_i  = ft.points_inside_contour(verts_i, lon_grid_i, lat_grid_i)
    var_masked = np.where(mask_i, var_2d_i, np.nan)

    # gyre centroid
    mask_flat = np.ravel(mask_i)
    lon_i = points_i[..., 0][mask_flat]
    lat_i = points_i[..., 1][mask_flat]
    dot_i = var_masked[mask_i]
    weights_i =  - dot_i # have the negative sign here to pull the centroid towards the min DOT values (as is the case for weddell and ross gyres)

    lon_centroid = np.nansum(lon_i*weights_i)/np.nansum(weights_i)
    lat_centroid = np.nansum(lat_i*weights_i)/np.nansum(weights_i)

    # gyre center
    min_idx    = np.unravel_index(np.nanargmin(var_masked), var_masked.shape)
    centre_lon = lon_grid_i[min_idx]
    centre_lat = lat_grid_i[min_idx]
    centre_var = var_masked[min_idx]

    gyre_centres.append({
        'time'      : result['time'],
        'time_idx'  : time_idx_i,
        'centre_lon': centre_lon,
        'centre_lat': centre_lat,
        'centre_var': centre_var,
        'centroid_lon': lon_centroid,
        'centroid_lat': lat_centroid,
    })


# --- MDT centre and centroid ---
mdt_contour_verts = mean_verts['mdt_contour_max_verts']
var_mean = mean_param_ds['var_mean'].values
lons_ref= mean_param_ds['lon'].values
lats_ref = mean_param_ds['lat'].values
lon_grid_ref, lat_grid_ref = np.meshgrid(lons_ref, lats_ref)

mask_mdt, points_mdt = ft.points_inside_contour(
    mdt_contour_verts, lon_grid_ref, lat_grid_ref)
var_masked_mdt = np.where(mask_mdt, var_mean, np.nan)


# centroid
mask_flat_mdt = np.ravel(mask_mdt)
lon_mdt = points_mdt[..., 0][mask_flat_mdt]
lat_mdt = points_mdt[..., 1][mask_flat_mdt]
weights_mdt = -var_masked_mdt[mask_mdt]
mdt_centroid_lon = np.nansum(lon_mdt * weights_mdt) / np.nansum(weights_mdt)
mdt_centroid_lat = np.nansum(lat_mdt * weights_mdt) / np.nansum(weights_mdt)

# centre
min_idx_mdt    = np.unravel_index(np.nanargmin(var_masked_mdt), var_masked_mdt.shape)
mdt_centre_lon = lon_grid_ref[min_idx_mdt]
mdt_centre_lat = lat_grid_ref[min_idx_mdt]
centre_var_mdt = var_masked_mdt[min_idx_mdt]

# update and save mean_contour_params.nc
mdt_updated = mean_param_ds.assign({
    'centre_lon'  : mdt_centre_lon,
    'centre_lat'  : mdt_centre_lat,
    'centroid_lon': mdt_centroid_lon,
    'centroid_lat': mdt_centroid_lat,
    'centre_var': centre_var_mdt,
})


print(f"Gyre centre computed for {len(gyre_centres)} timesteps.")

print('saving full ts center information...')
gyre_centres_df = pd.DataFrame(gyre_centres)
gyre_centres_ds = xr.Dataset.from_dataframe(gyre_centres_df.set_index('time'))
center_name = f'{tag}_center_centroid.nc'
gyre_centres_ds.to_netcdf(gyre_dir + center_name)
print('gyre centres saved to :',gyre_dir + center_name )

print('saving mean gyre location center information...')
mdt_updated.to_netcdf(gyre_dir + f'{tag}_center_centroid_mdt.nc')
print('mean gyre centres saved to :',gyre_dir + f'{tag}_center_centroid_mdt.nc' )
print(f"MDT centre:   ({mdt_centre_lon:.2f}°, {mdt_centre_lat:.2f}°)")
print(f"MDT centroid: ({mdt_centroid_lon:.2f}°, {mdt_centroid_lat:.2f}˚)")


# """"
#
# C2_gyre_centers.py
#
# - Loads dot data from C1
# - Calculates the center (min DOT) and centroid (min DOT distance vetweeb largest and smallest closed contours in the gyre)
# - Calculate the center and centroids for the mean DOT
# - Saves:  centroid and center to new  file for both full ts and mean gyre location (referred to as mdt)
#
# """""
# import numpy as np
# import matplotlib.pyplot as plt
# import xarray as xr
# import cartopy.crs as ccrs
# crs = ccrs.PlateCarree()
# import cartopy.io.shapereader as shpreader
# import cartopy.feature as cfeature
# from shapely.geometry import MultiPolygon
# import geopandas as gpd
# from shapely.geometry import box
# import matplotlib.cm as cm
# import pandas as pd
# from shapely.affinity import translate
# from matplotlib.path import Path
# from scipy.interpolate import interpn
# from scipy.interpolate import griddata
# from scipy.spatial import cKDTree
# import numpy.ma as ma
# import matplotlib.pyplot as plt
# import sys
# from scipy.ndimage import gaussian_filter
# from scipy.stats import pearsonr
# from pyproj import Geod
# from shapely.geometry import Polygon
# import warnings
# import os
#
#
# # Suppress only deprecation warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
#
# # PATHS
# workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
# data_dir = workdir + 'Data/'
# processing_dir = data_dir + 'C_gyre_processing/'
# script_dir = workdir + 'Scripts/'
# auxscriptdir = script_dir + 'aux_scripts/'
# save_dir = data_dir + 'C_gyre_processing/'
# sys.path.append(auxscriptdir)
# import aux_stereoplot as st
# from geometry_izzyv1 import grad_sphere
# import aux_func as ft
# clim_dir = data_dir + 'climate_indices/'
#
# # user settings gyre_name, variable, start_time ...
# gyre_name = 'weddell' #ross #weddell #kerguelen
# variable = 'dot' #'dot' #'sla'
# start_time = '2002-09-01'
# end_time = '2024-12-01'
#
# gyre_dir = processing_dir + gyre_name + '/'
# tag = f'{gyre_name}_{start_time}_{end_time}'
#
# # load dot ds
# file_path = data_dir + 'dot_all_30bmedian_egm2008_sig3.nc'
# ds = xr.open_dataset(file_path)
# ds = ds.sel(time=slice(start_time, end_time))
#
#
# # ---  cartopy directly to use local shapefiles ---
# land_shp       = data_dir + 'ne_50m_land/ne_50m_land.shp'
# coast_shp      = data_dir +'ne_50m_coastline/ne_50m_coastline.shp'
# ice_poly_shp   = data_dir +'ne_50m_antarctic_ice_shelves_polys/ne_50m_antarctic_ice_shelves_polys.shp'
# ice_lines_shp  = data_dir +'ne_50m_antarctic_ice_shelves_lines/ne_50m_antarctic_ice_shelves_lines.shp'
#
# # --- Create cartopy features from your local files ---
# land_feature      = cfeature.ShapelyFeature(shpreader.Reader(land_shp).geometries(),
#                                              ccrs.PlateCarree(), facecolor='lightgray', edgecolor='none')
# coast_feature     = cfeature.ShapelyFeature(shpreader.Reader(coast_shp).geometries(),
#                                              ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.8)
# ice_poly_feature  = cfeature.ShapelyFeature(shpreader.Reader(ice_poly_shp).geometries(),
#                                              ccrs.PlateCarree(), facecolor='lightblue', edgecolor='none', alpha=0.5)
# ice_lines_feature = cfeature.ShapelyFeature(shpreader.Reader(ice_lines_shp).geometries(),
#                                              ccrs.PlateCarree(), facecolor='none', edgecolor='steelblue', linewidth=0.8)
#
# # Load shapefiles as geodataframes
# land_gdf = gpd.read_file(land_shp)
# coast_gdf = gpd.read_file(coast_shp)
# ice_poly_gdf = gpd.read_file(ice_poly_shp)
# ice_lines_gdf = gpd.read_file(ice_lines_shp)
#
# # load params and verts
# ts_verts = np.load(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_verts.npz', allow_pickle=True)
# ts_params_ds = xr.open_dataset(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_params.nc')
# mean_verts = np.load(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_verts_mdt.npz', allow_pickle=True)
# mean_param_ds = xr.open_dataset(gyre_dir + f'{gyre_name}_{start_time}_{end_time}_params_mdt.nc')
#
#
# dot = ds['dot']
# MDT = dot.mean(dim='time')
# sla = dot - MDT
# ds['sla'] = sla
#
# # gyre specific settings (different masking lat limits and lat/lon areas)
# if gyre_name == 'ross':
#     lon_min  = 150        # western boundary of your sector
#     lon_max  =  -120  # eastern boundary of your sector
#     lat_min = -90
#     lat_max = -55
#     lat_mask_max = -70
#     crosses_dateline = True # set to False for normal sectors (<0 then >0)
# elif gyre_name == 'weddell':
#     lon_min = -80
#     lon_max = 50
#     lat_min = -90
#     lat_max = -50
#     lat_mask_max = -75
#     crosses_dateline = False # set to False for normal sectors (<0 then >0)
#
# lat_sigma = 1
# lon_sigma = 3
#
# to_plot = []
# for i, t in enumerate(ts_params_ds.time.values):
#     tidx = int(ts_params_ds['time_idx'].values[i])
#     to_plot.append({
#         'time'    : t,
#         'time_idx': tidx,
#         'verts'   : ts_verts[f'{tidx}_best'],
#     })
#
# gyre_centres = []
#
# for result in to_plot:
#     time_idx_i = result['time_idx']
#
#     # --- Dateline-aware slice ---
#     if not crosses_dateline:
#         ds_i = ds.sel(longitude=slice(lon_min, lon_max),
#                       latitude=slice(lat_min, lat_max),
#                       time=ds.time[time_idx_i])
#         lons_i = ds_i['longitude'].values
#         lats_i = ds_i['latitude'].values
#         lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
#         var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)
#
#         land_2d_i  = ds_i['land_mask'].values.T
#
#         total_mask = (land_2d_i == 0)
#         var_filled = np.where(total_mask, var_2d_i, 0.0)
#         weights = np.where(total_mask, 1.0, 0.0)
#         smoothed_vals  = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
#         smoothed_weights = gaussian_filter(weights,    sigma=(lat_sigma, lon_sigma))
#         smoothed  = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)
#         var_2d_smoothed  = np.where(total_mask, smoothed, var_2d_i)
#         var_2d_i  = np.where(land_2d_i == 1, np.nan, var_2d_smoothed)
#
#         lat_mask   = lat_grid_i < lat_mask_max
#         ocean_mask = ~np.isnan(var_2d_i)
#         var_2d_i   = np.where(lat_mask & ocean_mask, smoothed, var_2d_i)
#
#         _land  = ft.clip_gdf_to_domain(land_gdf, lon_min, lon_max, lat_min, lat_max)
#         _coast   = ft.clip_gdf_to_domain(coast_gdf, lon_min, lon_max, lat_min, lat_max)
#         _ice_poly  = ft.clip_gdf_to_domain(ice_poly_gdf, lon_min, lon_max, lat_min, lat_max)
#         _ice_lines = ft.clip_gdf_to_domain(ice_lines_gdf, lon_min, lon_max, lat_min, lat_max)
#
#     else:
#         ds_A = ds.sel(longitude=slice(lon_min, 180),
#                       latitude=slice(lat_min, lat_max),
#                       time=ds.time[time_idx_i])
#         ds_B = ds.sel(longitude=slice(-180, lon_max),
#                       latitude=slice(lat_min, lat_max),
#                       time=ds.time[time_idx_i])
#         ds_B = ds_B.assign_coords(longitude=ds_B.longitude.values + 360)
#         ds_i = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')
#         lons_i = ds_i['longitude'].values
#         lats_i = ds_i['latitude'].values
#         lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
#         var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)
#
#
#         land_2d_i  = ds_i['land_mask'].values.T
#
#         total_mask = (land_2d_i == 0)
#         var_filled = np.where(total_mask, var_2d_i, 0.0)
#         weights  = np.where(total_mask, 1.0, 0.0)
#         smoothed_vals  = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
#         smoothed_weights = gaussian_filter(weights,    sigma=(lat_sigma, lon_sigma))
#         smoothed   = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)
#         var_2d_smoothed  = np.where(total_mask, smoothed, var_2d_i)
#         var_2d_i  = np.where(land_2d_i == 1, np.nan, var_2d_smoothed)
#         lat_mask  = lat_grid_i < lat_mask_max
#         ocean_mask = ~np.isnan(var_2d_i)
#         var_2d_i = np.where(lat_mask & ocean_mask, smoothed, var_2d_i)
#
#         _land  = ft.split_and_shift_gdf(land_gdf, lons_i, lat_min, lat_max)
#         _coast = ft.split_and_shift_gdf(coast_gdf, lons_i, lat_min, lat_max)
#         _ice_poly = ft.split_and_shift_gdf(ice_poly_gdf, lons_i, lat_min, lat_max)
#         _ice_lines = ft.split_and_shift_gdf(ice_lines_gdf, lons_i, lat_min, lat_max)
#
#
#     # --- Minimum DOT inside the contour only ---
#     verts_i  = result['verts']
#     mask_i, points_i  = ft.points_inside_contour(verts_i, lon_grid_i, lat_grid_i)
#     var_masked = np.where(mask_i, var_2d_i, np.nan)
#
#     # gyre centroid - from Reagan et al, 2019 - good for assymetry of gyre analysis
#     mask_flat = np.ravel(mask_i)
#     lon_i = points_i[..., 0][mask_flat]
#     lat_i = points_i[..., 1][mask_flat]
#     dot_i = var_masked[mask_i]
#     weights_i =  - dot_i # have the negative sign here to pull the centroid towards the min DOT values (as is the case for weddell and ross gyres)
#
#     lon_centroid = np.nansum(lon_i*weights_i)/np.nansum(weights_i)
#     lat_centroid = np.nansum(lat_i*weights_i)/np.nansum(weights_i)
#
#     # gyre center
#     min_idx    = np.unravel_index(np.nanargmin(var_masked), var_masked.shape)
#     centre_lon = lon_grid_i[min_idx]
#     centre_lat = lat_grid_i[min_idx]
#     centre_var = var_masked[min_idx]
#
#     gyre_centres.append({
#         'time'      : result['time'],
#         'time_idx'  : time_idx_i,
#         'centre_lon': centre_lon,
#         'centre_lat': centre_lat,
#         'centre_var': centre_var,
#         'centroid_lon': lon_centroid,
#         'centroid_lat': lat_centroid,
#     })
#
#
# # --- MDT centre and centroid ---
# mdt_contour_verts = mean_verts['mdt_contour_verts']
# var_mean = mean_param_ds['var_mean'].values
# lons_ref= mean_param_ds['lon'].values
# lats_ref = mean_param_ds['lat'].values
# lon_grid_ref, lat_grid_ref = np.meshgrid(lons_ref, lats_ref)
#
# mask_mdt, points_mdt = ft.points_inside_contour(
#     mdt_contour_verts, lon_grid_ref, lat_grid_ref)
# var_masked_mdt = np.where(mask_mdt, var_mean, np.nan)
#
#
# # centroid
# mask_flat_mdt = np.ravel(mask_mdt)
# lon_mdt = points_mdt[..., 0][mask_flat_mdt]
# lat_mdt = points_mdt[..., 1][mask_flat_mdt]
# weights_mdt = -var_masked_mdt[mask_mdt]
# mdt_centroid_lon = np.nansum(lon_mdt * weights_mdt) / np.nansum(weights_mdt)
# mdt_centroid_lat = np.nansum(lat_mdt * weights_mdt) / np.nansum(weights_mdt)
#
# # centre
# min_idx_mdt    = np.unravel_index(np.nanargmin(var_masked_mdt), var_masked_mdt.shape)
# mdt_centre_lon = lon_grid_ref[min_idx_mdt]
# mdt_centre_lat = lat_grid_ref[min_idx_mdt]
# centre_var_mdt = var_masked_mdt[min_idx_mdt]
#
# # update and save mean_contour_params.nc
# mdt_updated = mean_param_ds.assign({
#     'centre_lon'  : mdt_centre_lon,
#     'centre_lat'  : mdt_centre_lat,
#     'centroid_lon': mdt_centroid_lon,
#     'centroid_lat': mdt_centroid_lat,
#     'centre_var': centre_var_mdt,
# })
#
#
# print(f"Gyre centre computed for {len(gyre_centres)} timesteps.")
#
# print('saving full ts center information...')
# gyre_centres_df = pd.DataFrame(gyre_centres)
# gyre_centres_ds = xr.Dataset.from_dataframe(gyre_centres_df.set_index('time'))
# center_name = f'{tag}_center_centroid.nc'
# gyre_centres_ds.to_netcdf(gyre_dir + center_name)
# print('gyre centres saved to :',gyre_dir + center_name )
#
# print('saving mean gyre location center information...')
# mdt_updated.to_netcdf(gyre_dir + f'{tag}_center_centroid_mdt.nc')
# print('mean gyre centres saved to :',gyre_dir + f'{tag}_center_centroid_mdt.nc' )
# print(f"MDT centre:   ({mdt_centre_lon:.2f}°, {mdt_centre_lat:.2f}°)")
# print(f"MDT centroid: ({mdt_centroid_lon:.2f}°, {mdt_centroid_lat:.2f}˚)")