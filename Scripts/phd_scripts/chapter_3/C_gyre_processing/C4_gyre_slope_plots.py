""""

C4_gyre_slope.py

- Loads contour data from C1 and the mean DOT contour
- Calculates lateral slope /spatial gradient - other people have used this as a proxy for intensity in NH / Arctic
- Plot contours on a rectangular plot/coordinates

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
gyre_fig_dir = fig_dir + gyre_name + '/lateral_slope_plots/'
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

lateral_slope = []

for result in to_plot:
    time_idx_i = result['time_idx']

    if not crosses_dateline:
        ds_i = ds.sel(longitude=slice(lon_min, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)

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
        ds_i  = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)

        land_2d_i = ds_i['land_mask'].values.T

        ocean_mask = ~np.isnan(var_2d_i)

        var_2d_i = np.where(ocean_mask,  var_2d_i, np.nan)

        _land  = ft.split_and_shift_gdf(land_gdf, lons_i, lat_min, lat_max)
        _coast = ft.split_and_shift_gdf(coast_gdf, lons_i, lat_min, lat_max)
        _ice_poly = ft.split_and_shift_gdf(ice_poly_gdf, lons_i, lat_min, lat_max)
        _ice_lines = ft.split_and_shift_gdf(ice_lines_gdf, lons_i, lat_min, lat_max)

    lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)
    df_dx_i, df_dy_i = grad_sphere(var_2d_i, lon_grid_i, lat_grid_i)
    lateral_slope_i = np.sqrt((df_dx_i**2 + df_dy_i**2))

    lateral_slope.append({
        'time': result['time'],
        'lateral_slope': lateral_slope_i,
        'lon_grid_i': lon_grid_i,
        'lat_grid_i': lat_grid_i,
    })

for plot_index in range(len(lateral_slope)):
    plot_time = str(lateral_slope[plot_index]['time'])[:7]
    print(plot_time)
    lateral_slope_t = lateral_slope[ plot_index]['lateral_slope']
    lon_grid_t = lateral_slope[ plot_index]['lon_grid_i']
    lat_grid_t = lateral_slope[ plot_index]['lat_grid_i']


    lon_min_plot = lon_grid_t.min()
    lon_max_plot = lon_grid_t.max()
    lat_min_plot = lat_grid_t.min()
    lat_max_plot = lat_grid_t.max()

    fig, ax = plt.subplots(figsize=(12, 4))
    # Shapefiles
    _land.plot(ax=ax,      color='lightgray', edgecolor='none',  zorder=2)
    _coast.plot(ax=ax,     color='none',      edgecolor='black', linewidth=0.8, zorder=3)
    _ice_poly.plot(ax=ax,  color='lightblue', edgecolor='none',  alpha=0.5, zorder=2)
    _ice_lines.plot(ax=ax, color='steelblue', edgecolor='none',  linewidth=0.8, zorder=3)

    ax.set_aspect('auto')

    ax.pcolormesh(lon_grid_t, lat_grid_t, lateral_slope_t, cmap='RdYlBu_r')
    cbar = fig.colorbar(ax.pcolormesh(lon_grid_t, lat_grid_t, lateral_slope_t, cmap='RdYlBu_r'), ax=ax)
    cbar.set_label('Lateral slope magnitude')

    ax.set_title(f'Lateral Slope Magnitude at t = {plot_time}')

    file_save_name = f'{gyre_name}_{plot_time}.png'
    plt.savefig(gyre_fig_dir + file_save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'saved plot for {plot_time}')
    # plt.show()
print(f'saved all timesteps to : {gyre_fig_dir+ file_save_name}')
