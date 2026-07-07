"""
C6b_center_centroid_plot_stereo.py

-Loads data from  C2 (params file)
- uses functions in aux_stereo
- Plots contours on stereo plot

"""
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
gyre_name = 'ross' #ross #weddell #kerguelen
variable = 'dot' #'dot' #'sla'
start_time = '2002-09-01'
end_time = '2024-12-01'
plot_key = 'centroids' # centroid #center

gyre_dir = processing_dir + gyre_name + '/'
gyre_fig_dir = fig_dir + gyre_name + '/centers_centroids/'
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
        'centre_lon'   : ts_center_centroid_ds['centre_lon'].values[i],
        'centre_lat' : ts_center_centroid_ds['centre_lat'].values[i],
        'mdt_centre_lon' : mean_center_centroid_ds['centre_lon'].values,
        'mdt_centre_lat' : mean_center_centroid_ds['centre_lat'].values,
        'centroid_lon' : ts_center_centroid_ds['centroid_lon'].values[i],
        'centroid_lat' : ts_center_centroid_ds['centroid_lat'].values[i],
        'mdt_centroid_lon': mean_center_centroid_ds['centroid_lon'].values,
        'mdt_centroid_lat': mean_center_centroid_ds['centroid_lat'].values
    })

land_shp       = data_dir + 'ne_50m_land/ne_50m_land.shp'
coast_shp      = data_dir +'ne_50m_coastline/ne_50m_coastline.shp'
ice_poly_shp   = data_dir +'ne_50m_antarctic_ice_shelves_polys/ne_50m_antarctic_ice_shelves_polys.shp'
ice_lines_shp  = data_dir +'ne_50m_antarctic_ice_shelves_lines/ne_50m_antarctic_ice_shelves_lines.shp'


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


for result in to_plot:
    time_idx_i = result['time_idx']

    if not crosses_dateline:
        ds_i = ds.sel(longitude=slice(lon_min, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        lons_i = ds_i['longitude'].values
        lats_i = ds_i['latitude'].values
        var_2d_i = np.where(ds_i['land_mask'].values.T == 1, np.nan, ds_i[variable].values.T)

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

# Background — use the first selected timestep's DOT field with dateline-aware slice
plot_idx = 50
ref = to_plot[plot_idx]

if not crosses_dateline:
    ds_ref = ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max),
        time=ds.time[ref['time_idx']]
    )
    lons_ref = ds_ref['longitude'].values
    lats_ref = ds_ref['latitude'].values
    var_ref  = np.where(ds_ref['land_mask'].values.T == 1, np.nan, ds_ref[variable].values.T)
else:
    ds_A = ds.sel(longitude=slice(lon_min, 180),
                  latitude=slice(lat_min, lat_max),
                  time=ds.time[ref['time_idx']])
    ds_B = ds.sel(longitude=slice(-180, lon_max),
                  latitude=slice(lat_min, lat_max),
                  time=ds.time[ref['time_idx']])
    ds_B = ds_B.assign_coords(longitude=ds_B.longitude.values + 360)
    ds_ref = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')
    lons_ref = ds_ref['longitude'].values
    lats_ref = ds_ref['latitude'].values
    var_ref  = np.where(ds_ref['land_mask'].values.T == 1, np.nan, ds_ref[variable].values.T)

lon_grid_ref, lat_grid_ref = np.meshgrid(lons_ref, lats_ref)


# ── Shared stereo base ────────────────────────────────────────────────────────
#spstere_plot_sector must return wedge_patch as 4th value
multiple = 5

plot_lat_min = np.round(np.min(lats_ref/multiple))*multiple
plot_lat_max = np.round(np.max(lats_ref/multiple))*multiple

# contours plot
if gyre_name == 'ross':
    # centres plot — new figure, same base
    fig, ax, m_sec, wedge_patch = st.spstere_plot_sector(
        lon_grid_ref, lat_grid_ref, var_ref,
        [-2.5, -1],
        'RdBu_r', f'{variable} (m)',
        lon_min=lon_min % 360,
        lon_max=lon_max % 360,
        lat_min=plot_lat_min,
        lat_max=plot_lat_max,
        contours=True,
        contour_levels=[-5000, -4000, -3000, -2000, -1000],
        # contour_colors=['pink', 'cyan', 'magenta'],
        contour_lws=[0.8, 0.8, 0.8],
        # contour_linestyles=[':', '-.', '--'],
        tglon=st.tglon, tglat=st.tglat, topo=st.topo
    )
    st.spstere_plot_gyre_centres(ax, m_sec, wedge_patch,
                                  to_plot, gyre_name = gyre_name, lat_mask_max = lat_mask_max, lon_sigma = lon_sigma, lat_sigma = lat_sigma , plot_mode=None, variable=variable)

    if plot_key == 'center':
        mdt_lon = float(mean_center_centroid_ds[f'centre_lon'].values)
        mdt_lat = float(mean_center_centroid_ds[f'centre_lat'].values)
        x_mdt, y_mdt = m_sec(mdt_lon % 360, mdt_lat)
    else:
        mdt_lon = float(mean_center_centroid_ds[f'centroid_lon'].values)
        mdt_lat = float(mean_center_centroid_ds[f'centroid_lat'].values)
        x_mdt, y_mdt = m_sec(mdt_lon % 360, mdt_lat)
    if abs(x_mdt) < 1e29 and abs(y_mdt) < 1e29:
        # sc = ax.scatter(x_mdt, y_mdt, color='black', s=80, marker='*', zorder=10, label='MDT centre')
        sc = ax.scatter(
            x_mdt, y_mdt, color = 'red', s = 150, marker = '*', zorder = 10, linewidths = 1, edgecolors = 'black', label = f'MDT {plot_key} '
        )
        sc.set_clip_path(wedge_patch)
        ax.legend(loc='lower left', fontsize=7, framealpha=0.95)

    save_fig_name = tag + plot_key + '_stereo.png'
    plt.savefig(gyre_fig_dir + save_fig_name, dpi=300, bbox_inches='tight')

    plt.show()

elif gyre_name == 'weddell' or gyre_name == 'kerguelen':

    # centres plot — new figure, same base
    fig, ax, m_sec, wedge_patch = st.spstere_plot_sector(
        lon_grid_ref, lat_grid_ref, var_ref,
        [-2.5, -0.5],
        'RdBu_r', f'{variable} (m)',
        lon_min=lon_min ,
        lon_max=lon_max ,
        lat_min=plot_lat_min,
        lat_max=plot_lat_max,
        contours=True,
        contour_levels=[-5000, -4000, -3000, -2000, -1000],
        # contour_colors=['pink', 'cyan', 'magenta'],
        contour_lws=[0.8, 0.8, 0.8],
        # contour_linestyles=[':', '-.', '--'],
        tglon=st.tglon, tglat=st.tglat, topo=st.topo
    )
    st.spstere_plot_gyre_centres(ax, m_sec, wedge_patch,
                                  to_plot, gyre_name = gyre_name, lat_mask_max = lat_mask_max, lon_sigma = lon_sigma, lat_sigma = lat_sigma , plot_mode=None, variable=variable)

    if plot_key == 'center':
        mdt_lon = float(mean_center_centroid_ds[f'centre_lon'].values)
        mdt_lat = float(mean_center_centroid_ds[f'centre_lat'].values)
        x_mdt, y_mdt = m_sec(mdt_lon % 360, mdt_lat)
    else:
        mdt_lon = float(mean_center_centroid_ds[f'centroid_lon'].values)
        mdt_lat = float(mean_center_centroid_ds[f'centroid_lat'].values)
        x_mdt, y_mdt = m_sec(mdt_lon % 360, mdt_lat)
    if abs(x_mdt) < 1e29 and abs(y_mdt) < 1e29:
        # sc = ax.scatter(x_mdt, y_mdt, color='black', s=80, marker='*', zorder=10, label='MDT centre')
        sc = ax.scatter(
            x_mdt, y_mdt, color = 'red', s = 150, marker = '*', zorder = 10, linewidths = 1, edgecolors = 'black', label = f'MDT {plot_key} '
        )
        sc.set_clip_path(wedge_patch)
        ax.legend(loc='lower left', fontsize=7, framealpha=0.95)

    save_fig_name = tag + plot_key + '_stereo.png'
    plt.savefig(gyre_fig_dir + save_fig_name, dpi=300, bbox_inches='tight')

    plt.show()