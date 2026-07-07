""""

D3_monthly_mean_geostrophic_plots.py

same as D2 but calculates the mean ug and vg for each month

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
import calendar


# Suppress only deprecation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
script_dir = workdir + 'Scripts/'
processing_dir = data_dir + 'C_gyre_processing/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/D_geostrophic_vels/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st
from geometry_izzyv1 import grad_sphere
import aux_func as ft
clim_dir = data_dir + 'climate_indices/'


# user settings gyre_name, variable, start_time ...
gyre_name = 'weddell' #ross #weddell #kerguelen # full
variable = 'dot' #'dot' #'sla'
start_time = '2002-09-01'
end_time = '2024-12-01'
chosen_time = '2012-09-01'
month_key = 12
month_name = calendar.month_name[month_key]
quiver_step = 3

gyre_dir = processing_dir + gyre_name + '/'
gyre_fig_dir = fig_dir + gyre_name + '/monthly_aves/'
os.makedirs(gyre_fig_dir, exist_ok=True)
tag = f'{gyre_name}_{start_time}_{end_time}'

# load dot ds
file_path = data_dir + 'dot_all_30bmedian_egm2008_sig3.nc'

dot_ds = xr.open_dataset(file_path)
dot_ds = dot_ds.sel(time=slice(start_time, end_time))
time = dot_ds.time.values
lat = dot_ds.latitude.values
lon = dot_ds.longitude.values
ug = dot_ds.ug
vg = dot_ds.vg
time_key = dot_ds.get_index('time').get_loc(chosen_time)

if gyre_name != 'full':
    # load params and verts
    ts_verts = np.load(gyre_dir + f'{tag}_verts.npz', allow_pickle=True)
    ts_params_ds = xr.open_dataset(gyre_dir + f'{tag}_params.nc')
    mean_verts = np.load(gyre_dir + f'{tag}_verts_mdt.npz', allow_pickle=True)
    mean_param_ds = xr.open_dataset(gyre_dir + f'{tag}_params_mdt.nc')
    # load centers and centroids
    ts_center_centroid_ds = xr.open_dataset(gyre_dir + tag +  '_center_centroid.nc')
    mean_center_centroid_ds = xr.open_dataset(gyre_dir + tag +  '_center_centroid_mdt.nc')


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
elif gyre_name == 'full':
    lon_min = -180
    lon_max = 180
    lat_min = -90
    lat_max = -50
    lat_mask_max = -75
    crosses_dateline = False # set to False for normal sectors (<0 then >0)

lat_sigma = 1
lon_sigma = 3

to_plot =[]

for i,t in enumerate(time):
    if not crosses_dateline:
        ds_i = dot_ds.sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
            time = t
        )
    else:
        ds_A = dot_ds.sel(
            longitude=slice(lon_min, 180),
            latitude=slice(lat_min, lat_max),
            time = t
        )
        ds_B = dot_ds.sel(
            longitude=slice(-180, lon_max),
            latitude=slice(lat_min, lat_max),
            time = t
        )
        ds_B = ds_B.assign_coords(longitude=ds_B.longitude + 360)
        ds_i = xr.concat([ds_A, ds_B], dim='longitude')

    lons_i = ds_i.longitude.values
    lats_i = ds_i.latitude.values
    llons_i, llats_i = np.meshgrid(lons_i, lats_i)
    dot_i = np.where(ds_i.land_mask.T ==1, np.nan, ds_i[variable].T)

    # shift geostrophic velocities
    if crosses_dateline:
        ug_shift = dot_ds.ug.assign_coords(longitude=(dot_ds.longitude % 360)).sortby('longitude')
        vg_shift = dot_ds.vg.assign_coords(longitude=(dot_ds.longitude % 360)).sortby('longitude')
        ug_i = ug_shift.sel(longitude=lons_i, latitude=lats_i, time = t).T
        vg_i = vg_shift.sel(longitude=lons_i, latitude=lats_i, time = t).T
    else:
        ug_i = dot_ds.ug.sel(longitude=lons_i, latitude=lats_i, time = t).T
        vg_i = dot_ds.vg.sel(longitude=lons_i, latitude=lats_i, time = t).T



    to_plot.append({
    'time': t,
    'time_idx': i,
    'lon': lons_i,
    'lat': lats_i,
    'dot': dot_i,
    'ug': ug_i,
    'vg': vg_i,
    })

land_shp   = data_dir + 'ne_50m_land/ne_50m_land.shp'
coast_shp   = data_dir +'ne_50m_coastline/ne_50m_coastline.shp'
ice_poly_shp = data_dir +'ne_50m_antarctic_ice_shelves_polys/ne_50m_antarctic_ice_shelves_polys.shp'
ice_lines_shp  = data_dir +'ne_50m_antarctic_ice_shelves_lines/ne_50m_antarctic_ice_shelves_lines.shp'


land_feature  = cfeature.ShapelyFeature(shpreader.Reader(land_shp).geometries(),
                                             ccrs.PlateCarree(), facecolor='lightgray', edgecolor='none')
coast_feature  = cfeature.ShapelyFeature(shpreader.Reader(coast_shp).geometries(),
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


fig, ax = plt.subplots(figsize=(13, 6))


# recompute shapefiles for the reference domain
if not crosses_dateline:
    _land      = ft.clip_gdf_to_domain(land_gdf,      lon_min, lon_max, lat_min, lat_max)
    _coast     = ft.clip_gdf_to_domain(coast_gdf,     lon_min, lon_max, lat_min, lat_max)
    _ice_poly  = ft.clip_gdf_to_domain(ice_poly_gdf,  lon_min, lon_max, lat_min, lat_max)
    _ice_lines = ft.clip_gdf_to_domain(ice_lines_gdf, lon_min, lon_max, lat_min, lat_max)
else:
    _land      = ft.split_and_shift_gdf(land_gdf,      lons_i, lat_min, lat_max)
    _coast     = ft.split_and_shift_gdf(coast_gdf,     lons_i, lat_min, lat_max)
    _ice_poly  = ft.split_and_shift_gdf(ice_poly_gdf,  lons_i, lat_min, lat_max)
    _ice_lines = ft.split_and_shift_gdf(ice_lines_gdf, lons_i, lat_min, lat_max)


months = []
for var in to_plot:
    t = var['time']
    month_num = pd.Timestamp(t).month
    months.append(month_num)

month_groups = {}
for m in range(1,13):
    month_groups[m] = []

for i,m in enumerate(months):
    month_groups[m].append(i)

dot_monthly_mean = {}
ug_monthly_mean = {}
vg_monthly_mean = {}

for m, idx in month_groups.items():
    dot_monthly_stack = np.stack([to_plot[i]['dot'] for i in idx], axis=0)
    ug_monthly_stack = np.stack([to_plot[i]['ug'] for i in idx], axis=0)
    vg_monthly_stack = np.stack([to_plot[i]['vg'] for i in idx], axis=0)

    dot_monthly_mean[m] = np.nanmean(dot_monthly_stack, axis=0)
    ug_monthly_mean[m] = np.nanmean(ug_monthly_stack, axis=0)
    vg_monthly_mean[m] = np.nanmean(vg_monthly_stack, axis=0)

monthly_data = {}

for m in range(1, 13):
    monthly_data[m] = {
        'lon': to_plot[0]['lon'],
        'lat': to_plot[0]['lat'],
        'dot': dot_monthly_mean[m],
        'ug': ug_monthly_mean[m],
        'vg': vg_monthly_mean[m],
    }

monthly_data = monthly_data[month_key]

dot_plot = plt.pcolormesh(monthly_data['lon'], monthly_data['lat'], monthly_data['dot'], cmap='RdBu_r')
if gyre_name == 'full':
    quiver = plt.quiver(
        monthly_data['lon'][::quiver_step],
        monthly_data['lat'][::quiver_step],
        monthly_data['ug'][::quiver_step, ::quiver_step],
        monthly_data['vg'][::quiver_step, ::quiver_step],
        scale = 0.2, scale_units = 'inches', width = 0.0015
    )
else:
    quiver = plt.quiver(
        monthly_data['lon'][::quiver_step],
        monthly_data['lat'][::quiver_step],
        monthly_data['ug'][::quiver_step, ::quiver_step],
        monthly_data['vg'][::quiver_step, ::quiver_step],
        scale = 0.1, scale_units = 'inches', width = 0.0015
    )

# Shapefiles
_land.plot(ax=ax,      color='lightgray', edgecolor='none',  zorder=2)
_coast.plot(ax=ax,     color='none',      edgecolor='black', linewidth=0.8, zorder=3)
_ice_poly.plot(ax=ax,  color='lightblue', edgecolor='none',  alpha=0.5, zorder=2)
_ice_lines.plot(ax=ax, color='steelblue', edgecolor='none',  linewidth=0.8, zorder=3)

ax.set_aspect('auto')
ax.set_xlim(lons_i[0], lons_i[-1])
ax.set_ylim(lat_min, lat_max)

if crosses_dateline:
    tick_vals = np.arange(np.ceil(lons_i[0]/10)*10, lons_i[-1]+1, 30)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'{int(x-360)}°E' if x > 180 else f'{int(x)}°E' for x in tick_vals])

# Colour-map contours by time
cmap_t = cm.RdBu_r
n      = len(time)

sm = plt.cm.ScalarMappable(cmap=cmap_t, norm=plt.Normalize(0, n - 1))
sm.set_array([])
pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0+pos.width *0.1, pos.y0-0.14, pos.width * 0.8, pos.height*0.05])

# cbar_t = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01, fraction=0.02)
# cbar_t.set_label('Timestep index')
plt.colorbar(dot_plot, cax=cbar_ax, label=f'{variable} (m)', orientation='horizontal', pad=0.12)
ax.set_title(f'Mean Geostrophic Velocities for {month_name}')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
# plt.tight_layout()

file_save_name = f'{month_key}_{gyre_name}_quiver_{month_name}_plot_rectangular.png'
plt.savefig(gyre_fig_dir + file_save_name, dpi=300, bbox_inches='tight')

plt.show()