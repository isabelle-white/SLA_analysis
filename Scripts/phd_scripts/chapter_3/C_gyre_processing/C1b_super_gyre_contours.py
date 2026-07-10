""""

C1b_super_gyre_contours.py

- Loads dot data
- selects gyre region of interest
- Sets parameters for that gyre
- Calculates contours and selects the largest closed contour at each time step
- Calcualtes area and other parameters per time step (for each largest closed contour)
- Calculates the time-mean (MDT) field (var_mean) and finds its largest closed contour
    --> this is mean gyre position

- Saves: parameters for the gyre over all TS, params for mdt gyre loc, verts for both (ragged file saved as npz)

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
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
save_dir = data_dir + 'C_gyre_processing/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st
from geometry_izzyv1 import grad_sphere
import aux_func as ft
clim_dir = data_dir + 'climate_indices/'

# user settings gyre_name, variable, start_time ...
gyre_name = 'super_gyre' #ross #weddell #super_gyre
variable = 'dot' #'dot' #'sla'

start_time = '2002-07-01'
end_time = '2024-12-01'
#
# start_time = '2003-01-01'
# end_time = '2010-12-01'

# start_time = '2011-01-01'
# end_time = '2015-12-01'

# start_time = '2016-01-01'
# end_time = '2024-12-01'

n_levels = 300        # number of contour levels to test
min_area = 0.5        # minimum enclosed area (km^2) to keep a contour
lat_sigma = 1
lon_sigma = 3
n_top_contours = 25
closed = 'yes' #'no'

# load dot ds
file_path = data_dir + 'dot_all_30bmedian_egm2008_sig3.nc'
ds = xr.open_dataset(file_path)
ds = ds.sel(time=slice(start_time, end_time))

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
    gap_lon_min, gap_lon_max = 156, 175 # this is the balleny islands
    gap_lat_min, gap_lat_max = -69, -60
    lat_mask_max = -70
    crosses_dateline = True # set to False for normal sectors (<0 then >0)
elif gyre_name == 'weddell':
    lon_min = -80
    lon_max = 50
    lat_min = -90
    lat_max = -50
    gap_lon_min, gap_lon_max = -30, -20 # this is the central islands north of the Weddell Sea
    gap_lat_min, gap_lat_max = -62, -50
    lat_mask_max = -70
    crosses_dateline = False # set to False for normal sectors (<0 then >0)
elif gyre_name == 'super_gyre':
    lon_min  = -80        #
    lon_max  =  -120  #
    lat_min = -90
    lat_max = -55
    gap_lon_min, gap_lon_max = 156, 175 # this is the balleny islands
    gap_lat_min, gap_lat_max = -69, -60
    lat_mask_max = -70
    crosses_dateline = True # set to False for normal sectors (<0 then >0)


# mean loop for all time steps between start and end time
all_timestep_results = []
# ssh_fields = []
var_stack = []

R_earth = 6371

for time_idx_i, time_val in enumerate(ds.time.values):

    # --- Dateline-aware slice ---
    if not crosses_dateline:
        ds_slice_i = ds.sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
            time=ds.time[time_idx_i]
        )
        lons_i = ds_slice_i['longitude'].values
        lats_i = ds_slice_i['latitude'].values
        var_2d_i  = ds_slice_i[variable].values.T
        lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)

        mdt_2d_i  = MDT.sel(
                longitude=slice(lon_min, lon_max),
                latitude=slice(lat_min, lat_max)
            ).values.T


    else:
        ds_A = ds.sel(longitude=slice(lon_min, 180),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        ds_B = ds.sel(longitude=slice(-180, lon_max),
                      latitude=slice(lat_min, lat_max),
                      time=ds.time[time_idx_i])
        ds_B = ds_B.assign_coords(longitude=ds_B.longitude.values + 360)

        ds_slice_i = xr.concat([ds_A, ds_B], dim='longitude', data_vars='all')
        lons_i = ds_slice_i['longitude'].values
        lats_i = ds_slice_i['latitude'].values
        var_2d_i  = ds_slice_i[variable].values.T

        lon_grid_i, lat_grid_i = np.meshgrid(lons_i, lats_i)

        mdt_A  = MDT.sel(longitude=slice(lon_min, 180),      # ← insert here
                        latitude=slice(lat_min, lat_max))
        mdt_B = MDT.sel(longitude=slice(-180, lon_max),
                            latitude=slice(lat_min, lat_max))
        mdt_B  = mdt_B.assign_coords(longitude=mdt_B.longitude.values + 360)
        mdt_2d_i  = xr.concat([mdt_A, mdt_B], dim='longitude').values.T

    # smoothing using the params set in the if gyre_name == loop...
    nan_mask = np.isnan(var_2d_i)
    var_filled = np.where(nan_mask, 0 , var_2d_i)
    weights = np.where(nan_mask, 0, 1 ).astype(float) #where nanmask is true = 0, where it is false = 1

    smoothed_vals = gaussian_filter(var_filled, sigma=(lat_sigma, lon_sigma))
    smoothed_weights = gaussian_filter(weights, sigma=(lat_sigma, lon_sigma))

    with np.errstate( invalid='ignore'):
        smoothed = np.where(smoothed_weights>0, smoothed_vals/ smoothed_weights, np.nan)

    # give mask S of 70˚
    lat_mask = lat_grid_i<lat_mask_max
    ocean_mask = ~np.isnan(var_2d_i)
    total_mask = lat_mask & ocean_mask
    var_2d_smoothed = np.where(total_mask, smoothed, var_2d_i)
    var_2d_i = var_2d_smoothed

    # ssh_2d_i  = (ds_slice_i['dot'] + ds_slice_i['sla']).values.T
    # ssh_2d_i  = (ds_slice_i['dot']).values.T
    land_2d_i = ds_slice_i['land_mask'].values.T

    # interpolating masked islands for Ross and Weddell Regions - prevents low resolution impacting closed contour calcualtions

    if gap_lon_min is not None:
        gap_lon_idx = np.where((lons_i >= gap_lon_min) & (lons_i <= gap_lon_max))[0]
        gap_lat_idx = np.where((lats_i >= gap_lat_min) & (lats_i <= gap_lat_max))[0]

        if len(gap_lon_idx) > 0 and len(gap_lat_idx) > 0:
            sub    = var_2d_i[np.ix_(gap_lat_idx, gap_lon_idx)]
            lons_gap = lons_i[gap_lon_idx]
            lats_gap = lats_i[gap_lat_idx]
            llon_gap, llat_gap = np.meshgrid(lons_gap, lats_gap)

            valid_gap = ~np.isnan(sub)
            nan_gap   = ~valid_gap

            if valid_gap.sum() >= 4 and nan_gap.sum() > 0:
                src_pts = np.column_stack([llon_gap[valid_gap], llat_gap[valid_gap]])
                src_val = sub[valid_gap]
                tgt_pts = np.column_stack([llon_gap[nan_gap],   llat_gap[nan_gap]])

                filled = griddata(src_pts, src_val, tgt_pts, method='linear')

                still_nan = np.isnan(filled)
                if still_nan.any():
                    filled[still_nan] = griddata(src_pts, src_val,
                                                 tgt_pts[still_nan], method='nearest')
                sub[nan_gap] = filled
                var_2d_i[np.ix_(gap_lat_idx, gap_lon_idx)] = sub

    #  contour checking statement,  skip if not enough valid data to contour
    var_min = np.nanmin(var_2d_i)
    var_max = np.nanmax(var_2d_i)

    if not np.isfinite(var_min) or not np.isfinite(var_max) or var_max <= var_min:
        print(f"Skipping {str(time_val)[:10]} — insufficient valid data")
        all_timestep_results.append({'time': time_val, 'time_idx': time_idx_i, 'found': False})
        continue

    # finding the closed contours
    levels_i = np.linspace(var_min, var_max, n_levels)
    fig, ax = plt.subplots()
    cs_i = ax.contour(lon_grid_i, lat_grid_i, var_2d_i, levels=levels_i)
    plt.close(fig)

    all_closed_i = []
    for level, segs in zip(cs_i.levels, cs_i.allsegs):
        for verts in segs:
            if len(verts) == 0:
                continue
            if closed == 'yes':
                if not np.allclose(verts[0], verts[-1], atol=1e-6):
                    continue

            x, y = verts[:, 0], verts[:, 1] # lon/lat in degs
            x_rad, y_rad = x*(np.pi/180), y*(np.pi/180)
            x_cartesian = R_earth * x_rad* np.cos(y_rad)
            y_cartesian = R_earth * y_rad
            area = 0.5 * np.abs(
                np.dot(x_cartesian, np.roll(y_cartesian, 1)) - np.dot(y_cartesian, np.roll(x_cartesian, 1))
            )

            if area >= min_area:
                all_closed_i.append((area, level, verts))

    all_closed_i.sort(key=lambda t: t[0], reverse=True) #use largest area, hence reverse

    # keep every contour that passes the lat filter
    valid_contours_i = [(area, level, verts) for area, level, verts in all_closed_i
                         if verts[:, 1].min() >= -75]

    if not valid_contours_i:
        all_timestep_results.append({
            'time': time_val,
            'time_idx': time_idx_i,
            'found': False
        })
        continue

    # keep only the top n_top_contours by area
    valid_contours_i.sort(key=lambda t: t[0], reverse=True)
    top_contours_i = valid_contours_i[:n_top_contours]

    if closed == 'yes':
        all_timestep_results.append({
            'time'      : time_val,
            'time_idx'  : time_idx_i,
            'found'     : True,
            'levels'    : [c[1] for c in top_contours_i],
            'verts_list': [c[2] for c in top_contours_i],
            'areas'     : [c[0] for c in top_contours_i],
    })
    else:
            all_timestep_results.append({
        'time'      : time_val,
        'time_idx'  : time_idx_i,
        'found'     : True,
        'levels'    : [c[1] for c in top_contours_i],
        'verts_list': [c[2] for c in top_contours_i],
})

    if time_idx_i % 20 == 0:
        print(f"Processed {time_idx_i+1}/{len(ds.time.values)}  —  {str(time_val)[:10]}")

    # append all var_2d_i (smoothed and interpolated SSH) to var_stack for mdt
    var_stack.append(var_2d_i.copy())

print(f"\nDone. {sum(r['found'] for r in all_timestep_results)} of {len(all_timestep_results)} timesteps have a contour.")

print('saving variables for full time series gyre calculation...')

# # SAve calcualted contours and params
gyre_dir = save_dir + gyre_name + '/'
os.makedirs(gyre_dir, exist_ok=True)

if closed == 'yes':
    tag = f'{gyre_name}_{start_time}_{end_time}_{n_top_contours}_closed_contours'
else:
    tag = f'{gyre_name}_{start_time}_{end_time}_{n_top_contours}_open_contours'
vert_name = f'{tag}_verts.npz'
# mdt_vert_name = f'{tag}_verts_mdt.npz'

found = [r for r in all_timestep_results if r['found']]

npz_dict = {}
for r in found:
    for j, verts in enumerate(r['verts_list']):
        npz_dict[f"{r['time_idx']}_{j}"] = verts
np.savez(gyre_dir + vert_name, **npz_dict)

print(f"Saved verts  {vert_name}")