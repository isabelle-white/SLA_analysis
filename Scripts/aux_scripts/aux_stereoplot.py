import numpy as np
from numpy import ma
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon as polyg

from palettable.colorbrewer.diverging import RdBu_11, PuOr_11

import sys

font = {'family' : 'serif', 'size' : 9}
plt.rc('font', **font)
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

#-----------------------------------------------------------------------------
# Define directories
# where data files ared
voldir = '/Users/iw2g24/PycharmProjects/CS2_extension/PhD/PhD_data/'
griddir = voldir + '../altimetry_cpom/3_grid_dot/'
eradir = voldir + 'reanalyses/'
topodir = voldir + 'topog/'

with xr.open_dataset(topodir + 'bathy_mask_2km.nc') as contour:
    print(contour.keys())
cglat, cglon = np.meshgrid(contour.lat, contour.lon)
shelf_mask = contour.bathy_mask.values


# with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon_izzy.nc') as topo:
#     print('TOPO keys',topo.keys())
# tglat, tglon = np.meshgrid(topo.lat, topo.lon)

with xr.open_dataset(topodir + 'coarse_gebco_p5x1_latlon_izzy.nc') as _topo:
    print('TOPO keys', _topo.keys())
    topo = _topo  # expose as module variable

# expose tglon/tglat as module variables
tglat, tglon = np.meshgrid(topo.lat.values, topo.lon.values)


# 2. MAP area and GRID
#------------------------------------------------------------------
print("Defining map area ...\n")
# coastline resolution can vary from coarse to fine: c, l, i, h, f
plt.ioff()
m = Basemap(projection='spstere',
            boundinglat=-50,
            lon_0=180,
            resolution='i',
            round=True)

# extract segments north of 60 S
print("Getting coastlines north of 60 S ... \n")
coast = m.drawcoastlines(linewidth=0)
segments = coast.get_segments()
lat_seg, lon_seg = [], []

for j in range(len(segments)):
    xy = np.vstack(segments[j]) 
    lons_b, lats_b = m(xy[:, 0], xy[:, 1], inverse=True)
    lats_bm = ma.masked_outside(lats_b, -60, -50.)
    lons_bm = ma.masked_array(lons_b, lats_bm.mask)
    if lats_bm.count() > 0:
        lat_seg.append(lats_bm)
        lon_seg.append(lons_bm)

# -----------------------
# coastlines south of 60S
print("Importing Antarctic Digital Database shapefile. \n")
# add path to coastline files
coastdir = "/Users/iw2g24/PycharmProjects/CS2_extension/PhD/PhD_data/land_masks/holland_vic/"
sys.path.append(coastdir)
from coastline_Antarctica import coastline

# extract coastline points from files (Tiago/Paul Holland)
## it returns a list of lists
[ilon_land, ilat_land, ilon_ice, ilat_ice] = coastline()


def spstere_plot(varlon, varlat, var, vlims, cmap, cbar_units, bcolor):
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = m.pcolormesh(varlon, varlat, var, 
                      vmin=vlims[0], vmax=vlims[1],
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True,
                      shading='auto')
    m.drawcoastlines(linewidth=0.25, zorder=3)
    m.fillcontinents(color='w')
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.09, extend='both')
    cb.ax.set_title(cbar_units)
    
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=3, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightsteelblue',
                    zorder=3, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=3, color='dimgrey')

    if bcolor is not None:
        m.plot(contour.llon, contour.llat,
               latlon=True,
               color=bcolor, zorder=3, lw=1)

    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10),
                    zorder=2, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=2, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m



#------------------------------------------------------------------
def spstere_plot_extend(varlon, varlat, var, vlims, cmap, cbar_units, cbar_extend):
    #plt.ion() (6.5, 7)
    fig, ax = plt.subplots(figsize=(4.5, 5), dpi=300)
    cs = m.pcolormesh(varlon, varlat, var,
                      vmin=vlims[0], vmax=vlims[1],
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True,
                      shading='auto')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='w')
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.07,
                      extend=cbar_extend)
    cb.ax.set_title(cbar_units)
    
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=1, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightgrey',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, color='dimgrey')
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=10, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=10, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)

    #m.drawmeridians([0, 160, -150, -62], linewidth=2, ax=ax)
    x1, y1 = m(190, -80.5)
    ax.annotate(r"$80^\circ$S", xy=(x1, y1),
                xycoords='data', xytext=(x1, y1),
                textcoords='data', zorder=10)
    x2, y2 = m(186, -70.5)
    ax.annotate(r"$70^\circ$S", xy=(x2, y2),
                xycoords='data', xytext=(x2, y2),
                textcoords='data', zorder=10)
    x3, y3 = m(184, -60.5)
    ax.annotate(r"$60^\circ$S", xy=(x3, y3),
                xycoords='data', xytext=(x3, y3),
                textcoords='data', zorder=10)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m

#------------------------------------------------------------------
def spstere_contourf(varlon, varlat, var, contourf_kw, cmap, cbar_units):
    #plt.ion()
    fig, ax = plt.subplots(figsize=(6.5, 7))
    cs = m.contourf(varlon, varlat, var,
                      contourf_kw=contourf_kw,
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True, shading='auto')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='w')
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.07,
                      extend=cbar_extend)
    cb.ax.set_title(cbar_units)
    
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=1, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightgrey',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, color='dimgrey')
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=10, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=10, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)

    #m.drawmeridians([0, 160, -150, -62], linewidth=2, ax=ax)
    x1, y1 = m(190, -80.5)
    ax.annotate(r"$80^\circ$S", xy=(x1, y1),
                xycoords='data', xytext=(x1, y1),
                textcoords='data', zorder=10)
    x2, y2 = m(186, -70.5)
    ax.annotate(r"$70^\circ$S", xy=(x2, y2),
                xycoords='data', xytext=(x2, y2),
                textcoords='data', zorder=10)
    x3, y3 = m(184, -60.5)
    ax.annotate(r"$60^\circ$S", xy=(x3, y3),
                xycoords='data', xytext=(x3, y3),
                textcoords='data', zorder=10)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m

def spstere_plot_nofig(fig, ax, varlon, varlat, var, vlims, cmap, cbar_units):
    m = Basemap(projection='spstere',
                boundinglat=-50,
                lon_0=180,
                resolution='i',
                round=True, ax=ax)
    cs = m.pcolormesh(varlon, varlat, var,
                      vmin=vlims[0], vmax=vlims[1],
                      cmap=cmap,
                      latlon=True, 
                      rasterized=True, shading='auto')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='w')
    """
    cb = fig.colorbar(cs, ax=ax, 
                      orientation='horizontal',
                      shrink=.7, pad=.07,
                      extend='both')
    cb.ax.set_title(cbar_units)
    """
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=1, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightgrey',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, color='dimgrey')
    """ 
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=10, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=10, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)

    #m.drawmeridians([0, 160, -150, -62], linewidth=2, ax=ax)
    x1, y1 = m(190, -80.5)
    ax.annotate(r"$80^\circ$S", xy=(x1, y1),
                xycoords='data', xytext=(x1, y1),
                textcoords='data', zorder=10)
    x2, y2 = m(186, -70.5)
    ax.annotate(r"$70^\circ$S", xy=(x2, y2),
                xycoords='data', xytext=(x2, y2),
                textcoords='data', zorder=10)
    x3, y3 = m(184, -60.5)
    ax.annotate(r"$60^\circ$S", xy=(x3, y3),
                xycoords='data', xytext=(x3, y3),
    """
    ax.set_rasterization_zorder(0)

    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    plt.tight_layout()
    return fig, ax, m, cs
def spstere_frame():
    #plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w', 
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightsteelblue',
                    zorder=2, edgecolor='dimgrey',
                    linewidth=0.5, transform=ax.transData)
        ax.add_patch(poly)

    for k in range(len(lat_seg)):
        m.plot(lon_seg[k], lat_seg[k], 
               lw=0.5, latlon=True, 
               zorder=2, color='dimgrey')
    # parallels and meridians
    m.drawparallels(np.arange(-80., -50., 10), 
                    zorder=1, linewdith=0.25, ax=ax)
    m.drawmeridians(np.arange(0., 360., 30.), 
                    zorder=1, labels=[1, 1, 1, 1],
                    linewidth=0.25, ax=ax)
    ax.set_rasterization_zorder(0)
    
    # don't clip the map boundary circle
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    ax.set_rasterization_zorder(0)

    fig.tight_layout(rect=[0, -.1, 1, .98])
    return fig, ax, m

def spstere_contour_plot(varlon, varlat, var, levels, cmap, cbar_units, bcolor):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Main contour plot
    cs = m.contourf(
        varlon, varlat, var,
        levels=levels,
        cmap=cmap,
        latlon=True,
        linewidths=0.7,
        zorder=2
    )

    # Coastlines + land
    m.drawcoastlines(linewidth=0.25, zorder=3)
    m.fillcontinents(color='w')

    # Colorbar
    cb = fig.colorbar(
        cs, ax=ax,
        orientation='horizontal',
        shrink=.8, pad=.09, extend='both'
    )
    cb.ax.set_title(cbar_units)

    # Land polygons
    for k in range(len(ilon_land)):
        xf, yf = m(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(
            xyf, facecolor='w',
            zorder=3, edgecolor='dimgrey',
            linewidth=0.5, transform=ax.transData
        )
        ax.add_patch(poly)

    # Ice polygons
    for k in range(len(ilon_ice)):
        xf, yf = m(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(
            xyf, facecolor='lightsteelblue',
            zorder=3, edgecolor='dimgrey',
            linewidth=0.5, transform=ax.transData
        )
        ax.add_patch(poly)

    # Segments
    for k in range(len(lat_seg)):
        m.plot(
            lon_seg[k], lat_seg[k],
            lw=0.5, latlon=True,
            zorder=3, color='dimgrey'
        )

    # Optional boundary overlay
    if bcolor is not None:
        m.plot(
            contour.llon, contour.llat,
            latlon=True,
            color=bcolor, zorder=3, lw=1
        )

    # Parallels + meridians
    m.drawparallels(
        np.arange(-80., -50., 10),
        zorder=2, linewidth=0.25, ax=ax
    )
    m.drawmeridians(
        np.arange(0., 360., 30.),
        zorder=2, labels=[1, 1, 1, 1],
        linewidth=0.25, ax=ax
    )

    # Circular boundary
    circle = m.drawmapboundary(linewidth=1, color='k')
    circle.set_clip_on(False)

    fig.tight_layout(rect=[0, -.1, 1, 1])
    return fig, ax, m
def spstere_plot_sector(varlon, varlat, var, vlims, cmap, cbar_units,
                        lon_min=160, lon_max=240,
                        lat_min=-78, lat_max=-55,
                        contours=False,
                        contour_levels=None,
                        contour_colors=None,      #  list, e.g. ['k','grey','lightgrey']
                        contour_lws=None,         # list, e.g. [1.0, 0.7, 0.5]
                        contour_linestyles=None,  #  e.g. ['-', '--', ':']
                        tglon=None, tglat=None, topo=None):


    lon_centre = (lon_min + lon_max) / 2.0
    lon_0 = lon_centre


    m_sec = Basemap(projection='spstere',
                    boundinglat=lat_max,
                    lon_0=lon_0,
                    resolution='i',
                    round=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. ----- Build wedge clip path for the loaded sector ------
    # bottom arc (inner, at lat_min) — lon_min → lon_max
    n_arc = 200
    lons_bot = np.linspace(lon_min, lon_max, n_arc)
    lats_bot = np.full(n_arc, lat_min)

    # top arc (outer, at lat_max) — lon_max → lon_min  (close the shape)
    lons_top = np.linspace(lon_max, lon_min, n_arc)
    lats_top = np.full(n_arc, lat_max)

    # convert both arcs to map projection coords
    x_bot, y_bot = m_sec(lons_bot, lats_bot)
    x_top, y_top = m_sec(lons_top, lats_top)

    # concatenate into a closed polygon
    wedge_x = np.concatenate([x_bot, x_top])
    wedge_y = np.concatenate([y_bot, y_top])
    wedge_verts = np.column_stack([wedge_x, wedge_y])

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    wedge_path   = Path(wedge_verts)
    wedge_patch  = PathPatch(wedge_path, transform=ax.transData,
                             facecolor='none', edgecolor='none')
    ax.add_patch(wedge_patch)   # needed so clip_path references a drawn artist

    # 2. ---- Load and clip the data ----
    cs = m_sec.pcolormesh(varlon, varlat, var,
                          vmin=vlims[0], vmax=vlims[1],
                          cmap=cmap, latlon=True,
                          rasterized=True, shading='auto',
                          zorder=1)
    cs.set_clip_path(wedge_patch)   # ← clip data to wedge

    #  3. ---- change ocean back in the wedge if there is ever no data  ------
    from matplotlib.patches import Polygon as mplPoly
    ocean_bg = mplPoly(wedge_verts, facecolor='seashell',
                       edgecolor='none', zorder=0, transform=ax.transData)
    ax.add_patch(ocean_bg)

    #  ---- 4. Use the Holland and Vic coastlines ---
    for k in range(len(ilon_land)):
        xf, yf = m_sec(ilon_land[k], ilat_land[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='w',
                     zorder=3, edgecolor='dimgrey',
                     linewidth=0.5, transform=ax.transData)
        poly.set_clip_path(wedge_patch)
        ax.add_patch(poly)

    for k in range(len(ilon_ice)):
        xf, yf = m_sec(ilon_ice[k], ilat_ice[k])
        xyf = np.c_[xf, yf]
        poly = polyg(xyf, facecolor='lightsteelblue',
                     zorder=3, edgecolor='dimgrey',
                     linewidth=0.5, transform=ax.transData)
        poly.set_clip_path(wedge_patch)
        ax.add_patch(poly)

    # coastlines 60S–50S
    for k in range(len(lat_seg)):
        lines = m_sec.plot(lon_seg[k], lat_seg[k],
                           lw=0.5, latlon=True,
                           zorder=3, color='dimgrey')
        for line in lines:
            line.set_clip_path(wedge_patch)

    # 5. --- Bathymetry contours from Gebco ---
    if contours and topo is not None and tglon is not None and tglat is not None:

        elev = topo.elevation.values.copy()
        if elev.shape != tglon.shape:
            elev = elev.T

        # --- convert all coords to 0–360 ----
        tglon_360  = tglon % 360
        lon_min_360 = lon_min % 360   # 280°
        lon_max_360 = lon_max % 360   # 50°

        # ---- handle wrap-around sectors (e.g. 280° to 50° crosses 360°) ----
        if lon_min_360 > lon_max_360:
            # sector wraps: keep points where lon >= lon_min OR lon <= lon_max
            lon_outside = (tglon_360 < lon_min_360) & (tglon_360 > lon_max_360)
        else:
            # normal sector: keep points where lon_min <= lon <= lon_max
            lon_outside = (tglon_360 < lon_min_360) | (tglon_360 > lon_max_360)

        outside = lon_outside | (tglat < lat_min) | (tglat > lat_max)
        elev_masked = np.ma.array(elev, mask=outside)

        n_unmasked = (~elev_masked.mask).sum()
        print(f"Unmasked points after wrap fix: {n_unmasked}")

        if n_unmasked == 0:
            print("WARNING: still no unmasked points — check lon conventions")
        else:
            emin = elev_masked.compressed().min()
            emax = elev_masked.compressed().max()
            print(f"Masked elev range: {emin:.0f} to {emax:.0f}")
            levels_in_range = [l for l in contour_levels if emin < l < emax]
            print(f"Levels within data range: {levels_in_range}")

            # --- convert back to −180 to 180 for Basemap latlon=True ----
            tglon_180 = np.where(tglon_360 > 180, tglon_360 - 360, tglon_360)

    # --- use black solid lines as defaults if not provided ---
            n_levels = len(contour_levels)
            if contour_colors     is None: contour_colors     = ['darkgrey'] * n_levels
            if contour_lws        is None: contour_lws        = [0.7] * n_levels
            if contour_linestyles is None: contour_linestyles = ['-'] * n_levels

            # --- draw one contour at a time so each gets its own style ---
            for level, color, lw, ls in zip(contour_levels, contour_colors,
                                             contour_lws, contour_linestyles):
                cs_i = m_sec.contour(
                    tglon_180, tglat, elev_masked,
                    levels=[level], # single level at a time
                    colors=[color],
                    linewidths=[lw],
                    linestyles=[ls],
                    latlon=True,
                    zorder=6
                )
                # clip this level's artists to wedge
                for artist in ax.get_children():
                    if hasattr(artist, 'get_paths') and hasattr(artist, 'set_clip_path'):
                        if artist.get_zorder() == 6:
                            artist.set_clip_path(wedge_patch)
                            artist.set_clip_on(True)

    # --- clip ---
    for artist in ax.get_children():
        if hasattr(artist, 'get_paths') and hasattr(artist, 'set_clip_path'):
            if artist.get_zorder() == 6:
                artist.set_clip_path(wedge_patch)
                artist.set_clip_on(True)


    # # --- 7. Add gridlines (clipped to wedge) ---
    # draw meridians
    mer = m_sec.drawmeridians(np.arange(lon_min, lon_max + 1, 10.),
                               zorder=2, linewidth=0.4, ax=ax,
                               labels=[0, 0, 0, 0], color='grey')
    for key, (lines, labels) in mer.items():
        for line in lines:
            line.set_clip_path(wedge_patch)
            line.set_clip_on(True)

    # draw parallels
    par = m_sec.drawparallels(np.arange(lat_min, -49, 5.),
                               zorder=2, linewidth=0.4, ax=ax,
                               labels=[0, 0, 0, 0], color='grey')
    for key, (lines, labels) in par.items():
        for line in lines:
            line.set_clip_path(wedge_patch)
            line.set_clip_on(True)

    # -- 8. Add a border to the plot ---
    # draw the wedge outline as a thick black border
    border = plt.Polygon(wedge_verts, fill=False,
                         edgecolor='black', linewidth=0.5,
                         zorder=10, transform=ax.transData)
    ax.add_patch(border)

    all_x = np.concatenate([x_bot, x_top])
    all_y = np.concatenate([y_bot, y_top])

    # --- 9. Add lon and lat labels on the plot edges ----
    # compute a single offset scale used for BOTH meridian and parallel labels
    # so they sit at the same visual distance from the wedge edge
    meridional_offset = 0.03 * (all_x.max() - all_x.min())   # tune once here
    zonal_offset = 0.03 * (all_x.max() - all_x.min())

    # MERIDIAN LABELS — true radial offset from outer arc
    for lon_label in np.arange(lon_min, lon_max + 2, 10.):
        x0, y0 = m_sec(lon_label, lat_max)
        x1, y1 = m_sec(lon_label, lat_max - 2.5)   # 2° inward for stable direction
        dx = x0 - x1
        dy = y0 - y1
        norm = np.sqrt(dx**2 + dy**2)
        xl = x0 + (dx / norm) * meridional_offset
        yl = y0 + (dy / norm) * meridional_offset

        if lon_label > 180:
            txt = f"{360 - int(lon_label)}°W"
        elif lon_label == 180:
            txt = "180°"
        else:
            txt = f"{int(lon_label)}°E"

        # rotate label to follow the arc tangent
        tang_x, tang_y = m_sec(lon_label + 1, lat_max)
        tang_dx = tang_x - x0
        tang_dy = tang_y - y0
        angle = np.degrees(np.arctan2(tang_dy, tang_dx))

        ax.text(xl, yl, txt, fontsize=7, ha='center', va='center',
                zorder=5, rotation=angle, rotation_mode='anchor')

    # PARALLEL LABELS — offset perpendicularly outward from left edge
    for lat_label in np.arange(lat_min, -49, 5.):
        # point ON the left edge
        x0, y0 = m_sec(lon_min, lat_label)
        # point just inside — used to compute edge direction
        x1, y1 = m_sec(lon_min + 1, lat_label)
        dx = x0 - x1
        dy = y0 - y1
        norm = np.sqrt(dx**2 + dy**2)
        # place label one offset-distance outside the left edge
        xl = x0 + (dx / norm) * zonal_offset
        yl = y0 + (dy / norm) * zonal_offset
        ax.text(xl, yl, f"{abs(int(lat_label))}°S",
                fontsize=7, ha='center', va='center', zorder=5)

    # --- 10. Limit the axis so the edges match the stereo 'wedge' ----
    all_x = np.concatenate([x_bot, x_top])
    all_y = np.concatenate([y_bot, y_top])
    pad_x = (all_x.max() - all_x.min()) * 0.08
    pad_y = (all_y.max() - all_y.min()) * 0.08
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax.set_aspect('equal')
    ax.axis('off')   # hide the plain rectangular frame

    # --- 11. Add a colour bar for the mca data ---
    cb = fig.colorbar(cs, ax=ax, orientation='horizontal',
                      shrink=0.4, pad=0.02, extend='both')
    cb.ax.set_title(cbar_units, fontsize=9)

    fig.tight_layout()
    return fig, ax, m_sec

# def spstere_sector_plot(varlon, varlat, var, vlims, cmap, cbar_units,
#                         lonmin, lonmax, bcolor=None):
#     """
#     Polar stereographic sector plot using Basemap.
#     Matches style of spstere_plot but clips to a longitude sector.
#     """
#
#     # Create figure/axes
#     fig, ax = plt.subplots(figsize=(5, 5))
#
#     # Convert longitudes to 0–360 for masking
#     varlon360 = np.where(varlon < 0, varlon + 360, varlon)
#
#     # Correct wrap-around masking
#     if lonmin < lonmax:
#         mask = (varlon360 < lonmin) | (varlon360 > lonmax)
#     else:
#         mask = (varlon360 > lonmax) & (varlon360 < lonmin)
#
#     var_sector = np.copy(var)
#     var_sector[mask] = np.nan
#
#     # # Mask outside the sector
#     # mask = (varlon < lonmin) | (varlon > lonmax)
#     # var_sector = np.copy(var)
#     # var_sector[mask] = np.nan
#
#     # Main pcolormesh
#     cs = m.pcolormesh(
#         varlon, varlat, var_sector,
#         vmin=vlims[0], vmax=vlims[1],
#         cmap=cmap,
#         latlon=True,
#         rasterized=True,
#         shading='auto'
#     )
#
#     # Coastlines + land
#     m.drawcoastlines(linewidth=0.25, zorder=3)
#     m.fillcontinents(color='w')
#
#     # Colorbar
#     cb = fig.colorbar(
#         cs, ax=ax,
#         orientation='horizontal',
#         shrink=.7, pad=.09, extend='both'
#     )
#     cb.ax.set_title(cbar_units)
#
#     # Draw parallels/meridians
#     m.drawparallels(np.arange(-80., -50., 10),
#                     zorder=2, linewidth=0.25, ax=ax)
#     m.drawmeridians(np.arange(0., 360., 30.),
#                     zorder=2, labels=[1, 1, 1, 1],
#                     linewidth=0.25, ax=ax)
#
#     # Circular boundary
#     circle = m.drawmapboundary(linewidth=1, color='k')
#     circle.set_clip_on(False)
#
#     fig.tight_layout(rect=[0, -.1, 1, 1])
#     return fig, ax, m
#
#
# def spstere_sector_contour_plot(varlon, varlat, var, levels, cmap, cbar_units,
#                                 lonmin, lonmax, bcolor=None):
#
#     fig, ax = plt.subplots(figsize=(5, 5))
#
#     # Convert longitudes to 0–360 for masking
#     varlon360 = np.where(varlon < 0, varlon + 360, varlon)
#
#     # Correct wrap-around masking
#     if lonmin < lonmax:
#         mask = (varlon360 < lonmin) | (varlon360 > lonmax)
#     else:
#         mask = (varlon360 > lonmax) & (varlon360 < lonmin)
#
#     var_sector = np.copy(var)
#     var_sector[mask] = np.nan
#
#     cs = m.contour(
#         varlon, varlat, var_sector,
#         levels=levels,
#         cmap=cmap,
#         latlon=True,
#         linewidths=0.7,
#         zorder=2
#     )
#
#     m.drawcoastlines(linewidth=0.25, zorder=3)
#     m.fillcontinents(color='w')
#
#     cb = fig.colorbar(
#         cs, ax=ax,
#         orientation='horizontal',
#         shrink=.7, pad=.09, extend='both'
#     )
#     cb.ax.set_title(cbar_units)
#
#     m.drawparallels(np.arange(-80., -50., 10),
#                     zorder=2, linewidth=0.25, ax=ax)
#     m.drawmeridians(np.arange(0., 360., 30.),
#                     zorder=2, labels=[1, 1, 1, 1],
#                     linewidth=0.25, ax=ax)
#
#     circle = m.drawmapboundary(linewidth=1, color='k')
#     circle.set_clip_on(False)
#
#     fig.tight_layout(rect=[0, -.1, 1, 1])
#     return fig, ax, m
#
#
# # def find_largest_closed_contour_cartopy(lon2d, lat2d, field, lonmin, lonmax, levels=60):
# #     """
# #     Detect largest closed contour inside a chosen longitude sector.
# #     Returns:
# #         best_level : float
# #         best_lon   : 1D array
# #         best_lat   : 1D array
# #     """
# #
# #     # --- Convert longitudes to 0–360 for masking ---
# #     lon360 = np.where(lon2d < 0, lon2d + 360, lon2d)
# #
# #     # Sector mask (wrap-around handled automatically)
# #     if lonmin < lonmax:
# #         mask = (lon360 < lonmin) | (lon360 > lonmax)
# #     else:
# #         mask = ~((lon360 >= lonmin) | (lon360 <= lonmax))
# #
# #     field_masked = np.copy(field)
# #     field_masked[mask] = np.nan
# #
# #     # --- Run contour detection in PlateCarree ---
# #     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
# #
# #     cs = ax.contour(
# #         lon360, #lon2d had 180 divide
# #         lat2d,
# #         field_masked,
# #         levels=levels,
# #         transform=ccrs.PlateCarree()
# #     )
# #
# #     closed = []
# #
# #     for level, seglist in zip(cs.levels, cs.allsegs):
# #         for seg in seglist:
# #             if seg.shape[0] < 4:
# #                 continue
# #
# #             # Check closure
# #             if not np.allclose(seg[0], seg[-1], atol=1e-6):
# #                 continue
# #
# #             # Compute polygon area
# #             x = seg[:, 0]
# #             y = seg[:, 1]
# #             area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
# #                                 np.dot(y, np.roll(x, 1)))
# #
# #             if area > 0:
# #                 closed.append((level, seg, area))
# #
# #     plt.close(fig)
# #
# #     if len(closed) == 0:
# #         return None, None, None
# #
# #     best_level, best_seg, best_area = max(closed, key=lambda x: x[2])
# #
# #     # --- Convert longitudes back to -180–180 if desired ---
# #     best_lon = np.where(best_seg[:, 0] > 180, best_seg[:, 0] - 360, best_seg[:, 0])
# #     best_lat = best_seg[:, 1]
# #
# #     return best_level, best_lon, best_lat
# #
# #     # return best_level, best_seg[:, 0], best_seg[:, 1]
# #
#
# def plot_closed_contour_on_stereo(m, best_lon, best_lat, color='w', lw=0.5):
#     """
#     Plot a closed contour polygon (lon/lat) onto a Basemap stereographic map.
#     """
#     if best_lon is None:
#         print("No closed contour to plot.")
#         return
#
#     m.plot(
#         best_lon,
#         best_lat,
#         latlon=True,
#         color=color,
#         linewidth=lw,
#         zorder=10
#     )
#
# # def extract_closed_contour_for_time(ds_dot, t_index, lonmin, lonmax, levels=60):
# #     """
# #     Extract largest closed contour for a single time index.
# #     Returns:
# #         best_level, best_lon, best_lat
# #     """
# #     # Extract field at this time
# #     field = ds_dot.isel(time=t_index).values
# #
# #     # Build lon/lat grids
# #     lon = ds_dot.longitude.values
# #     lat = ds_dot.latitude.values
# #     lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')
# #
# #     # Use your Cartopy-based function
# #     best_level, best_lon, best_lat = find_largest_closed_contour_cartopy(
# #         lon2d, lat2d, field,
# #         lonmin=lonmin,
# #         lonmax=lonmax,
# #         levels=levels
# #     )
# #
# #     return best_level, best_lon, best_lat
#
#
# def find_largest_closed_contour_cartopy(lon2d, lat2d, field, lonmin, lonmax, levels=60):
#     """
#     Detect largest closed contour inside a chosen longitude sector.
#     Works with sectors that may wrap across 0/360 (e.g. 280 → 60).
#
#     Returns:
#         best_level : float
#         best_lon   : 1D array (in -180–180)
#         best_lat   : 1D array
#     """
#
#     # --- Convert longitudes to 0–360 for masking ---
#     lon360 = np.where(lon2d < 0, lon2d + 360, lon2d)
#
#     # --- Build mask for the chosen sector (in 0–360) ---
#     if lonmin < lonmax:
#         # simple case, e.g. 60 → 120
#         in_sector = (lon360 >= lonmin) & (lon360 <= lonmax)
#     else:
#         # wrap-around case, e.g. 280 → 60
#         in_sector = (lon360 >= lonmin) | (lon360 <= lonmax)
#
#     field_masked = np.copy(field)
#     field_masked[~in_sector] = np.nan
#
#     # --- Contour in PlateCarree with true geographic longitudes ---
#     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
#
#     cs = ax.contour(
#         lon360,          # 0–360 longitudes
#         lat2d,
#         field_masked,
#         levels=levels,
#         transform=ccrs.PlateCarree()
#     )
#
#     closed = []
#
#     for level, seglist in zip(cs.levels, cs.allsegs):
#         for seg in seglist:
#             if seg.shape[0] < 4:
#                 continue
#
#             # Check closure
#             if not np.allclose(seg[0], seg[-1], atol=1e-6):
#                 continue
#
#             x = seg[:, 0]
#             y = seg[:, 1]
#
#             # Simple polygon area in lon/lat space (good enough for ranking)
#             area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
#                                 np.dot(y, np.roll(x, 1)))
#
#             if area > 0:
#                 closed.append((level, seg, area))
#
#     plt.close(fig)
#
#     if len(closed) == 0:
#         return None, None, None
#
#     best_level, best_seg, best_area = max(closed, key=lambda x: x[2])
#
#     # Convert longitudes back to -180–180 for plotting
#     best_lon = np.where(best_seg[:, 0] > 180, best_seg[:, 0] - 360, best_seg[:, 0])
#     best_lat = best_seg[:, 1]
#
#     return best_level, best_lon, best_lat
#
# def extract_closed_contour_for_time(ds_dot, t_index, lonmin, lonmax, levels=60):
#     """
#     Extract largest closed contour for a single time index.
#     ds_dot: DataArray with dims (time, lon, lat) or (time, x, y) with longitude/latitude coords.
#     """
#     field = ds_dot.isel(time=t_index).values
#
#     lon = ds_dot.longitude.values
#     lat = ds_dot.latitude.values
#     lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')
#
#     best_level, best_lon, best_lat = find_largest_closed_contour_cartopy(
#         lon2d, lat2d, field,
#         lonmin=lonmin,
#         lonmax=lonmax,
#         levels=levels
#     )
#
#     return best_level, best_lon, best_lat
#

