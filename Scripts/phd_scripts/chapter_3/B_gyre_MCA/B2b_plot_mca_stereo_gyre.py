"""
B2b_plot_mca_stereo_gyre.py
Load sector saved scores and comps
Plot MCA on a stereo grid for the gyre sector

last modified 27/05/2026"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import xeofs as xe
from pygments.modeline import modeline_re
import sys
import cartopy.crs as ccrs


crs = ccrs.PlateCarree()
import warnings
import os
warnings.filterwarnings('ignore')

# path = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
# save_dir = path + 'Oana/test_figs/'
# aux_scripts = path + 'Scripts/aux_scripts/'
# print(path)

workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
mca_dir = data_dir + 'mca_processing/sectors/'
clim_dir = data_dir + 'climate_indices/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/A_MCA/sectors/'


sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func
import aux_stereoplot as st

# select time range for vairable calculation
start_time = '2002-07-01'
end_time = '2024-12-01'

# Select gyre
gyre_name = 'ross' #ross #weddell #kerguelen

# set variable names
var_1_name = 'SLA'
var_1_units = 'cm/std'
var_2_name = 'WSC'
var_2_units = 'Nm^-3/std'

# set number of modes plotted
n_modes = 5

if gyre_name == 'ross':
    lon_min  = 140        # western boundary of your sector
    lon_max  =  -80  # eastern boundary of your sector
    lat_min = -90
    lat_max = -60
    sector_dir = mca_dir + 'ross/'
    vars_dir = sector_dir + 'preprocessed_vars/'
    scores_comps_dir = sector_dir + 'scores_comps/'
    fig_dir = fig_dir + 'ross/stereo/'
    crosses_dateline = True # set to False for normal sectors (<0 then >0)
elif gyre_name == 'weddell':
    lon_min = -80
    lon_max = 50
    sector_dir = mca_dir + 'weddell/'
    vars_dir = sector_dir + 'preprocessed_vars/'
    scores_comps_dir = sector_dir + 'scores_comps/'
    fig_dir = fig_dir + 'weddell/stereo/'
    crosses_dateline = False # set to False for normal sectors (<0 then >0)
elif gyre_name == 'kerguelen':
    lon_min = 60
    lon_max = 150
    sector_dir = mca_dir + 'kerguelen/'
    vars_dir = sector_dir + 'preprocessed_vars/'
    scores_comps_dir = sector_dir + 'scores_comps/'
    fig_dir = fig_dir + 'kerguelen/stereo/'
    crosses_dateline = False # set to False for normal sectors (<0 then >0)


save_dir = fig_dir + f'{var_1_name}_{var_2_name}/'
os.makedirs(save_dir, exist_ok=True)


# load dot dataset
dot_ds = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
# full ds time
dot_ds = dot_ds.sel(time=slice(start_time, end_time))
dot = dot_ds.dot.values
dot_time = dot_ds.time.values
lon = dot_ds.longitude.values
lat = dot_ds.latitude.values

print('loading scores and comps....')
# load scores and comps
scores_comps_ds = xr.open_dataset(scores_comps_dir + f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}_{gyre_name}.nc')
scores1 = scores_comps_ds['scores_1']
scores2 = scores_comps_ds['scores_2']
comps1 = scores_comps_ds['comps_1']
comps2 = scores_comps_ds['comps_2']

r_pears = []; p_pears = []
r_spear = []; p_spear = []
for m in range(1,6):
    x = scores1.sel(mode=m)
    y = scores2.sel(mode=m)
    r, p = scipy.stats.pearsonr(x, y)
    r_pears.append(np.round(r, 2)); p_pears.append(np.round(p, 2))
    r, p = scipy.stats.spearmanr(x, y)
    r_spear.append(np.round(r, 2)); p_spear.append(np.round(p, 2))


# # ── Manual sign flip to match reference paper ──────────────────────
flip_both = []  # modes where both fields need flipping

for mode_num in flip_both:
    comps1.loc[dict(mode=mode_num)]  *= -1
    comps2.loc[dict(mode=mode_num)]  *= -1
    scores1.loc[dict(mode=mode_num)] *= -1
    scores2.loc[dict(mode=mode_num)] *= -1


for mode_num in range(1, n_modes + 1):

    # Scale spatial patterns by std of the corresponding PC score
    # This preserves original units when reconstructed
    comp1_m = (comps1.sel(mode=mode_num) * scores1.sel(mode=mode_num).std().item())*100
    comp2_m = (comps2.sel(mode=mode_num) * scores2.sel(mode=mode_num).std().item())

    # Normalise the scores for the time series plot
    s1_norm = scores1.sel(mode=mode_num) / scores1.sel(mode=mode_num).std()
    s2_norm = scores2.sel(mode=mode_num) / scores2.sel(mode=mode_num).std()

    maxabs_1 = float(np.nanpercentile(np.abs(comp1_m.values), 99))
    maxabs_2 = float(np.nanpercentile(np.abs(comp2_m.values), 99))
    vlims_1 = [-maxabs_1, maxabs_1]
    vlims_2 = [-maxabs_2, maxabs_2]

    vlims_2 = [-0.5e-7, 0.5e-7]
    vlims_1 = [-2, 2]

    # --- Grid + wrap for stereo plotting ---
    lons = comp1_m.lon.values
    lats = comp1_m.lat.values

    lon_min = np.round(np.min(lons))
    lon_max = np.round(np.max(lons))
    lat_min = np.round(np.min(lats/10))*10 # round to nearest 10
    lat_max = np.round(np.max(lats/10))*10 # round to nearest 10
    print(lon_min, lon_max, lat_min, lat_max)
    llon, llat = np.meshgrid(lons, lats)

    llon_w1, llat_w1, comp1_wrapped = st.wrap_lon_for_plot(comp1_m.values, lons, lats)
    llon_w2, llat_w2, comp2_wrapped = st.wrap_lon_for_plot(comp2_m.values, lons, lats)

    cmap_1 = 'RdBu_r'
    cmap_2 = 'RdBu_r'
    cbar_units_1 = var_1_units # set as empty to leave space or edit for each mode (for paper figures)
    cbar_units_2 = var_2_units
    cbar_extend  = 'both'

    fig1, ax1, _, _ = st.spstere_plot_sector(
    llon, llat, comp1_m.values,
    vlims_1, cmap_1, cbar_units_1,
    lon_min=lon_min, lon_max=lon_max,
    lat_min=lat_min, lat_max=lat_max,
    contours=True,
    contour_levels=[-5000, -4000, -3000, -2000, -1000],
    #contour_colors=['pink', 'cyan', 'magenta'],
    contour_lws=[0.8, 0.8, 0.8],
    # contour_linestyles=[':', '-.', '--'],
    tglon=st.tglon, tglat=st.tglat, topo=st.topo
)

    if gyre_name == 'ross':
        fig1.tight_layout(rect=[0, 0, 1, 1])
        fig1.suptitle(
            f'MCA Mode {mode_num} — {var_1_name}',
            fontsize=12, fontweight='bold', y=0.86
        )
    if gyre_name == 'weddell':
        fig1.tight_layout(rect=[0, 0, 1, 1])
        fig1.suptitle(
            f'MCA Mode {mode_num} — {var_1_name}',
            fontsize=12, fontweight='bold', y=0.89
        )
    if gyre_name == 'kerguelen':
        fig1.tight_layout(rect=[0, 0, 1, 1])
        fig1.suptitle(
            f'MCA Mode {mode_num} — {var_1_name}',
            fontsize=12, fontweight='bold', y=0.99
        )



    fig1_name = f'mode_{mode_num}_{var_1_name}_{start_time}_{end_time}_{gyre_name}.png'
    fig1.savefig(save_dir + fig1_name, dpi=300)

    # ============================
    # FIGURE 2 — Stereo var_2
    # ============================
    # fig2, ax2, _ = st.spstere_plot(
    #     llon_w2, llat_w2, comp2_wrapped,
    #     vlims_2, 'RdBu_r', 'Nm^-3/std', bcolor=None
    # )

    fig2, ax2, _, _ = st.spstere_plot_sector(
    llon, llat, comp2_m.values,
    vlims_2, cmap_2, cbar_units_2,
    lon_min=lon_min, lon_max=lon_max,
    lat_min=lat_min, lat_max=lat_max,
    contours=True,
    contour_levels=[-5000, -4000, -3000, -2000, -1000],
    # contour_colors=['pink', 'cyan', 'magenta'],
    contour_lws=[0.8, 0.8, 0.8],
    # contour_linestyles=[':', '-.', '--'],
    tglon=st.tglon, tglat=st.tglat, topo=st.topo
)

    if gyre_name == 'ross':
        fig2.tight_layout(rect=[0, 0, 1, 1])
        fig2.suptitle(
            f'MCA Mode {mode_num} — {var_2_name}',
            fontsize=12, fontweight='bold', y=0.86
        )
    if gyre_name == 'weddell':
        fig2.tight_layout(rect=[0, 0, 1, 1])
        fig2.suptitle(
            f'MCA Mode {mode_num} — {var_2_name}',
            fontsize=12, fontweight='bold', y=0.89
        )
    if gyre_name == 'kerguelen':
        fig2.tight_layout(rect=[0, 0, 1, 1])
        fig2.suptitle(
            f'MCA Mode {mode_num} — {var_2_name}',
            fontsize=12, fontweight='bold', y=0.99
        )

    fig2_name = f'mode_{mode_num}_{var_2_name}_{start_time}_{end_time}_{gyre_name}.png'
    fig2.savefig(save_dir + fig2_name, dpi=300)

    # ============================
    # FIGURE 3 — Timeseries
    # ============================
    # s1_norm = s1 / s1.std()
    # s2_norm = s2 / s2.std()

    fig3, ax3 = plt.subplots(figsize=(15, 6), dpi=150)
    ax3.plot(s1_norm.time, s1_norm, label=var_1_name, linewidth=1.5, color = 'r')
    ax3.plot(s2_norm.time, s2_norm, label=var_2_name, linewidth=1.5, color = 'blue')
    ax3.axhline(0, color='grey', linewidth=0.7, linestyle='--')

    ax3.set_xlabel('Year')
    ax3.set_ylabel('Normalised score')
    ax3.set_title(
        f'MCA Mode {mode_num}: {var_1_name} vs {var_2_name}\n| r = {r_pears[mode_num-1]}, p = {p_pears[mode_num-1]}',
        fontsize=12, fontweight='bold'
    )
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=30)

    fig3_name = f'timeseries_mode_{mode_num}_{start_time}_{end_time}_{gyre_name}.png'
    fig3.savefig(save_dir + fig3_name, dpi=300)

    plt.show()
