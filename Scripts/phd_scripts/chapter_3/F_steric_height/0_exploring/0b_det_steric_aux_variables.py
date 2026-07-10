"""
0d_plot_sha_mca_stereo.py

calculate mca for sla_sha and sla_eha

sha = steric height anom (due to density changes)
eha = eustatic height anom (due to volume changes)
wsc = wind stress crul
"""

import sys
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import os
import xeofs as xe
from pygments.modeline import modeline_re
import sys
import cartopy.crs as ccrs
import scipy

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
script_dir = workdir + 'Scripts/'
data_dir = workdir + 'Data/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/F_steric_height/'
scores_comps_dir = data_dir + '/mca_processing/F_steric_height/scores_comps/'
mca_dir = data_dir + 'mca_processing/full/'


os.makedirs(fig_dir, exist_ok=True)

sys.path.append(auxscriptdir)
from geometry_izzyv1 import grad_sphere
from regression_izzyv1 import linregress_3D
from regression_izzyv1 import linregress_3D_spatial_time
import aux_func as ft
import mca_preprocessing_func as mca_func
import aux_stereoplot as st

# set variable names
var_1_name = 'SP'
var_2_name = 'SHA'

save_dir = fig_dir + 'full/stereo/'  + f'{var_1_name}_{var_2_name}/'
os.makedirs(save_dir, exist_ok=True)

# set number of modes plotted
n_modes = 5

sla_sha_eha_ds_det = xr.open_dataset(data_dir+'steric_height_cocks/sla_sha_eha_det.nc')

time = sla_sha_eha_ds_det['time'].values
lon = sla_sha_eha_ds_det.longitude
lat = sla_sha_eha_ds_det.latitude

print(lon)
print(lat)

sla = sla_sha_eha_ds_det["sla_det"].values #time, lat, lon
sha = sla_sha_eha_ds_det["sha_det"].values # time, lat, lon
eha = sla_sha_eha_ds_det["eha_det"].values #time, lat, lon


print(lon.values, lat.values)

dot_ds = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
# full ds time
dot = dot_ds.dot.values
dot_time = dot_ds.time.values
lon_ds = dot_ds.longitude.values
lat_ds = dot_ds.latitude.values

print(lon_ds, lat_ds)





print('loading scores and comps....')
# load scores and comps
scores_comps_ds = xr.open_dataset(scores_comps_dir + f'{var_1_name}_{var_2_name}_scores_comps_full_ts.nc')
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
# flip_both = flip all four outputs (comps1, comps2, scores1, scores2)
# flip_osc  = flip only OSC side (comps2, scores2) — pattern ok, relative sign wrong
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

    maxabs_1 = float(np.nanpercentile(np.abs(comp1_m.values), 90))
    maxabs_2 = float(np.nanpercentile(np.abs(comp2_m.values), 90))
    vlims_1 = [-maxabs_1, maxabs_1]
    vlims_2 = [-maxabs_2, maxabs_2]

    # vlims_2 = [-0.5,0.5]
    # vlims_1 = [-0.5, 0.5]

    # --- Grid + wrap for stereo plotting ---
    lons = comp1_m.longitude.values
    lats = comp1_m.latitude.values

    # llon_w1, llat_w1, comp1_wrapped = st.wrap_lon_for_plot(comp1_m.values, lons, lats)
    # llon_w2, llat_w2, comp2_wrapped = st.wrap_lon_for_plot(comp2_m.values, lons, lats)

    llon_w1, llat_w1, comp1_wrapped = st.wrap_lon_for_plot(
    comp1_m.values.copy(), lons.copy(), lats.copy()
)
    llon_w2, llat_w2, comp2_wrapped = st.wrap_lon_for_plot(
        comp2_m.values.copy(), lons.copy(), lats.copy()
    )

    print(mode_num, 'comp1_m range:', np.nanmin(comp1_m.values), np.nanmax(comp1_m.values))
    print(mode_num, 'comp1_wrapped range:', np.nanmin(comp1_wrapped), np.nanmax(comp1_wrapped))
    print('any non-nan?', np.isfinite(comp1_wrapped).sum(), '/', comp1_wrapped.size)

    # ============================
    # FIGURE 1 — Stereo var_1
    # ============================
    fig1, ax1, _ = st.spstere_plot(
        llon_w1, llat_w1, comp1_wrapped,
        vlims_1, 'RdBu_r',  'cm/std', bcolor=None
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    fig1.suptitle(
        f'MCA Mode {mode_num} — {var_1_name}',
        fontsize=12, fontweight='bold'
    )

    fig1_name = f'mode_{mode_num}_{var_1_name}_full_ts.png'
    fig1.savefig(save_dir + fig1_name, dpi=300)

    # ============================
    # FIGURE 2 — Stereo var_2
    # ============================
    fig2, ax2, _ = st.spstere_plot(
        llon_w2, llat_w2, comp2_wrapped,
        vlims_2, 'RdBu_r', 'cm/std', bcolor=None
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.suptitle(
        f'MCA Mode {mode_num} — {var_2_name}',
        fontsize=12, fontweight='bold'
    )

    fig2_name = f'mode_{mode_num}_{var_2_name}__full_ts.png'
    fig2.savefig(save_dir + fig2_name, dpi=300)

    # ============================
    # FIGURE 3 — Timeseries
    # ============================
    # s1_norm = s1 / s1.std()
    # s2_norm = s2 / s2.std()

    fig3, ax3 = plt.subplots(figsize=(15, 6), dpi=150)
    ax3.plot(s1_norm.time, s1_norm, label=var_1_name, linewidth=1.5, color='r')
    ax3.plot(s2_norm.time, s2_norm, label=var_2_name, linewidth=1.5, color='b')
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

    fig3_name = f'timeseries_mode_{mode_num}_full_ts.png'
    fig3.savefig(save_dir + fig3_name, dpi=300)

    plt.show()
