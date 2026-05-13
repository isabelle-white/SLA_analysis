# A2c_plot_mca_stereo_sector.py
"""
Loads MCA scores + components for the full domain, and produces three figures
for a selected mode:

  Fig 1 — Stereographic plot of comps1 (var_1)
  Fig 2 — Stereographic plot of comps2 (var_2)
  Fig 3 — Scores timeseries with Pearson/Spearman correlations

Set save_figs = True/False to save figs

Last modified: 15/04/2026
"""

import numpy as np
import xarray as xr
import os
import sys
import scipy.stats
import matplotlib.pyplot as plt

# set paths
workdir      = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir     = workdir + 'Data/'
mca_dir      = data_dir + 'mca_processing/'
script_dir   = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st

fig_dir  = workdir + 'Figures/'
save_dir = fig_dir + 'A_MCA/'

# variable names and secotr name

var_1_name  = 'osc'
var_2_name  = 'sla'
sector_name = 'Ross' # don't use full
mode        = 1 # which MCA mode to plot

deseas    = False
detrended = True
save_figs = False # set False to skip saving

# # colourmap + limits for each variable — adjust as needed
# vlims_1 = [None, None] # comps1 (var_1)
# vlims_2 = [None, None] # comps2 (var_2)

cmap_1 = 'RdBu_r'
cmap_2 = 'RdBu_r'
cbar_units_1 = ''# set as empty to leave space or edit for each mode (for paper figures)
cbar_units_2 = ''
cbar_extend  = 'both'

#load files using if deseas or detrended
suffix  = 'deseas' if deseas else 'with_seasonal_cycle'
suffix += '_det'   if detrended else '_raw'

file_name        = f'{var_1_name}_{var_2_name}_{sector_name}_{suffix}.nc'
scores_file_name = 'scores_' + file_name
comps_file_name  = 'comps_'  + file_name


#load mca results for the full region from A1
scores_ds = xr.open_dataset(mca_dir + scores_file_name)
comps_ds  = xr.open_dataset(mca_dir + comps_file_name)

scores1 = scores_ds['scores1']
scores2 = scores_ds['scores2']
comps1  = comps_ds['comps1']
comps2  = comps_ds['comps2']


n_modes = scores1.sizes['mode']
print(f"Loaded: {file_name}  ({n_modes} modes)")

# calculate the correlations between different modes
corrs = []
for m in range(1, n_modes + 1):
    x = scores1.sel(mode=m).values
    y = scores2.sel(mode=m).values
    r_p, p_p = scipy.stats.pearsonr(x, y)
    r_s, p_s = scipy.stats.spearmanr(x, y)
    corrs.append({
        'mode'      : m,
        'r_pearson' : np.round(r_p, 3),
        'p_pearson' : np.round(p_p, 3),
        'r_spearman': np.round(r_s, 3),
        'p_spearman': np.round(p_s, 3),
    })

c = corrs[mode - 1]
print(f"Mode {mode}  |  Pearson r={c['r_pearson']} (p={c['p_pearson']})  "
      f"|  Spearman r={c['r_spearman']} (p={c['p_spearman']})")

# create a shared grid for both modes using lat and lon
comp1_m = comps1.sel(mode=mode)
comp2_m = comps2.sel(mode=mode)

maxabs1 = np.nanmax(np.abs(comp1_m.values))
maxabs2 = np.nanmax(np.abs(comp2_m.values))
maxabs = max(maxabs1, maxabs2)
vlims_1 = [-maxabs, maxabs]
vlims_2 = [-maxabs, maxabs]

lons = comp1_m.lon.values
lats = comp1_m.lat.values
#

lon_min = np.round(np.min(lons))
lon_max = np.round(np.max(lons))
lat_min = -80
lat_max = -50

print(lon_min, lon_max, lat_min, lat_max)

llon, llat = np.meshgrid(lons, lats)


fig1, ax1, _ = st.spstere_plot_sector(
    llon, llat, comp1_m.values,
    vlims_1, cmap_1, cbar_units_1,
    lon_min=lon_min, lon_max=lon_max,
    lat_min=lat_min, lat_max=lat_max,
    contours=True,
    contour_levels=[-5000, -3000, -1000],
    contour_colors=['pink', 'cyan', 'magenta'],
    contour_lws=[0.7, 0.7, 0.7],
    contour_linestyles=[':', '-.', '--'],
    tglon=st.tglon, tglat=st.tglat, topo=st.topo
)


fig1.tight_layout(rect=[0, 0, 1, 1])
fig1.suptitle(
    f'MCA Mode {mode} — {var_1_name}\n{sector_name} ({suffix})',
    fontsize=12, fontweight='bold',
    y=0.91   # lower this value to bring title down, e.g. try 0.93–0.97
)

if save_figs:
    path1 = os.path.join(
        save_dir,
        f'mca_stereo_mode{mode}_{var_1_name}_{sector_name}_{suffix}.png'
    )
    if os.path.exists(path1):
        print(f"Already exists, skipping: {path1}")
    else:
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        print(f"Saved: {path1}")

# stereoplot for var2 (comps2)
fig2, ax2, _ = st.spstere_plot_sector(
    llon, llat, comp2_m.values,
    vlims_2, cmap_2, cbar_units_2,
    lon_min=lon_min, lon_max=lon_max,
    lat_min=lat_min, lat_max=lat_max,
    contours=True,
    contour_levels=[-5000, -3000, -1000],
    contour_colors=None,
    contour_lws=None,
    contour_linestyles=None,
    tglon=st.tglon, tglat=st.tglat, topo=st.topo
)
fig2.tight_layout(rect=[0, 0, 1, 1])
fig2.suptitle(
    f'MCA Mode {mode} — {var_2_name}\n{sector_name} ({suffix})',
    fontsize=12, fontweight='bold', y=0.91
)

if save_figs:
    path2 = os.path.join(
        save_dir,
        f'mca_stereo_mode{mode}_{var_2_name}_{sector_name}_{suffix}.png'
    )
    if os.path.exists(path2):
        print(f"Already exists, skipping: {path2}")
    else:
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        print(f"Saved: {path2}")

# timeseries plot for scores 1 and 2 with correlation
s1 = scores1.sel(mode=mode)
s2 = scores2.sel(mode=mode)
s1_norm = (s1 - s1.mean()) / s1.std()
s2_norm = (s2 - s2.mean()) / s2.std()

fig3, ax3 = plt.subplots(figsize=(12, 4), dpi=150)

ax3.plot(s1_norm.time, s1_norm, label=var_1_name, linewidth=1.5)
ax3.plot(s2_norm.time, s2_norm, label=var_2_name, linewidth=1.5)
ax3.axhline(0, color='grey', linewidth=0.7, linestyle='--')

ax3.set_xlabel('Year')
ax3.set_ylabel('Normalised score')
ax3.set_title(
    f'MCA Mode {mode}: {var_1_name} vs {var_2_name} — {sector_name} ({suffix})',
    fontsize=12, fontweight='bold'
)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=30)

corr_txt = (f"Pearson  r = {c['r_pearson']:.2f}  (p = {c['p_pearson']:.3f})\n"
            f"Spearman r = {c['r_spearman']:.2f}  (p = {c['p_spearman']:.3f})")
ax3.text(0.98, 0.04, corr_txt, transform=ax3.transAxes, fontsize=10,
         va='bottom', ha='right',
         bbox=dict(facecolor='white', edgecolor='grey', alpha=0.85))

fig3.tight_layout()

if save_figs:
    path3 = os.path.join(
        save_dir,
        f'mca_scores_mode{mode}_{var_1_name}_{var_2_name}_{sector_name}_{suffix}.png'
    )
    if os.path.exists(path3):
        print(f"Already exists, skipping: {path3}")
    else:
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        print(f"Saved: {path3}")

plt.show()