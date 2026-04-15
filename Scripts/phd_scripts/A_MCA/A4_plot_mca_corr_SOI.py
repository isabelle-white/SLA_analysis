# A3_plot_mca_corr_SAM.py
"""
Load MCA scores + components, compute correlations,
and plot MCA modes + climate index.

Last modified: 14/04/2026
"""

import numpy as np
import xarray as xr
import scipy.stats
import matplotlib.pyplot as plt
import os
import sys

# paths
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
mca_dir = data_dir + 'mca_processing/'
clim_dir = data_dir + 'climate_indices/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'

sys.path.append(auxscriptdir)
import aux_func as ft

# choose variables + sector
var_1_name = 'total_ws'
var_2_name = 'sla'
sector_name = 'full'

deseas = True
detrended = True

if deseas:
    suffix = 'deseas'
else:
    suffix = 'with_seasonal_cycle'

if detrended:
    suffix += '_det'
else:
    suffix += '_raw'

file_name = f'{var_1_name}_{var_2_name}_{sector_name}_{suffix}.nc'
scores_file_name = 'scores_' + file_name
comps_file_name  = 'comps_'  + file_name

# load MCA results
scores_file = xr.open_dataset(mca_dir + scores_file_name)
comps_file  = xr.open_dataset(mca_dir + comps_file_name)

scores1 = scores_file['scores1']
scores2 = scores_file['scores2']
comps1  = comps_file['comps1']
comps2  = comps_file['comps2']

# number of modes to plot
n_modes = 2

# time window for index correlation
time_start = '2005-01'
time_end   = '2020-12'

# -------------------------------------------------------------------------
# LOAD CLIMATE INDEX (SAM example)
# -------------------------------------------------------------------------
# Expecting a 2-column text file: YYYY-MM, value
soi_da = ft.load_climate_index(clim_dir + 'SOI/soi.txt')

# -------------------------------------------------------------------------
# ALIGN TIME AXES
# -------------------------------------------------------------------------
common_time = np.intersect1d(soi_da.time.values, scores1.time.values)

if time_start is not None:
    common_time = common_time[common_time >= np.datetime64(time_start)]
if time_end is not None:
    common_time = common_time[common_time <= np.datetime64(time_end)]

# normalise index
index_da = soi_da.sel(time=common_time)
index_norm = (index_da - index_da.mean()) / index_da.std()

# -------------------------------------------------------------------------
# COMPUTE CORRELATIONS BETWEEN SCORES
# -------------------------------------------------------------------------
corrs = []
for m in range(1, n_modes + 1):
    x = scores1.sel(mode=m).sel(time=common_time).values
    y = scores2.sel(mode=m).sel(time=common_time).values
    r_p, p_p = scipy.stats.pearsonr(x, y)
    r_s, p_s = scipy.stats.spearmanr(x, y)
    corrs.append({
        'mode': m,
        'r_pearson': np.round(r_p, 3),
        'p_pearson': np.round(p_p, 3),
        'r_spearman': np.round(r_s, 3),
        'p_spearman': np.round(p_s, 3),
    })

# -------------------------------------------------------------------------
# PLOT MODES + INDEX
# -------------------------------------------------------------------------
fig, axes = plt.subplots(n_modes, 1, figsize=(12, 5 * n_modes), sharex=True)
if n_modes == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    m = i + 1

    s1 = scores1.sel(mode=m, time=common_time)
    s2 = scores2.sel(mode=m, time=common_time)
    s1_n = (s1 - s1.mean()) / s1.std()
    s2_n = (s2 - s2.mean()) / s2.std()

    r_12, _ = scipy.stats.pearsonr(s1_n.values, s2_n.values)
    r_i1, _ = scipy.stats.pearsonr(index_norm.values, s1_n.values)
    r_i2, _ = scipy.stats.pearsonr(index_norm.values, s2_n.values)

    ax.plot(s1_n.time, s1_n, label=f'MCA{m} {var_1_name}')
    ax.plot(s2_n.time, s2_n, label=f'MCA{m} {var_2_name}')
    ax.plot(index_norm.time, index_norm, '--', color='black', label=index_da.name)

    ax.set_title(f'Mode {m}: {var_1_name} vs {var_2_name}')
    ax.set_ylabel('Normalised score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    corr_txt = (
        f'{var_1_name}–{var_2_name}  r={r_12:.2f}\n'
        f'index–{var_1_name}  r={r_i1:.2f}\n'
        f'index–{var_2_name}  r={r_i2:.2f}'
    )
    ax.text(0.98, 0.03, corr_txt, transform=ax.transAxes,
            fontsize=10, va='bottom', ha='right',
            bbox=dict(facecolor='white', edgecolor='grey', alpha=0.8))

axes[-1].set_xlabel('Year')
fig.tight_layout()
plt.show()