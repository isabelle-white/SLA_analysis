# A2_plot_corr_mca.py
"""
This script loads MCA scores + components, computes correlations,
and plots MCA modes.

Last modified: 14/04/2026
"""

import numpy as np
import xarray as xr
import os
import sys
import warnings
import scipy.stats
import matplotlib.pyplot as plt

# paths
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
mca_dir = data_dir + 'mca_processing/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
sys.path.append(auxscriptdir)

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

# number of modes
n_modes = scores1.sizes['mode']

# compute correlations locally
corrs = []
for m in range(1, n_modes + 1):
    x = scores1.sel(mode=m).values
    y = scores2.sel(mode=m).values
    r_p, p_p = scipy.stats.pearsonr(x, y)
    r_s, p_s = scipy.stats.spearmanr(x, y)
    corrs.append({
        'mode'       : m,
        'r_pearson'  : np.round(r_p, 3),
        'p_pearson'  : np.round(p_p, 3),
        'r_spearman' : np.round(r_s, 3),
        'p_spearman' : np.round(p_s, 3),
    })

# plot MCA modes
fig = plt.figure(figsize=(12, 3 * n_modes))

for i in range(0,4):
    m = i + 1
    j = 3 * i + 1

    r_label = f"r = {corrs[i]['r_spearman']:.2f}"

    # scores
    plt.subplot(n_modes, 3, j)
    scores1.sel(mode=m).plot(label=var_1_name)
    scores2.sel(mode=m).plot(label=var_2_name)
    plt.title(f'Mode {m}  {r_label}')
    plt.xlabel('Year')
    plt.ylabel(f'PC{m}')
    plt.xticks(rotation=30)
    plt.legend(fontsize=8)

    # comps1
    plt.subplot(n_modes, 3, j + 1)
    comps1.sel(mode=m).plot(add_colorbar=True, cbar_kwargs={'label': ''})
    plt.title(f'{var_1_name}')

    # comps2
    plt.subplot(n_modes, 3, j + 2)
    comps2.sel(mode=m).plot(add_colorbar=True, cbar_kwargs={'label': ''})
    plt.title(f'{var_2_name}')

fig.tight_layout()
plt.show()
