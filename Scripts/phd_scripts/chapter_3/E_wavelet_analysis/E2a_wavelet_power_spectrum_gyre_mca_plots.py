"""
E2_wavelet_power_spectrum_gyre_mca_plots.py

how one timeseries is distributed accross difference frequencies and how that changes over time
this produces a time-frequency map for one variable
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()
import sys
from scipy.stats import pearsonr
import warnings
import scipy
import pycwt as wavelet
from pycwt.helpers import find
import os



# Suppress only deprecation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
script_dir = workdir + 'Scripts/'
data_dir = workdir + 'Data/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/E_wavelet_analysis/'
mca_dir = data_dir + 'mca_processing/sectors/'
sys.path.append(auxscriptdir)
import aux_stereoplot as st
from geometry_izzyv1 import grad_sphere
import aux_func as ft

start_time = '2002-07-01'
end_time = '2024-12-01'
gyre_name = 'ross'
var_1_name = 'SLA'
var_2_name = 'OSC'
dt = 1
#
# gyre_dir = processing_dir + gyre_name + '/'
# gyre_fig_dir = fig_dir + gyre_name + '/timeseries/climate_indices/'
# os.makedirs(gyre_fig_dir, exist_ok=True)
# tag = f'{gyre_name}_{start_time}_{end_time}'

if gyre_name == 'ross':
    sector_dir = mca_dir + 'ross/'
    scores_comps_dir = sector_dir + 'scores_comps/'
elif gyre_name == 'weddell':
    sector_dir = mca_dir + 'weddell/'
    scores_comps_dir = sector_dir + 'scores_comps/'

tag = f'{gyre_name}_{start_time}_{end_time}_{var_1_name}_{var_2_name}'
gyre_dir = f'{fig_dir}{gyre_name}/{var_1_name}_{var_2_name}/'
os.makedirs(gyre_dir, exist_ok=True)

scores_comps_ds = xr.open_dataset(scores_comps_dir + f'{var_1_name}_{var_2_name}_scores_comps_{start_time}_{end_time}_{gyre_name}.nc')
scores1 = scores_comps_ds['scores_1']
scores2 = scores_comps_ds['scores_2']
comps1 = scores_comps_ds['comps_1']
comps2 = scores_comps_ds['comps_2']
sq_cov_percent = scores_comps_ds['sq_cov_frac']

for mode in range(1,6):
    ts1 = scores1.sel(mode=mode)
    ts2 = scores2.sel(mode=mode)

    # Prepare both series
    def prep(ts):
        y = ts.values.astype(float)
        y = y - np.nanmean(y)
        y = y / np.nanstd(y)
        return ts.time.values, y

    time1, y1 = prep(ts1)
    time2, y2 = prep(ts2)

    # Compute wavelets
    mother = wavelet.Morlet(6)
    s0 = 2*dt
    dj = 1/12
    J = 7/dj

    W1, scales1, freqs1, coi1, _, _ = wavelet.cwt(y1, dt, dj, s0, J, mother)
    W2, scales2, freqs2, coi2, _, _ = wavelet.cwt(y2, dt, dj, s0, J, mother)

    period1 = 1 / freqs1
    period2 = 1 / freqs2

    power1 = (np.abs(W1))**2
    power2 = (np.abs(W2))**2

    ## significance lines/testing
    alpha1, _, _ = wavelet.ar1(y1)  # lag-1 autocorrelation for red-noise background
    signif1, fft_theor1 = wavelet.significance(
        1.0, dt, scales1, 0, alpha1,
        significance_level=0.95, wavelet=mother
    )
    # signif1 has shape (n_scales,) — the significance threshold per scale
    sig95_1 = np.ones_like(power1) * signif1[:, None]
    sig95_1 = power1 / sig95_1  # power/sig ratio; contour at 1 marks the 95% line

    alpha2, _, _ = wavelet.ar1(y2)  # lag-2 autocorrelation for red-noise background
    signif2, fft_theor2 = wavelet.significance(
        1.0, dt, scales2, 0, alpha2,
        significance_level=0.95, wavelet=mother
    )
    # signif1 has shape (n_scales,) — the significance threshold per scale
    sig95_2 = np.ones_like(power2) * signif2[:, None]
    sig95_2 = power2 / sig95_2  # power/sig ratio; contour at 1 marks the 95% line

    # save path
    savepath = f'{gyre_dir}{tag}_wavelet_power_MCA_mode{mode}.png'

    # Plot side-by-side
    fig, axes = plt.subplots(
        1, 2, figsize=(16,6), sharey=True, constrained_layout=True
    )

    # panels = [
    #     (axes[0], power1, time1, scales1, coi1, 'SLA score'),
    #     (axes[1], power2, time2, scales2, coi2, 'OSC score')
    # ]

    panels = [
    (axes[0], power1, time1, period1, coi1, sig95_1, 'SLA score'),
    (axes[1], power2, time2, period2, coi2, sig95_2, 'OSC score')
]


    # for ax, power, time, scales, coi, sig95, label in panels:
    #     T, S = np.meshgrid(time, scales)
    #     im = ax.contourf(T, S, power, 60, cmap='jet')
    #
    #     # COI as dashed line (clean)
    #     ax.plot(time, coi, 'w--', linewidth=1.2)
    #     ax.fill_between(time, coi, scales[-1], color='white', alpha=0.2)
    #     ax.set_yscale('log')
    #     ax.set_title(f'Mode {mode} – {label}', fontsize=18)
    #     ax.set_xlabel('Time', fontsize=14)
    #     ax.set_ylabel('Period (months)', fontsize=14)
    #     ax.tick_params(axis='both', which='major', labelsize=14)
    #     ax.set_ylim(scales.min(), scales.max())
    #     ax.invert_yaxis()
    #     ax.contour(T, S, sig95, [1], colors='k', linewidths=1.5)
    #
    #     period_ticks = [3, 6, 12, 24, 48, 96]  # months: quarterly, biannual, annual, 2yr, 4yr, 8yr
    #     ax.set_yticks(period_ticks)
    #     ax.set_yticklabels([str(p) for p in period_ticks])
    #     ax.yaxis.set_minor_locator(plt.NullLocator())

    for ax, power, time, period, coi, sig95, label in panels:
        T, P = np.meshgrid(time, period)
        im = ax.contourf(T, P, power, 60, cmap='jet')

        # COI as dashed line (clean)
        ax.plot(time, coi, 'w--', linewidth=1.2)
        ax.fill_between(time, coi, period[-1], color='white', alpha=0.2)
        ax.set_yscale('log')
        ax.set_title(f'Mode {mode} – {label}', fontsize=18)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Period (months)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(period.min(), period.max())
        ax.invert_yaxis()
        ax.contour(T, P, sig95, [1], colors='k', linewidths=1.5)

        period_ticks = [3, 6, 12, 24, 48, 96]  # months: quarterly, biannual, annual, 2yr, 4yr, 8yr
        ax.set_yticks(period_ticks)
        ax.set_yticklabels([str(p) for p in period_ticks])
        ax.yaxis.set_minor_locator(plt.NullLocator())

    # ONE shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('Wavelet Power', labelpad = 15, fontsize=14)
    cbar.ax.tick_params(labelsize=14)



    plt.savefig(savepath, dpi=300, bbox_inches='tight')#
    plt.show()
    plt.close()

    print(f'saved to {savepath}')