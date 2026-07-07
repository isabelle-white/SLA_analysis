"""
E6b_across_wavelet_spectrum_gyre_mca_plots.py

MCA mode vs param

cross-wavelet transform/spectrum (XWT)

XWT  = cross power using watelet transform of TS 1 and complex conjugate of wavelet transfrom of TS2. Do both time series have a strong amplitude and are they in phase at each time and period (not normalised so one time series can dominate the signal/result)

wavelet coherence (WTC)

WTC = cross power as above. THEN smooths it and divides by smoothed individual power of each series - 0-1. local time/f resolved correlation corefficient . highlights areas of lower power but higher coherence between the two timeseries
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
from matplotlib.image import NonUniformImage
import matplotlib.dates as mdates



# Suppress only deprecation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
script_dir = workdir + 'Scripts/'
data_dir = workdir + 'Data/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/E_wavelet_analysis/'
mca_dir = data_dir + 'mca_processing/sectors/'
scores_comps_dir = mca_dir + 'scores_comps/'
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

    time1 = mdates.date2num(time1)
    time2 = mdates.date2num(time2)

    tnum = mdates.date2num(time1)
    n = len(tnum)

    # Compute wavelets
    mother = wavelet.Morlet(6)
    s0 = 2*dt
    dj = 1/12
    J = int(7/dj)

    W12, cross_coi, freq, signif = wavelet.xwt(y1, y2, dt, dj, s0, J, significance_level = 0.95, wavelet = mother,
                                             normalize=True )



    cross_power = (np.abs(W12))**2
    cross_sig = np.ones([1, n]) * signif[:, None]
    cross_sig = cross_power / cross_sig
    cross_period = 1 / freq

    # Calculate the wavelet coherence (WTC). The WTC finds regions in time
    # frequency space where the two time seris co-vary, but do not necessarily have
    # high power.
    WCT, aWCT, corr_coi, freq, sig = wavelet.wct(y1, y2, dt, dj, s0, J,
                                                 significance_level=0.95,
                                                 wavelet=mother, normalize=True)

    cor_sig = np.ones([1, n]) * sig[:, None]
    cor_sig = np.abs(WCT) / cor_sig  # Power is significant where ratio > 1
    cor_period = 1 / freq

    angle = 0.5 * np.pi - aWCT
    u, v = np.cos(angle), np.sin(angle)

    # -------------------------
    # PLOTTING (PyCWT style)
    # -------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6),
        sharey=True,
        constrained_layout=True
    )

    panels = [
        (f"Cross-Wavelet Power, {gyre_name}\n, {var_1_name}_{var_2_name}", cross_power, cross_period, cross_coi, cross_sig),
        (f"Wavelet Coherence, {gyre_name}\n, {var_1_name}_{var_2_name}", WCT, cor_period, corr_coi, cor_sig)
    ]

    for ax, (title, field, period, coi, sig95) in zip(axes, panels):

        T, P = np.meshgrid(tnum, period)

        # Main field
        im = ax.contourf(T, P, field, 60, cmap='jet')

        # Significance contour
        ax.contour(T, P, sig95, levels=[1], colors='k', linewidths=1.5)

        # COI
        ax.plot(tnum, coi, 'w--', linewidth=1.2)
        ax.fill_between(tnum, coi, period[-1], color='white', alpha=0.2)

        # Phase arrows (only for coherence)
        if "Wavelet Coherence" in title:
            step_t = 3
            step_p = 3
            ax.quiver(
                tnum[::step_t], period[::step_p],
                u[::step_p, ::step_t], v[::step_p, ::step_t],
                pivot='mid', color='k', scale=40
            )

        # Axis formatting
        ax.set_yscale('log')
        ax.set_ylim(period.min(), period.max())
        ax.invert_yaxis()

        ax.set_title(f'Mode {mode} – {title}', fontsize=18)
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Period (months)', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)

        # Period ticks
        period_ticks = [3, 6, 12, 24, 48, 96]
        ax.set_yticks(period_ticks)
        ax.set_yticklabels([str(p) for p in period_ticks])
        ax.yaxis.set_minor_locator(plt.NullLocator())

        # Time formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('Power / Coherence', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    plt.show()
    # save path
    savepath = f'{gyre_dir}{tag}_XWT_WTC_MCA_mode{mode}.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {savepath}")
