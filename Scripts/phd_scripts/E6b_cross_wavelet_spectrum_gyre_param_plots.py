# """
# E6b_across_wavelet_spectrum_gyre_param_plots.py
#
# how one timeseries is distributed accross difference frequencies and how that changes over time
# this produces a time-frequency map for one variable
#
# cross-wavelet spectrum (XWT)
# """

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import warnings
import os
import matplotlib.dates as mdates
import pycwt as wavelet
from scipy.signal import detrend

warnings.filterwarnings("ignore", category=RuntimeWarning)

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
processing_dir = data_dir + 'C_gyre_processing/'
fig_dir = workdir + 'Figures/E_wavelet_analysis/'
mca_dir = data_dir + 'mca_processing/sectors/'

start_time = '2002-07-01'
end_time = '2024-12-01'
gyre_name = 'ross'
dt = 1

# Select sector
if gyre_name == 'ross':
    sector_dir = mca_dir + 'ross/'
    scores_comps_dir = sector_dir + 'scores_comps/'
elif gyre_name == 'weddell':
    sector_dir = mca_dir + 'weddell/'
    scores_comps_dir = sector_dir + 'scores_comps/'

# scores_comps_dir = sector_dir + 'scores_comps/'
tag = f'{gyre_name}_{start_time}_{end_time}'
gyre_fig_dir = f'{fig_dir}{gyre_name}/wavelet_cross/'
os.makedirs(gyre_fig_dir, exist_ok=True)

# -------------------------
# Load gyre parameters
# -------------------------
ts_params_ds = xr.open_dataset(processing_dir + gyre_name + f'/{tag}_params.nc')
ts_center_centroid_ds = xr.open_dataset(processing_dir + gyre_name + f'/{tag}_center_centroid.nc')
ts_strength_ds = xr.open_dataset(processing_dir + gyre_name + f'/{tag}_strength.nc')

time = ts_params_ds['time'].values

dot_mean = ts_params_ds['dot_mean'].values
area_km2 = ts_params_ds['area_km2'].values
contour_level = ts_params_ds['level'].values
strength_basic = ts_strength_ds['strength_basic'].values
strength_norm = ts_strength_ds['strength_normalised'].values
centre_lat = ts_center_centroid_ds['centre_lat'].values
centroid_lat = ts_center_centroid_ds['centroid_lat'].values

var_to_plot = [
    dot_mean, area_km2, contour_level,
    strength_basic, strength_norm,
    centre_lat, centroid_lat
]

name_to_plot = [
    'dot_mean', 'area_km2', 'contour_level',
    'strength_basic', 'strength_normalised',
    'centre_lat', 'centroid_lat'
]

# -------------------------
# Load MCA scores (SLA & OSC)
# -------------------------
scores_ds = xr.open_dataset(scores_comps_dir + f'SLA_OSC_scores_comps_{start_time}_{end_time}_{gyre_name}.nc')
scores1 = scores_ds['scores_1']   # SLA
scores2 = scores_ds['scores_2']   # OSC

# -------------------------
# Prep function
# -------------------------
def prep_series(y):
    y = np.asarray(y, dtype=float)
    nans = np.isnan(y)
    if nans.any():
        idx = np.arange(len(y))
        y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    y = detrend(y)
    y = y - np.nanmean(y)
    y = y / np.nanstd(y)
    return y

mother = wavelet.Morlet(6)
dj = 1/12
s0 = 2 * dt
J = int(7 / dj)

tnum = mdates.date2num(time)

# ============================================================
# LOOP: gyre parameter × MCA mode
# ============================================================
score_options = [(scores1, 'SLA'), (scores2, 'OSC')]

for var, varname in zip(var_to_plot, name_to_plot):

    y1 = prep_series(var)

    for score_da, score_name in score_options:
        for mode in range(1, 6):

            y2 = prep_series(score_da.sel(mode=mode).values)
            # -------------------------
            # XWT
            # -------------------------
            W12, cross_coi, freq, signif = wavelet.xwt(
                y1, y2, dt, dj, s0, J,
                significance_level=0.95,
                wavelet=mother,
                normalize=True
            )

            cross_power = np.abs(W12)**2
            cross_period = 1 / freq
            sig95_xwt = cross_power / (signif[:, None])

            # -------------------------
            # WTC
            # -------------------------
            WCT, aWCT, corr_coi, freq2, sig_coh = wavelet.wct(
                y1, y2, dt, dj, s0, J,
                significance_level=0.95,
                wavelet=mother,
                normalize=True
            )

            coh_period = 1 / freq2
            sig95_wtc = np.abs(WCT) / (sig_coh[:, None])

            angle = 0.5 * np.pi - aWCT
            u = np.cos(angle)
            v = np.sin(angle)

            # -------------------------
            # PLOTTING
            # -------------------------
            fig, axes = plt.subplots(
                1, 2, figsize=(16, 6),
                sharey=True,
                constrained_layout=True
            )

            panels = [
                (f"XWT: {gyre_name} \n {varname} vs {score_name} mode {mode}", cross_power, cross_period, cross_coi, sig95_xwt),
                (f"WTC: {gyre_name} \n {varname} vs {score_name} mode {mode}", WCT, coh_period, corr_coi, sig95_wtc)
            ]

            for ax, (title, field, period, coi, sig95) in zip(axes, panels):

                T, P = np.meshgrid(tnum, period)
                im = ax.contourf(T, P, field, 60, cmap='jet')

                ax.contour(T, P, sig95, levels=[1], colors='k', linewidths=1.5)

                ax.plot(tnum, coi, 'w--', linewidth=1.2)
                ax.fill_between(tnum, coi, period[-1], color='white', alpha=0.2)

                if "WTC" in title:
                    ax.quiver(
                        tnum[::3], period[::3],
                        u[::3, ::3], v[::3, ::3],
                        pivot='mid', color='k', scale=40
                    )

                ax.set_yscale('log')
                ax.set_ylim(period.min(), period.max())
                ax.invert_yaxis()

                ax.set_title(title, fontsize=18)
                ax.set_xlabel('Time', fontsize=14)
                ax.set_ylabel('Period (months)', fontsize=14)
                ax.tick_params(axis='both', labelsize=14)

                period_ticks = [3, 6, 12, 24, 48, 96]
                ax.set_yticks(period_ticks)
                ax.set_yticklabels([str(p) for p in period_ticks])
                ax.yaxis.set_minor_locator(plt.NullLocator())

                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

            cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
            cbar.set_label('Power / Coherence', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

            savepath = f"{gyre_fig_dir}{varname}_vs_{score_name}_mode{mode}_XWT_WTC.png"
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"Saved: {savepath}")
