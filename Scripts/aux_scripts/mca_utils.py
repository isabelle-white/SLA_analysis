"""
mca_utils.py
============
Reusable functions for Maximum Covariance Analysis (MCA).

Designed to be imported by any analysis notebook:

    import mca_utils as mca

    ds        = mca.load_preprocessed('preprocessed_seasonal.nc')
    xa_1, xa_2 = mca.prepare_fields(ds, 'zonal_ws', 'sla')
    model, scores1, scores2, comps1, comps2 = mca.run_mca(xa_1, xa_2, ds['seamask'])
    corrs     = mca.compute_score_correlations(scores1, scores2, n_modes=4)
    mca.correlate_with_index(scores1, scores2, index_da, 'zonal_ws', 'sla')
"""

import warnings
import numpy as np
import xarray as xr
import xeofs as xe
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── suppress known xeofs warnings about NaN/MultiIndex ───────────────────────
def _suppress_xeofs_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning,
                            message='Deleting a single level of a MultiIndex')
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message='Degrees of freedom <= 0')
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='Dataset has.*components')


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOADING PREPROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_preprocessed(filepath):
    """
    Load a preprocessed NetCDF file produced by preprocessing.py.

    Parameters
    ----------
    filepath : str — path to preprocessed_raw.nc or preprocessed_seasonal.nc

    Returns
    -------
    ds : xr.Dataset
        Contains all preprocessed variables + seamask.
        Global attribute 'seasonal_removed' records which file this is.

    Example
    -------
    ds_raw      = mca.load_preprocessed('data/preprocessed_raw.nc')
    ds_seasonal = mca.load_preprocessed('data/preprocessed_seasonal.nc')
    """
    ds = xr.open_dataset(filepath)
    seas = ds.attrs.get('seasonal_removed', 'unknown')
    print(f"Loaded: {filepath}")
    print(f"  seasonal_removed = {seas}")
    print(f"  variables: {list(ds.data_vars)}")
    print(f"  time: {ds.time.values[0]} → {ds.time.values[-1]}  ({len(ds.time)} steps)")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILDING XARRAY FIELDS FOR MCA
# ══════════════════════════════════════════════════════════════════════════════

def prepare_fields(ds, var_1_name, var_2_name, lon_slice=None, lat_slice=None):
    """
    Extract two variables from the preprocessed dataset, optionally subset to
    a spatial region, apply the sea mask, and return (time, lat, lon) DataArrays
    ready for xeofs MCA.

    Parameters
    ----------
    ds          : xr.Dataset — output of load_preprocessed()
    var_1_name  : str        — forcing variable name (e.g. 'zonal_ws')
    var_2_name  : str        — response variable name (e.g. 'sla')
    lon_slice   : tuple or None — (lon_min, lon_max) to subset, e.g. (0, 90)
    lat_slice   : tuple or None — (lat_min, lat_max) to subset, e.g. (-80, -50)

    Returns
    -------
    xa_1_masked, xa_2_masked : xr.DataArray, dims (time, latitude, longitude)

    Notes
    -----
    - The seamask is re-derived from the subset region if spatial slicing is used,
      so land masking is always correct for the domain.
    - For dateline-crossing sectors use prepare_fields_dateline() instead.

    Example
    -------
    # Full domain
    xa_1, xa_2 = mca.prepare_fields(ds, 'zonal_ws', 'sla')

    # Indian sector
    xa_1, xa_2 = mca.prepare_fields(ds, 'zonal_ws', 'sla',
                                     lon_slice=(20, 90), lat_slice=(-70, -40))
    """
    # ── select variables ───────────────────────────────────────────────────────
    v1 = ds[var_1_name]
    v2 = ds[var_2_name]

    # ── ensure (lon, lat, time) → transpose to (time, lat, lon) ───────────────
    v1 = _to_time_lat_lon(v1)
    v2 = _to_time_lat_lon(v2)

    # ── optional spatial subset ────────────────────────────────────────────────
    if lon_slice is not None:
        v1 = v1.sel(lon=slice(*lon_slice))
        v2 = v2.sel(lon=slice(*lon_slice))
    if lat_slice is not None:
        v1 = v1.sel(lat=slice(*lat_slice))
        v2 = v2.sel(lat=slice(*lat_slice))

    # ── build sea mask for the (possibly subsetted) domain ────────────────────
    seamask_2d = _build_seamask(ds['seamask'], v1)

    # ── apply mask ────────────────────────────────────────────────────────────
    xa_1_masked = v1.where(seamask_2d == 1)
    xa_2_masked = v2.where(seamask_2d == 1)

    print(f"Fields prepared: {var_1_name} + {var_2_name}")
    print(f"  shape: {xa_1_masked.shape}")

    return xa_1_masked, xa_2_masked


def prepare_fields_dateline(ds, var_1_name, var_2_name,
                             lon_min, lon_max, lat_min=-90, lat_max=-50):
    """
    Prepare fields for a sector that crosses the dateline (e.g. Ross Gyre:
    150°E → -80°E via 180°).

    Splits the longitude range at 180°, shifts the eastern chunk by +360°,
    and concatenates — matching the approach in largest_closed_contour.ipynb.

    Parameters
    ----------
    ds           : xr.Dataset
    var_1_name   : str
    var_2_name   : str
    lon_min      : float — western edge  (e.g. 150)
    lon_max      : float — eastern edge  (e.g. -80, i.e. 280 after shift)
    lat_min      : float
    lat_max      : float

    Returns
    -------
    xa_1_masked, xa_2_masked : xr.DataArray, dims (time, latitude, longitude)
      Longitudes run monotonically from lon_min to lon_max+360.
    """
    def _split_shift_concat(da):
        chunk_A = da.sel(lon=slice(lon_min, 180),
                         lat=slice(lat_min, lat_max))
        chunk_B = da.sel(lon=slice(-180, lon_max),
                         lat=slice(lat_min, lat_max))
        chunk_B = chunk_B.assign_coords(lon=chunk_B.lon.values + 360)
        return xr.concat([chunk_A, chunk_B], dim='lon')

    v1 = _to_time_lat_lon(ds[var_1_name])
    v2 = _to_time_lat_lon(ds[var_2_name])
    sm = ds['seamask']
    if 'time' in sm.dims:
        sm = sm.isel(time=0)

    v1 = _split_shift_concat(v1)
    v2 = _split_shift_concat(v2)
    sm = _split_shift_concat(sm)

    seamask_2d = _build_seamask(sm, v1)

    xa_1_masked = v1.where(seamask_2d == 1)
    xa_2_masked = v2.where(seamask_2d == 1)

    print(f"Dateline-aware fields prepared: {var_1_name} + {var_2_name}")
    print(f"  lon range: {float(v1.lon[0]):.1f} → {float(v1.lat[-1]):.1f}")
    print(f"  shape: {xa_1_masked.shape}")

    return xa_1_masked, xa_2_masked


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUNNING MCA
# ══════════════════════════════════════════════════════════════════════════════

def run_mca(xa_1, xa_2, n_modes=22, print_variance=True):
    """
    Fit an xeofs MCA model and return components and scores.

    Parameters
    ----------
    xa_1, xa_2    : xr.DataArray, dims (time, latitude, longitude)
    n_modes       : int — number of MCA modes to retain
    print_variance: bool — print squared covariance fraction per mode

    Returns
    -------
    model   : fitted xe.cross.MCA object
    scores1 : xr.DataArray — expansion coefficients for xa_1
    scores2 : xr.DataArray — expansion coefficients for xa_2
    comps1  : xr.DataArray — spatial patterns for xa_1
    comps2  : xr.DataArray — spatial patterns for xa_2

    Example
    -------
    model, s1, s2, c1, c2 = mca.run_mca(xa_1, xa_2)
    """
    _suppress_xeofs_warnings()

    print(f"Running MCA (n_modes={n_modes})...")
    model = xe.cross.MCA(n_modes=n_modes, standardize=True)
    model.fit(xa_1, xa_2, dim='time')

    comps1, comps2 = model.components()
    scores1, scores2 = model.scores()

    if print_variance:
        sq_cov = model.squared_covariance_fraction() * 100
        cum    = np.cumsum(sq_cov.values)
        print("  Squared covariance fraction (%):")
        for m in range(min(4, n_modes)):
            print(f"    Mode {m+1}: {sq_cov.isel(mode=m).values:.2f}%  "
                  f"(cumulative: {cum[m]:.2f}%)")

    return model, scores1, scores2, comps1, comps2


# ══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_score_correlations(scores1, scores2, n_modes=4):
    """
    Compute Pearson and Spearman correlations between paired MCA scores.

    Parameters
    ----------
    scores1, scores2 : xr.DataArray with 'mode' dimension
    n_modes          : int — how many modes to evaluate

    Returns
    -------
    corrs : list of dicts, one per mode:
        {'mode', 'r_pearson', 'p_pearson', 'r_spearman', 'p_spearman'}

    Example
    -------
    corrs = mca.compute_score_correlations(scores1, scores2)
    for c in corrs:
        print(f"Mode {c['mode']}: r_pearson={c['r_pearson']:.2f}")
    """
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
    return corrs


def correlate_with_index(scores1, scores2, index_da,
                          var_1_name, var_2_name,
                          n_modes=2,
                          time_start=None, time_end=None):
    """
    Correlate MCA scores with a climate index (e.g. SAM, SOI) and plot.

    Parameters
    ----------
    scores1, scores2 : xr.DataArray with 'mode' and 'time' dims
    index_da         : xr.DataArray with 'time' dim (output of ft.load_climate_index)
    var_1_name       : str — label for scores1
    var_2_name       : str — label for scores2
    n_modes          : int — number of modes to plot (default 2)
    time_start       : str or None — e.g. '2005-01' to subset common time axis
    time_end         : str or None — e.g. '2020-12'

    Returns
    -------
    fig : matplotlib Figure

    Example
    -------
    fig = mca.correlate_with_index(scores1, scores2, sam_da,
                                    'zonal_ws', 'sla',
                                    time_start='2005-01', time_end='2020-12')
    """
    common_time = np.intersect1d(index_da.time.values, scores1.time.values)

    if time_start is not None:
        common_time = common_time[common_time >= np.datetime64(time_start)]
    if time_end is not None:
        common_time = common_time[common_time <= np.datetime64(time_end)]

    index_norm = index_da.sel(time=common_time)
    index_norm = (index_norm - index_norm.mean()) / index_norm.std()

    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 5 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        m = i + 1
        s1 = scores1.sel(mode=m, time=common_time)
        s2 = scores2.sel(mode=m, time=common_time)
        s1_n = (s1 - s1.mean()) / s1.std()
        s2_n = (s2 - s2.mean()) / s2.std()

        r_12,  _ = scipy.stats.pearsonr(s1_n.values, s2_n.values)
        r_i1,  _ = scipy.stats.pearsonr(index_norm.values, s1_n.values)
        r_i2,  _ = scipy.stats.pearsonr(index_norm.values, s2_n.values)

        ax.plot(s1_n.time, s1_n,      label=f'MCA{m} {var_1_name}')
        ax.plot(s2_n.time, s2_n,      label=f'MCA{m} {var_2_name}')
        ax.plot(index_norm.time, index_norm, label=f'{index_da.name or "index"}',
                linestyle='--', color='black')

        ax.set_title(f'Mode {m}: {var_1_name} vs {var_2_name}')
        ax.set_ylabel('Normalised score')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

        corr_txt = (f'{var_1_name}–{var_2_name}  r={r_12:.2f}\n'
                    f'index–{var_1_name}  r={r_i1:.2f}\n'
                    f'index–{var_2_name}  r={r_i2:.2f}')
        ax.text(0.98, 0.03, corr_txt, transform=ax.transAxes, fontsize=10,
                va='bottom', ha='right',
                bbox=dict(facecolor='white', edgecolor='grey', alpha=0.8))

    axes[-1].set_xlabel('Year')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_mca_modes(scores1, scores2, comps1, comps2,
                   var_1_name, var_2_name,
                   corrs=None, n_modes=4):
    """
    Standard 3-column MCA plot: scores | comp1 | comp2, one row per mode.

    Parameters
    ----------
    scores1, scores2 : xr.DataArray
    comps1, comps2   : xr.DataArray
    var_1_name       : str
    var_2_name       : str
    corrs            : list of dicts from compute_score_correlations(), or None
    n_modes          : int

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=(12, 3 * n_modes))

    for i in range(n_modes):
        m = i + 1
        j = 3 * i + 1

        r_label = ''
        if corrs is not None and i < len(corrs):
            r_label = f"r = {corrs[i]['r_spearman']:.2f}"

        plt.subplot(n_modes, 3, j)
        scores1.sel(mode=m).plot(label=var_1_name)
        scores2.sel(mode=m).plot(label=var_2_name)
        plt.title(f'Mode {m}  {r_label}')
        plt.xlabel('Year');  plt.ylabel(f'PC{m}')
        plt.xticks(rotation=30);  plt.legend(fontsize=8)

        plt.subplot(n_modes, 3, j + 1)
        comps1.sel(mode=m).plot()
        plt.title(f'{var_1_name} — Mode {m}')

        plt.subplot(n_modes, 3, j + 2)
        comps2.sel(mode=m).plot()
        plt.title(f'{var_2_name} — Mode {m}')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_time_lat_lon(da):
    """Ensure DataArray has dims (time, latitude, longitude)."""
    if set(da.dims) == {'lon', 'lat', 'time'}:
        return da.transpose('time', 'lat', 'lon')
    if set(da.dims) == {'longitude', 'latitude', 'time'}:
        return da.transpose('time', 'latitude', 'longitude')

    return da


def _build_seamask(seamask_da, reference_da):
    """
    Return a 2-D (lat, lon) seamask aligned to reference_da's spatial grid.
    Handles the case where seamask_da may have been sliced or shifted.
    """
    # drop time if present
    if 'time' in seamask_da.dims:
        seamask_da = seamask_da.isel(time=0)

    # reindex to match the reference (handles +360 longitude shifts)
    sm = seamask_da.reindex(
        lat=reference_da.lat,
        lon=reference_da.lon,
        method='nearest',
        tolerance=0.5
    )
    return sm