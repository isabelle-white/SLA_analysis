"""
MCA_preprocessing_func.py

- has functions to compute all derived fields (wind stress, WSC, OSS, OSC),

"""

import argparse
import numpy as np
import xarray as xr
import sys
import os

# ── paths ──────────────────────────────────────────────────────────────────────
workdir    = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir   = workdir + 'Data/'
script_dir = workdir + 'Scripts/'

sys.path.append(script_dir + 'aux_scripts/')
from geometry_izzyv1 import grad_sphere
import aux_func as ft

# set physical constants
rho_a   = 1.225       # air density          [kg m⁻³]
rho_sw  = 1027        # seawater density     [kg m⁻³]
cd_aw   = 1.25e-3     # air-water drag coeff
cd_iw   = 5.5e-3      # ice-water drag coeff

# # set variable list
# var_names = [
#     'dot', 'sla', 'sic',
#     'total_ws', 'wsc',
#     'total_oss', 'osc',
#     'zonal_ws', 'zonal_oss',
# ]


# # --------
# # 1. DATA LOADING
# # --------
#
# def load_datasets():
#     """Load raw datasets and apply bad-month mask. Returns dict of datasets."""
#     print("Loading datasets...")
#
#     ds_dot_original = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
#     ds_era5 = xr.open_dataset(data_dir + 'ERA5/era5_regridded_2002_2024_monthly.nc')
#     ds_sic  = xr.open_dataset(data_dir + 'NOAA_SIC_monthly/sic_regridded_2002_2024_monthly.nc')
#     ds_sid  = xr.open_dataset(data_dir + 'NASA_SID_weekly/sid_regridded_2002_2024_monthly.nc')
#
#     # apply bad-month mask consistently across all datasets
#     good_months = ds_dot_original['bad_month_flag'] == 0
#     ds_dot  = ds_dot_original.isel(time=good_months)
#     ds_era5 = ds_era5.isel(time=good_months)
#     ds_sic  = ds_sic.isel(time=good_months)
#     ds_sid  = ds_sid.isel(time=good_months)
#
#     print(f"  {int(good_months.sum())} good months retained "
#           f"({len(good_months) - int(good_months.sum())} flagged bad months removed)")
#
#     return ds_dot, ds_era5, ds_sic, ds_sid


# -----
# STRESS COMPUTATIONS
# ------

def compute_wind_stress(ds_era5):
    """
    Compute wind stress components and magnitude from ERA5 10-m winds.

    Returns
    -------
    tau_x_ws, tau_y_ws, total_ws : np.ndarray, shape (lon, lat, time)
    u10_air, v10_air, U_air      : wind components/magnitude for re-use in OSS
    """
    print("Computing wind stress...")

    u10_air = ds_era5['u10'].values
    v10_air = ds_era5['v10'].values
    U_air   = np.sqrt(u10_air**2 + v10_air**2)

    tau_x_ws = rho_a * cd_aw * U_air * u10_air
    tau_y_ws = rho_a * cd_aw * U_air * v10_air
    total_ws = rho_a * cd_aw * U_air**2

    return tau_x_ws, tau_y_ws, total_ws, u10_air, v10_air, U_air


def compute_wind_stress_curl(tau_x_ws, tau_y_ws, dot, llon, llat):
    """
    Compute wind stress curl (WSC) at every timestep.

    Parameters
    ----------
    tau_x_ws, tau_y_ws : (lon, lat, time)
    dot                : (lon, lat, time)  — used only for shape
    llon, llat         : meshgrid arrays   — (lat, lon)

    Returns
    -------
    wsc : np.ndarray, shape (lon, lat, time)
    """
    print("Computing wind stress curl...")

    n_lon, n_lat, n_time = dot.shape
    wsc = np.zeros((n_time, n_lat, n_lon))

    for t in range(n_time):
        tau_x_t = tau_x_ws[:, :, t].T   # → (lat, lon)
        tau_y_t = tau_y_ws[:, :, t].T
        dtauy_dx, _      = grad_sphere(tau_y_t, llon, llat)
        _,        dtaux_dy = grad_sphere(tau_x_t, llon, llat)
        wsc[t] = dtauy_dx - dtaux_dy

    return wsc.transpose(2, 1, 0)   # → (lon, lat, time)


def compute_ocean_surface_stress(ds_sid, ds_sic, u10_air, v10_air, U_air):
    """
    Compute ocean surface stress (OSS) components and magnitude.

    Ice-free regions use air-ocean drag; ice-covered regions use ice-ocean drag,
    weighted by SIC as α.

    Returns
    -------
    oss_x, oss_y, total_oss : np.ndarray, shape (lon, lat, time)
    """
    print("Computing ocean surface stress...")

    sic    = ds_sic['cdr_seaice_conc_monthly'].values

    # ice drift: cm/s → m/s; NaN drift treated as zero (stationary/absent ice)
    drift_u       = ds_sid.u.values * 1e-2
    drift_v       = ds_sid.v.values * 1e-2
    drift_u_clean = np.nan_to_num(drift_u, nan=0.0)
    drift_v_clean = np.nan_to_num(drift_v, nan=0.0)
    U_drift       = np.sqrt(drift_u_clean**2 + drift_v_clean**2)

    # component stresses
    tau_iw_x = rho_sw * cd_iw * U_drift * drift_u_clean
    tau_iw_y = rho_sw * cd_iw * U_drift * drift_v_clean
    tau_aw_x = rho_a  * cd_aw * U_air   * u10_air
    tau_aw_y = rho_a  * cd_aw * U_air   * v10_air

    # SIC-weighted blend  (NaNs in sic (where land/missing) are handled by sea mask later)
    oss_x = sic * tau_iw_x + (1 - sic) * tau_aw_x
    oss_y = sic * tau_iw_y + (1 - sic) * tau_aw_y

    tau_iw    = rho_sw * cd_iw * np.abs(U_drift) * U_drift
    tau_aw    = rho_a  * cd_aw * np.abs(U_air)   * U_air
    total_oss = sic * tau_iw + (1 - sic) * tau_aw

    return oss_x, oss_y, total_oss, sic


def compute_ocean_stress_curl(oss_x, oss_y, dot, llon, llat):
    """
    Compute ocean surface stress curl (OSC) at every timestep.

    Returns
    -------
    osc : np.ndarray, shape (lon, lat, time)
    """
    print("Computing ocean surface stress curl...")

    n_lon, n_lat, n_time = dot.shape
    osc = np.zeros((n_time, n_lat, n_lon))

    for t in range(n_time):
        oss_x_t = oss_x[:, :, t].T
        oss_y_t = oss_y[:, :, t].T
        doss_y_dx, _         = grad_sphere(oss_y_t, llon, llat)
        _,         doss_x_dy = grad_sphere(oss_x_t, llon, llat)
        osc[t] = doss_y_dx - doss_x_dy

    return osc.transpose(2, 1, 0)   # → (lon, lat, time)


# -----
# SEASONAL REMOVAL
# -----

def remove_seasonal_cycle(arrays, time_coords, lat, lon):
    """
    Remove the monthly climatological mean from every variable in `arrays`.

    Parameters
    ----------
    arrays      : dict {name: np.ndarray}  shape (lon, lat, time) or (time, lat, lon)
    time_coords : array of np.datetime64
    lat, lon    : 1-D coordinate arrays

    Returns
    -------
    arrays_anom : dict {name: np.ndarray} — seasonal anomalies, shape (lon, lat, time)
    """
    print("Removing seasonal cycle...")

    n_time = len(time_coords)
    n_lat  = len(lat)
    n_lon  = len(lon)
    arrays_anom = {}

    for name, arr in arrays.items():
        if arr.shape == (n_lon, n_lat, n_time):
            da = xr.DataArray(arr, dims=['longitude', 'latitude', 'time'],
                              coords={'time': time_coords, 'latitude': lat, 'longitude': lon})
        elif arr.shape == (n_time, n_lat, n_lon):
            da = xr.DataArray(arr, dims=['time', 'latitude', 'longitude'],
                              coords={'time': time_coords, 'latitude': lat, 'longitude': lon})
        else:
            raise ValueError(f"Unexpected shape for '{name}': {arr.shape}")

        anom = (da.groupby('time.month') - da.groupby('time.month').mean('time'))
        arrays_anom[name] = anom.transpose('longitude', 'latitude', 'time').values

    print("  Done.")
    return arrays_anom


# -------
# DETRENDING
# --------

def detrend_all(arrays, time_coords, lat, lon):
    """
    Remove linear trend + mean from every variable using ft.detrend3d.

    Parameters
    ----------
    arrays      : dict {name: np.ndarray}  shape (lon, lat, time)
    time_coords : array of np.datetime64
    lat, lon    : 1-D coordinate arrays

    Returns
    -------
    detrended : dict {name: np.ndarray}
    """
    print("Detrending...")

    detrended = {}
    for name, arr in arrays.items():
        da = xr.DataArray(arr, dims=['longitude', 'latitude', 'time'],
                          coords={'time': time_coords, 'latitude': lat, 'longitude': lon})
        det, _, _ = ft.detrend3d(da)
        detrended[name] = det
        print(f"  {name} detrended")

    return detrended
#
# # -------
# # 5. SAVE TO NETCDF
# # -------
#
# def save_to_netcdf(detrended, seamask, time_coords, lat, lon, outpath, seasonal_removed):
#     """
#     Save all detrended fields + seamask + coordinates to a NetCDF file.
#
#     Parameters
#     ----------
#     detrended        : dict {name: np.ndarray}  shape (lon, lat, time)
#     seamask          : np.ndarray               shape (lon, lat)
#     time_coords      : array of np.datetime64
#     lat, lon         : 1-D coordinate arrays
#     outpath          : str  — full output file path
#     seasonal_removed : bool — recorded as a global attribute
#     """
#     print(f"Saving to {outpath} ...")
#
#     data_vars = {}
#     for name, arr in detrended.items():
#         data_vars[name] = xr.DataArray(
#             arr,
#             dims=['longitude', 'latitude', 'time'],
#             coords={'longitude': lon, 'latitude': lat, 'time': time_coords},
#             attrs={'long_name': name, 'seasonal_removed': str(seasonal_removed)}
#         )
#
#     data_vars['seamask'] = xr.DataArray(
#         seamask,
#         dims=['longitude', 'latitude'],
#         coords={'longitude': lon, 'latitude': lat},
#         attrs={'long_name': 'sea mask (1=ocean, NaN=land)'}
#     )
#
#     ds_out = xr.Dataset(
#         data_vars,
#         attrs={
#             'description'      : 'Preprocessed MCA fields',
#             'seasonal_removed' : str(seasonal_removed),
#             'variables'        : ', '.join(detrended.keys()),
#         }
#     )
#
#     ds_out.to_netcdf(outpath)
#     print(f"  Saved: {outpath}")
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # 6. MAIN
# # ══════════════════════════════════════════════════════════════════════════════
#
# def main(save_raw=True, save_seasonal=True, outdir=None):
#     """
#     Main preprocessing pipeline.
#
#     Parameters
#     ----------
#     save_raw      : bool — save preprocessed_raw.nc      (seasonal cycle kept)
#     save_seasonal : bool — save preprocessed_seasonal_removed.nc (seasonal cycle removed)
#     outdir        : str  — output directory (defaults to data_dir)
#     """
#     out = outdir or data_dir
#
#     # ── load ──────────────────────────────────────────────────────────────────
#     ds_dot, ds_era5, ds_sic, ds_sid = load_datasets()
#
#     dot         = ds_dot.dot.values
#     sla         = ds_dot.sla.values
#     time_coords = ds_dot.time.values
#     lon         = ds_dot.longitude.values
#     lat         = ds_dot.latitude.values
#     llon, llat  = np.meshgrid(lon, lat)
#
#     # sea mask from first valid time slice of DOT
#     seamask = dot[:, :, 0].copy() / dot[:, :, 0]
#     seamask[seamask == 0] = np.nan
#
#     # ── compute stresses ──────────────────────────────────────────────────────
#     tau_x_ws, tau_y_ws, total_ws, u10_air, v10_air, U_air = compute_wind_stress(ds_era5)
#     wsc       = compute_wind_stress_curl(tau_x_ws, tau_y_ws, dot, llon, llat)
#     oss_x, oss_y, total_oss, sic = compute_ocean_surface_stress(ds_sid, ds_sic, u10_air, v10_air, U_air)
#     osc       = compute_ocean_stress_curl(oss_x, oss_y, dot, llon, llat)
#
#     # ── build arrays dict ─────────────────────────────────────────────────────
#     arrays = {
#         'dot'       : dot,
#         'sla'       : sla,
#         'sic'       : sic,
#         'total_ws'  : total_ws,
#         'wsc'       : wsc,
#         'total_oss' : total_oss,
#         'osc'       : osc,
#         'zonal_ws'  : tau_x_ws,
#         'zonal_oss' : oss_x,
#     }
#
#     # ── raw branch (seasonal cycle kept) ──────────────────────────────────────
#     if save_raw:
#         print("\n── RAW branch (seasonal cycle kept) ──")
#         detrended_raw = detrend_all(arrays, time_coords, lat, lon)
#         save_to_netcdf(
#             detrended_raw, seamask, time_coords, lat, lon,
#             outpath=os.path.join(out, 'preprocessed_raw.nc'),
#             seasonal_removed=False
#         )
#
#     # ── seasonal branch (seasonal cycle removed) ──────────────────────────────
#     if save_seasonal:
#         print("\n── SEASONAL branch (seasonal cycle removed) ──")
#         arrays_anom       = remove_seasonal_cycle(arrays, time_coords, lat, lon)
#         detrended_seasonal = detrend_all(arrays_anom, time_coords, lat, lon)
#         save_to_netcdf(
#             detrended_seasonal, seamask, time_coords, lat, lon,
#             outpath=os.path.join(out, 'preprocessed_seasonal_removed.nc'),
#             seasonal_removed=True
#         )
#
#     print("\nPreprocessing complete.")
#
#
# # ── CLI ───────────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Preprocess MCA fields.')
#     parser.add_argument('--no-raw',      action='store_true', help='Skip saving raw (seasonal kept) file')
#     parser.add_argument('--no-seasonal', action='store_true', help='Skip saving seasonal-anomaly file')
#     parser.add_argument('--outdir',      type=str, default=None, help='Output directory')
#     args = parser.parse_args()
#
#     main(
#         save_raw      = not args.no_raw,
#         save_seasonal = not args.no_seasonal,
#         outdir        = args.outdir,
#     )