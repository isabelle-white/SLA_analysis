"""
This script uses the functions defined in : /Users/iw2g24/PycharmProjects/SLA_analysis/Scripts/aux_scripts/MCA_preprocessing_func.py
to calculate and save the stresses for the full time series using my data product.

Can optionally include deseas (removal of seasonal cycle) and detrend (removal of linear trend - always do this one)

Last modified: 14/04/2026
"""

import numpy as np
import xarray as xr
import os
import sys

workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
data_dir = workdir + 'Data/'
script_dir = workdir + 'Scripts/'
auxscriptdir = script_dir + 'aux_scripts/'
clim_dir = data_dir + 'climate_indices/'

sys.path.append(auxscriptdir)
import mca_preprocessing_func as mca_func


seasonal_removal = False     # remove seasonal cycle/monthly climatologies
apply_detrend    = True     # detrend after seasonal removal


# Load data
ds_dot_original = xr.open_dataset(data_dir + 'dot_all_30bmedian_egm2008_sig3.nc')
ds_era5 = xr.open_dataset(data_dir + 'ERA5/era5_regridded_2002_2024_monthly.nc')
ds_sic  = xr.open_dataset(data_dir + 'NOAA_SIC_monthly/sic_regridded_2002_2024_monthly.nc')
ds_sid  = xr.open_dataset(data_dir + 'NASA_SID_weekly/sid_regridded_2002_2024_monthly.nc')

good_months = ds_dot_original['bad_month_flag'] == 0
ds_dot  = ds_dot_original.isel(time=good_months)
ds_era5 = ds_era5.isel(time=good_months)
ds_sic  = ds_sic.isel(time=good_months)
ds_sid  = ds_sid.isel(time=good_months)

print(f"{int(good_months.sum())} good months retained "
      f"({len(good_months) - int(good_months.sum())} flagged bad months removed)")


# extract data as arrays
dot  = ds_dot.dot.values
sla  = ds_dot.sla.values
time = ds_dot.time.values
lon  = ds_dot.longitude.values
lat  = ds_dot.latitude.values
llon, llat = np.meshgrid(lon, lat)


# calculate stresses using the function in MCA_preprossecing_func.py
tau_x_ws, tau_y_ws, total_ws, u10_air, v10_air, U_air = mca_func.compute_wind_stress(ds_era5)
wsc = mca_func.compute_wind_stress_curl(tau_x_ws, tau_y_ws, dot, llon, llat)
oss_x, oss_y, total_oss, sic = mca_func.compute_ocean_surface_stress(ds_sid, ds_sic, u10_air, v10_air, U_air)
osc = mca_func.compute_ocean_stress_curl(oss_x, oss_y, dot, llon, llat)

seamask = dot[:, :, 0] / dot[:, :, 0]
seamask[seamask == 0] = np.nan

# create a dictionary for the fields
arrays = {
    'dot': dot,
    'sla': sla,
    'sic': sic,
    'total_ws': total_ws,
    'wsc': wsc,
    'total_oss': total_oss,
    'osc': osc,
    'zonal_ws': tau_x_ws,
    'zonal_oss': oss_x,
}


#optional seasonal removal as defined above
if seasonal_removal:
    print("Removing seasonal cycle...")
    arrays_to_process = mca_func.remove_seasonal_cycle(arrays, time, lat, lon)
    suffix = "deseas"
else:
    print("Keeping seasonal cycle...")
    arrays_to_process = arrays
    suffix = "with_seasonal_cycle"


# optional detrending  - always set to True
if apply_detrend:
    print("Detrending fields...")
    arrays_final = mca_func.detrend_all(arrays_to_process, time, lat, lon)
    suffix += "_det"
else:
    print("Skipping detrending...")
    arrays_final = arrays_to_process
    suffix += "_raw"


# save
outfile = data_dir + 'mca_processing/' + f"mca_preprocessed_{suffix}.nc"
print(f"Saving to {outfile}")

ds_out = xr.Dataset({
    name: xr.DataArray(arr, dims=['lon','lat','time'],
                       coords={'lon': lon, 'lat': lat, 'time': time})
    for name, arr in arrays_final.items()
})

ds_out['seamask'] = xr.DataArray(
    seamask,
    dims=['lon','lat'],
    coords={'lon': lon, 'lat': lat}
)

ds_out.attrs['seasonal_removed'] = 'True' if seasonal_removal else 'False'
ds_out.attrs['detrended'] = 'True' if apply_detrend else 'False'
ds_out.attrs['created_by'] = 'run_preprocessing.py'
ds_out.attrs['description'] = (
    'Preprocessed DOT/SLA/stress fields for MCA. '
    'Seasonal cycle removed.' if seasonal_removal else
    'Preprocessed DOT/SLA/stress fields for MCA. Seasonal cycle retained.'
)
ds_out.attrs['notes'] = (
    'Linear trend removed from all variables.' if apply_detrend else
    'No detrending applied.'
)

if os.path.exists(outfile):
    print(f"Files already exist: {outfile}")
else:
    print(f"Writing to {outfile}")
    ds_out.to_netcdf(outfile)

print("Done.")