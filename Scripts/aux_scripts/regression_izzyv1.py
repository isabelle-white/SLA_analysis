def linregress_3D(y_array):
    import numpy as np
    # y_array is a 3-D array formatted like (time,lon,lat)
    # The purpose of this function is to do linear regression using time series of data over each (lon,lat) grid box with consideration of ignoring np.nan
    # Construct x_array indicating time indexes of y_array, namely the independent variable.
    x_array=np.empty(y_array.shape)
    for i in range(y_array.shape[0]): x_array[i,:,:]=i+1 # This would be fine if time series is not too long. Or we can use i+yr (e.g. 2019).
    x_array[np.isnan(y_array)]=np.nan
    # Compute the number of non-nan over each (lon,lat) grid box.
    n=np.sum(~np.isnan(x_array),axis=0)
    # Compute mean and standard deviation of time series of x_array and y_array over each (lon,lat) grid box.
    x_mean=np.nanmean(x_array,axis=0)
    y_mean=np.nanmean(y_array,axis=0)
    x_std=np.nanstd(x_array,axis=0)
    y_std=np.nanstd(y_array,axis=0)
    # Compute co-variance between time series of x_array and y_array over each (lon,lat) grid box.
    cov=np.nansum((x_array-x_mean)*(y_array-y_mean),axis=0)/n
    # Compute correlation coefficients between time series of x_array and y_array over each (lon,lat) grid box.
    cor=cov/(x_std*y_std)
    # Compute slope between time series of x_array and y_array over each (lon,lat) grid box.
    slope=cov/(x_std**2)
    # Compute intercept between time series of x_array and y_array over each (lon,lat) grid box.
    intercept=y_mean-x_mean*slope
    # Compute tstats, stderr, and p_val between time series of x_array and y_array over each (lon,lat) grid box.
    tstats=cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr=slope/tstats
    from scipy.stats import t
    p_val=t.sf(tstats,n-2)*2
    # Compute r_square and rmse between time series of x_array and y_array over each (lon,lat) grid box.
    # r_square also equals to cor**2 in 1-variable lineare regression analysis, which can be used for checking.
    r_square=np.nansum((slope*x_array+intercept-y_mean)**2,axis=0)/np.nansum((y_array-y_mean)**2,axis=0)
    rmse=np.sqrt(np.nansum((y_array-slope*x_array-intercept)**2,axis=0)/n)
    # Do further filteration if needed (e.g. We stipulate at least 3 data records are needed to do regression analysis) and return values
    n=n*1.0 # convert n from integer to float to enable later use of np.nan
    n[n<3]=np.nan
    slope[np.isnan(n)]=np.nan
    intercept[np.isnan(n)]=np.nan
    p_val[np.isnan(n)]=np.nan
    r_square[np.isnan(n)]=np.nan
    rmse[np.isnan(n)]=np.nan
    return n,slope,intercept,p_val,r_square,rmse

def linregress_3D_spatial_time(y_array, x_series):
    """
    Regresses y_array (time, y, x) on x_series (time,)
    """
    import numpy as np
    from scipy.stats import t

    # Broadcast x_series to match shape of y_array
    x_array = np.broadcast_to(x_series[:, None, None], y_array.shape).copy()
    x_array[np.isnan(y_array)] = np.nan  # match masking

    n = np.sum(~np.isnan(x_array), axis=0)

    x_mean = np.nanmean(x_array, axis=0)
    y_mean = np.nanmean(y_array, axis=0)

    x_std = np.nanstd(x_array, axis=0)
    y_std = np.nanstd(y_array, axis=0)

    cov = np.nansum((x_array - x_mean) * (y_array - y_mean), axis=0) / n
    cor = cov / (x_std * y_std)
    slope = cov / (x_std**2)
    intercept = y_mean - slope * x_mean

    tstats = cor * np.sqrt(n - 2) / np.sqrt(1 - cor**2)
    stderr = slope / tstats
    p_val = t.sf((tstats), n - 2) * 2

    r_square = np.nansum((slope * x_array + intercept - y_mean)**2, axis=0) / \
               np.nansum((y_array - y_mean)**2, axis=0)
    rmse = np.sqrt(np.nansum((y_array - (slope * x_array + intercept))**2, axis=0) / n)

    # filter for insufficient data
    n = n * 1.0
    n[n < 3] = np.nan
    slope[np.isnan(n)] = np.nan
    intercept[np.isnan(n)] = np.nan
    p_val[np.isnan(n)] = np.nan
    r_square[np.isnan(n)] = np.nan
    rmse[np.isnan(n)] = np.nan

    return n, slope, intercept, p_val, r_square, rmse

def linregress_1D_on_3D(x_array, y_series):
    """
    Regresses y_series (1D, e.g., ice melt over time)
    on x_array (3D: time, y, x, e.g., ocean variable)
    Returns regression outputs for each grid point.
    """
    import numpy as np
    from scipy.stats import t

    # Broadcast y_series to 3D to match x_array
    y_array = np.broadcast_to(y_series[:, None, None], x_array.shape).copy()
    x_array[np.isnan(y_array)] = np.nan  # match masking

    n = np.sum(~np.isnan(x_array), axis=0)

    x_mean = np.nanmean(x_array, axis=0)
    y_mean = np.nanmean(y_array, axis=0)

    x_std = np.nanstd(x_array, axis=0)
    y_std = np.nanstd(y_array, axis=0)

    cov = np.nansum((x_array - x_mean) * (y_array - y_mean), axis=0) / n
    cor = cov / (x_std * y_std)
    slope = cov / (x_std**2)
    intercept = y_mean - slope * x_mean

    tstats = cor * np.sqrt(n - 2) / np.sqrt(1 - cor**2)
    stderr = slope / tstats
    p_val = t.sf((tstats), n - 2) * 2

    r_square = np.nansum((slope * x_array + intercept - y_mean)**2, axis=0) / \
               np.nansum((y_array - y_mean)**2, axis=0)
    rmse = np.sqrt(np.nansum((y_array - (slope * x_array + intercept))**2, axis=0) / n)

    # mask low data count
    n = n * 1.0
    n[n < 3] = np.nan
    slope[np.isnan(n)] = np.nan
    intercept[np.isnan(n)] = np.nan
    p_val[np.isnan(n)] = np.nan
    r_square[np.isnan(n)] = np.nan
    rmse[np.isnan(n)] = np.nan

    return n, slope, intercept, p_val, r_square, rmse
