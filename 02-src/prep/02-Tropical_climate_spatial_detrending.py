# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: climate
#     language: python
#     name: python3
# ---

# %%
"""
We here detrended spatial tropical climate
"""

# Load packages
import xarray as xr
import numpy as np
import pandas as pd
import proplot as pplt
import statsmodels.api as sm

import dask
import os

# %%
# project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'

# change directory
os.chdir(proj_path)
print('We are locating in', os.getcwd())

# %%
start_date = pd.Timestamp('1958-03-01')
end_date   = pd.Timestamp('2019-07-31')


# %%
def process_single_timeseries(ts, method='sum', lowess_frac=4/5):
    """
    Detrend a single time series (monthly data) using seasonal cycle removal, annual aggregation, and LOWESS smoothing.
    
    Parameters:
    -----------
    ts : array-like
    method : str, 'sum' or 'mean' 
    lowess_frac : float
    Returns:
    --------
    numpy.ndarray
    """
    
    # 1d array
    ts = np.asarray(ts).flatten()
    
    if len(ts) == 0 or np.all(np.isnan(ts)):
        return np.full_like(ts, np.nan)
    
    n_months = len(ts)
    
    months = np.tile(np.arange(1, 13), n_months // 12 + 1)[:n_months]
    
    # Step 1: climatological seasonal cycle removal
    mean_seasonal_cycle = np.zeros(12)
    for month in range(1, 13):
        month_mask = months == month
        month_data = ts[month_mask]
        if len(month_data) > 0:
            mean_seasonal_cycle[month-1] = np.nanmean(month_data)
    
    # Extend the seasonal cycle to match the length of the time series
    full_seasonal_pattern = np.tile(mean_seasonal_cycle, n_months // 12 + 1)[:n_months]
    
    # Deseasonalize the time series
    df_deseason_var = ts - full_seasonal_pattern

    # Step 2: 12-month moving window aggregation
    df_annual_var = np.full_like(df_deseason_var, np.nan)
    
    for i in range(11, len(df_deseason_var)):  # from the 12th month (index 11)
        window_data = df_deseason_var[i-11:i+1]  # 12-month window
        if method in ['sum']:  # for 'CGR',
            if np.sum(~np.isnan(window_data)) > 0: 
                df_annual_var[i] = np.nansum(window_data)
        else:  # for temperature, ENSO index,
               # 'pre_cru', 'pre_gpcc', and 'tws_grace'
            if np.sum(~np.isnan(window_data)) > 0:
                df_annual_var[i] = np.nanmean(window_data)
    
    # Step 3: Use LOWESS to detrend the annual data
    time_index = np.arange(len(df_annual_var))
    
    valid_mask = ~np.isnan(df_annual_var)
    if np.sum(valid_mask) < 10:
        return df_annual_var
    
    try:

        trend = sm.nonparametric.lowess(
            exog=time_index, 
            endog=df_annual_var, 
            frac=lowess_frac, 
            return_sorted=False
        )
        
        df_annual_detrended = df_annual_var - trend
        
        return df_annual_detrended
        
    except:
        return df_annual_var


# %%

# %% [markdown]
# # Test different windows of LOWESS

# %% [markdown]
# ## TWS

# %%
mm_tws_grace = pd.read_csv(proj_path+'/01-data/tws_grace.csv', index_col=0) 

mm_tws_grace['time'] = pd.date_range(start='1901-01-01', end='2019-07-01', freq='MS', inclusive='both')

mm_tws_grace = mm_tws_grace[(mm_tws_grace['time'] >= start_date) & (mm_tws_grace['time'] <= end_date)]

# %%
ts_0_15 = process_single_timeseries(mm_tws_grace['tws'], method='mean', lowess_frac=0.15)

ts_0_35 = process_single_timeseries(mm_tws_grace['tws'], method='mean', lowess_frac=0.35)

ts_0_75 = process_single_timeseries(mm_tws_grace['tws'], method='mean', lowess_frac=0.75)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=3, journal='nat2')

time_series = mm_tws_grace['time'].values
ax.plot(time_series, ts_0_15, color='blue' )
ax.plot(time_series, ts_0_35, color='green')
ax.plot(time_series, ts_0_75, color='red'  )

ax.plot(time_series, mm_tws_grace['tws'], color='black')

# %% [markdown]
# ## PRE

# %%
mm_pre_gpcc = pd.read_csv(proj_path+'/01-data/pre_gpcc.csv', index_col=0) 

mm_pre_gpcc['time'] = pd.date_range(start='1891-01-01', end='2019-12-01', freq='MS', inclusive='both')

mm_pre_gpcc = mm_pre_gpcc[(mm_pre_gpcc['time'] >= start_date) & (mm_pre_gpcc['time'] <= end_date)]

# %%
ts_0_15 = process_single_timeseries(mm_pre_gpcc['pre'], method='mean', lowess_frac=0.15)

ts_0_35 = process_single_timeseries(mm_pre_gpcc['pre'], method='mean', lowess_frac=0.35)

ts_0_75 = process_single_timeseries(mm_pre_gpcc['pre'], method='mean', lowess_frac=0.75)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=3, journal='nat2')

time_series = mm_tws_grace['time'].values
ax.plot(time_series, ts_0_15, color='blue' )
ax.plot(time_series, ts_0_35, color='green')
ax.plot(time_series, ts_0_75, color='red'  )



# %% [markdown]
# # TMP

# %%
# =========================================
# CRU
# =========================================

# project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'

# dataset path
data_dir  = proj_path + '/01-data/'
file_name = 'tmp_cru.nc'

ds = xr.open_dataset(os.path.join(data_dir, file_name))

ds = ds.sel(time=slice(start_date, end_date))

# %%
var = 'tmp'

ds_detrend = xr.apply_ufunc(
    process_single_timeseries,
    ds[var], 
    kwargs={'method': 'mean', 'lowess_frac': 4/5},
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True,
    dask='forbidden',  
    output_dtypes=[ds[var].dtype]
)

# Transpose the dimensions to match the original dataset
ds_detrend = ds_detrend.transpose('time', 'lat', 'lon')



# %%
# Export the detrended dataset
ds_detrend.to_netcdf(os.path.join(proj_path, f'01-data/tmp_cru_detrend.nc'))

# %%

# %% [markdown]
# # TWS

# %%
# =========================================
# GRACE
# =========================================

# project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'

# dataset path
data_dir  = proj_path + '/01-data/'
file_name = 'tws_grace.nc'

ds = xr.open_dataset(os.path.join(data_dir, file_name))

ds = ds.sel(time=slice(start_date, end_date))

# %%
var = 'rec_ensemble_mean'

ds_detrend = xr.apply_ufunc(
    process_single_timeseries,
    ds[var], 
    kwargs={'method': 'mean', 'lowess_frac': 4/5},
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True,
    dask='forbidden',  
    output_dtypes=[ds[var].dtype]
)

# Transpose the dimensions to match the original dataset
ds_detrend = ds_detrend.transpose('time', 'lat', 'lon')



# %%
# Export the detrended dataset
ds_detrend.to_netcdf(os.path.join(proj_path, f'01-data/tws_grace_detrend.nc'))

# %%

# %% [markdown]
# # SST

# %%

# %%
import xesmf as xe

def regrid_dataset(ds, resolution=1.0, method='bilinear'):
    """
    Regrid dataset to specified resolution while maintaining dask compatibility
    """
    # Create a target grid with specified resolution
    target_grid = xe.util.grid_global(resolution, resolution)
    
    # Create a regridder object
    regridder = xe.Regridder(ds, target_grid, method=method, periodic=True)
    
    # Conduct the regridding operation
    ds_regridded = regridder(ds)

    if 'y' in ds_regridded.dims and 'x' in ds_regridded.dims:
        ds_regridded = ds_regridded.rename({'y': 'lat', 'x': 'lon'})

    if 'lat' in ds_regridded.coords and 'lon' in ds_regridded.coords:
        lat_coord = ds_regridded.coords['lat']
        lon_coord = ds_regridded.coords['lon']
        
        if lat_coord.ndim == 2 and lon_coord.ndim == 2:
            lat_1d = lat_coord.values[:, 0] 
            lon_1d = lon_coord.values[0, :]  
            
            ds_regridded = ds_regridded.assign_coords({
                'lat': ('lat', lat_1d),
                'lon': ('lon', lon_1d)
            })
            
    return ds_regridded


# %%
# =========================================
# ERA5
# =========================================

# project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'

# dataset path
data_dir  = proj_path + '/01-data/'
file_name = 'sst_era5.nc'

ds = xr.open_dataset(os.path.join(data_dir, file_name))

ds = ds.sel(time=slice(start_date, end_date))

ds

# %%
# interpolate ERA5
ds = regrid_dataset(ds, resolution=1.0, method='bilinear')

# %%
var = 'sst'

ds_detrend = xr.apply_ufunc(
    process_single_timeseries,
    ds[var], 
    kwargs={'method': 'mean', 'lowess_frac': 4/5},
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True,
    dask='forbidden',  
    output_dtypes=[ds[var].dtype]
)

# Transpose the dimensions to match the original dataset
ds_detrend = ds_detrend.transpose('time', 'lat', 'lon')
ds_detrend = ds_detrend.sel(lat=slice(-23.5, 23.5))

# Export the detrended dataset
ds_detrend.to_netcdf(os.path.join(proj_path, f'01-data/sst_era5_detrend.nc'))
