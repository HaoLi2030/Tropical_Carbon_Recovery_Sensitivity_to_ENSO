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
We here extracted the regional average values of tropical climate, including precipitation (from CRU and GPCC), temperature (from CRU), terrestrial water storage (TWS, from resconstruction) and sea surface temperature (from ERA5).
"""

# Load packages
import xarray as xr 
import pandas as pd
import numpy  as np
import proplot as pplt
import seaborn as sns 

import os 

# %%
# project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'

# change directory
os.chdir(proj_path)
print('We are locating in', os.getcwd())

# %% [markdown]
# # Tropical average climate

# %%
# project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'
proj_path = os.path.expanduser(proj_path)

# change directory
os.chdir(proj_path)
print('We are locating in', os.getcwd()) 

# %% [markdown]
# ## CRU

# %%
# =========================================
# CRU Precipitation
# =========================================

# define dataframe
df = pd.DataFrame(index=pd.date_range('1901-01-01', '2024-12-31', freq='MS'))

# load data
data_dir = '/Users/hao/EXTERNAL/DATA/CRU'
file_dir = 'cru_ts4.09.1901.2024.pre.dat.nc'

ds = xr.open_dataset(os.path.join(data_dir, file_dir))

# calculate the average between 23.5°N and 23.5°S
ds = ds['pre'].sel(lat=slice(-23.5, 23.5))

# #! export to netcdf
ds.to_netcdf(os.path.join(proj_path, '01-data/pre_cru.nc'))

# compute weights as cosine of latitude (in radians)
weights = np.cos(np.deg2rad(ds.lat))

# apply weights to latitude dimension and compute weighted mean
df['pre'] = ds.weighted(weights).sum(dim=['lat', 'lon'])

# save data
df.to_csv(os.path.join(proj_path, '01-data/pre_cru.csv'), index=True)

# %%
# =========================================
# CRU Air temperature
# =========================================

# define dataframe
df = pd.DataFrame(index=pd.date_range('1901-01-01', '2024-12-31', freq='MS'))

# load data
data_dir = '/Users/hao/EXTERNAL/DATA/CRU'
file_dir = 'cru_ts4.09.1901.2024.tmp.dat.nc'

ds = xr.open_dataset(os.path.join(data_dir, file_dir))

# calculate the average between 23.5°N and 23.5°S
ds = ds['tmp'].sel(lat=slice(-23.5, 23.5))

# #! export to netcdf
ds.to_netcdf(os.path.join(proj_path, '01-data/tmp_cru.nc'))

# compute weights as cosine of latitude (in radians)
weights = np.cos(np.deg2rad(ds.lat))

# apply weights to latitude dimension and compute weighted mean
df['tmp'] = ds.weighted(weights).mean(dim=['lat', 'lon'])

# #! save data
df.to_csv(os.path.join(proj_path, '01-data/tmp_cru.csv'), index=True)

# %% [markdown]
# ## GPCC

# %%
# =========================================
# GPCC
# =========================================

# define dataframe
df = pd.DataFrame(index=pd.date_range('1891-01-01', '2019-12-31', freq='MS'))

# load data
data_dir = '/Users/hao/EXTERNAL/DATA/GPCC'
file_dir = 'precip.mon.total.1x1.v2020.nc'
ds = xr.open_dataset(os.path.join(data_dir, file_dir))

# adjust the dataset
# 1. transpose the lat (lat: 90 to -90)
ds = ds.sortby('lat', ascending=True)
# 2. convert the lon (lon: 0 to 360)
ds['lon'] = (ds.lon + 180) % 360 - 180
ds = ds.sortby('lon')

# calculate the average between 23.5°N and 23.5°S
ds = ds['precip'].sel(lat=slice(-23.5, 23.5))

# #! export to netcdf
ds.to_netcdf(os.path.join(proj_path, '01-data/pre_gpcc.nc'))

# compute weights as cosine of latitude (in radians)
weights = np.cos(np.deg2rad(ds.lat))

# apply weights to latitude dimension and compute weighted mean
df['pre'] = ds.weighted(weights).mean(dim=['lat', 'lon'])

# #! save data
df.to_csv(os.path.join(proj_path, '01-data/pre_gpcc.csv'), index=True)

# %% [markdown]
# ## GRACE

# %%
GRACE_type   = ['GSFC', 'JPL']
Climate_type = ['GSWP3', 'ERA5', 'MSWEP']

# %%
"""
GSWP3: 1901 - 2014 
ERA:   1979 - 2019
MSWEP: 1979 - 2016
"""

before_period = ('1901-01-01', '1978-12-31')
middle_period = ('1979-01-01', '2014-12-31')
after_period  = ('2015-01-01', '2019-07-31')

# define dataframe
df = pd.DataFrame(index=pd.date_range('1901-01-01', '2019-07-31', freq='MS'))

# %%
data_dir = '/Users/hao/EXTERNAL/DATA/GRACE-REC'

##########
file_dir = 'GRACE_REC_v03_' + GRACE_type[0] + '_' + Climate_type[0] + '_monthly_ensemble_mean.nc'
ds1 = xr.open_dataset(os.path.join(data_dir, file_dir), chunks='auto')['rec_ensemble_mean']

#########
file_dir = 'GRACE_REC_v03_' + GRACE_type[0] + '_' + Climate_type[1] + '_monthly_ensemble_mean.nc'
ds2 = xr.open_dataset(os.path.join(data_dir, file_dir), chunks='auto')['rec_ensemble_mean']

# %%
before_ds = ds1.sel(time=slice(before_period[0], before_period[1]))

middle_ds = (ds1.sel(time=slice(middle_period[0], middle_period[1])) + ds2.sel(time=slice(middle_period[0], middle_period[1])))/2

after_ds  = ds2.sel(time=slice(after_period[0], after_period[1]))

ds = xr.concat([before_ds, middle_ds, after_ds], dim='time')

# %%
# calculate the average between 23.5°N and 23.5°S
ds = ds.sel(lat=slice(-23.5, 23.5))

# #! export to netcdf
ds.to_netcdf(os.path.join(proj_path, '01-data/tws_grace.nc'))

# compute weights as cosine of latitude (in radians)
weights = np.cos(np.deg2rad(ds.lat))

# apply weights to latitude dimension and compute weighted mean
df['tws'] = ds.weighted(weights).mean(dim=['lat', 'lon'])

# #! save data
df.to_csv(os.path.join(proj_path, '01-data/tws_grace.csv'), index=True)

# %% [markdown]
# ## ERA5

# %%
# =========================================
# ERA5
# =========================================

# load data
data_dir = '/Users/hao/EXTERNAL/DATA/ERA5/data/month'
file_dir = 'era5_sea_surface_temperature_monthly_1940-2024.nc'
ds = xr.open_dataset(os.path.join(data_dir, file_dir))

# 1. transpose the lat (lat: 90 to -90)
ds = ds.sortby('latitude', ascending=True)

# 2. convert the longitude (longitude: 0 to 360)
ds['longitude'] = (ds.longitude + 180) % 360 - 180
ds = ds.sortby('longitude')

# 3. rename the latitufde and longitude dimensions
ds = ds.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'})

# 4. calculate the average between 23.5°N and 23.5°S
ds = ds['sst'].sel(lat=slice(-23.5, 23.5))

# 5. drop unnecessary variables
ds = ds.drop_vars(['expver', 'number'], errors='ignore')

# #! export to netcdf
ds.to_netcdf(os.path.join(proj_path, '01-data/sst_era5.nc'))
