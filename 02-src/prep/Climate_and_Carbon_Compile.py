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
Here we compile the time series of tropical climate and carbon data into a file.

"""

# --- Loading packages
# Numerical computing
import pandas as pd
import numpy  as np

# Statistical analysis
import statsmodels.api as sm
import pingouin as pg

# Path 
import os

# Warnings and errors
import warnings; warnings.filterwarnings("ignore")

# %%
# project path
proj_path = '~/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'
proj_path = os.path.expanduser(proj_path)

# change directory
os.chdir(proj_path)
print('We are locating in', os.getcwd())

# %% [markdown]
# # Monthly Data

# %% [markdown]
# ## Data Preparation

# %%

#! --- Mauna Loa CO2 concentration ---
# Monthly data, end date inferred dynamically from year/month columns

# load the data
mm_co2 = pd.read_csv('01-data/co2_mm_mlo.csv', sep=',', skiprows=40)
# select the columns
mm_co2 = mm_co2[['year', 'month', 'average']]
# remove missing values (-99.99)
mm_co2 = mm_co2[mm_co2['average'] > 0].reset_index(drop=True)
# construct time dynamically from year/month — no hardcoded end date
mm_co2['time'] = pd.to_datetime(dict(year=mm_co2['year'], month=mm_co2['month'], day=1))
mm_co2 = mm_co2[['time', 'year', 'month', 'average']]

# --- copy the data
mm_cgr = mm_co2.copy()
mm_cgr.rename(columns={'average': 'CGR'}, inplace=True)

# --- calculate the CGR
# #! Refer to "Variations in atmospheric CO2 growth rates coupled with tropical temperature"
#  https://www.pnas.org/doi/10.1073/pnas.1219683110
carbon_coefficient = 2.13
mm_cgr['CGR'] = mm_co2['average'].diff(periods=1) * carbon_coefficient  # ppm → Gt C

print(f"CO2 monthly: {mm_cgr['time'].iloc[0].date()} → {mm_cgr['time'].iloc[-1].date()}  (n={len(mm_cgr)})")


# %%

#! --- Nino3.4 / Nino3 / Nino4 ---
# Monthly data, end date inferred dynamically; -99.99 treated as NaN

months_cols = [str(i) for i in range(1, 13)]

def read_enso_index(path):
    """Read year×12month ENSO index file → long-format monthly DataFrame, index on time."""
    df = pd.read_csv(path, skiprows=3, names=months_cols, index_col=0, sep=r'\s+')
    df = df.replace(-99.99, np.nan)
    long = (df.reset_index()
              .melt(id_vars='index', var_name='month', value_name='val')
              .rename(columns={'index': 'year'}))
    long['year']  = long['year'].astype(int)
    long['month'] = long['month'].astype(int)
    long['time']  = pd.to_datetime(dict(year=long['year'], month=long['month'], day=1))
    return (long.dropna(subset=['val'])
               .sort_values('time')
               .set_index('time')['val'])   # return Series indexed by time

nina34_s = read_enso_index('01-data/nina34.csv')
nina3_s  = read_enso_index('01-data/nina3.csv')
nina4_s  = read_enso_index('01-data/nina4.csv')

# Build mm_enso on nina34's time index; reindex others to match
mm_enso = pd.DataFrame(index=nina34_s.index)
mm_enso['nina34'] = nina34_s
mm_enso['nina3']  = nina3_s.reindex(mm_enso.index)   # NaN where nina3 doesn't extend
mm_enso['nina4']  = nina4_s.reindex(mm_enso.index)
mm_enso = mm_enso.reset_index().rename(columns={'index': 'time'})
mm_enso['year']   = mm_enso['time'].dt.year
mm_enso['month']  = mm_enso['time'].dt.month

print(f"ENSO: {mm_enso['time'].iloc[0].date()} → {mm_enso['time'].iloc[-1].date()}  (n={len(mm_enso)})")
print(mm_enso[['nina34','nina3','nina4']].isna().sum().rename('NaN count'))


# %%
#! --- Process monthly climate data ---
# Define datasets to process
climate_datasets = {
    'pre_gpcc': ('01-data/pre_gpcc.csv', '1891-01-01', '2019-12-01', 'pre'),
    'pre_cru': ('01-data/pre_cru.csv', '1901-01-01', '2024-12-01', 'pre'),
    'tmp_cru': ('01-data/tmp_cru.csv', '1901-01-01', '2024-12-01', 'tmp'),
    'tws_grace': ('01-data/tws_grace.csv', '1901-01-01', '2019-07-01', 'tws')
}

# Dictionary to store monthly and annual dataframes
mm_datasets = {}

# Process each dataset
for var_name, (file_path, start_date, end_date, column) in climate_datasets.items():
    
    # Read and prepare data
    df = pd.read_csv(file_path, index_col=0)
    df['time'] = pd.date_range(start=start_date, end=end_date, freq='MS', inclusive='both')
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    mm_datasets[var_name] = df[['time', 'year', 'month', column]].copy()
    
# Assign to individual variables for compatibility with existing code
mm_pre_gpcc = mm_datasets['pre_gpcc']
mm_pre_cru = mm_datasets['pre_cru']
mm_tmp_cru = mm_datasets['tmp_cru']
mm_tws_grace = mm_datasets['tws_grace']

# %%

start_date = pd.Timestamp('1958-03-01')
end_date   = pd.Timestamp('2026-02-01')  # common end: CGR/ENSO/CRU/SST all available

# Clip each variable to its own available range
mm_cgr       = mm_cgr      [(mm_cgr['time']       >= start_date) & (mm_cgr['time']       <= end_date)]
mm_enso      = mm_enso     [(mm_enso['time']       >= start_date) & (mm_enso['time']       <= end_date)]
mm_tmp_cru   = mm_tmp_cru  [(mm_tmp_cru['time']   >= start_date) & (mm_tmp_cru['time']   <= end_date)]
mm_pre_cru   = mm_pre_cru  [(mm_pre_cru['time']   >= start_date) & (mm_pre_cru['time']   <= end_date)]
mm_pre_gpcc  = mm_pre_gpcc [(mm_pre_gpcc['time']  >= start_date) & (mm_pre_gpcc['time']  <= end_date)]
mm_tws_grace = mm_tws_grace[(mm_tws_grace['time'] >= start_date) & (mm_tws_grace['time'] <= end_date)]

# Compile using index-based alignment (reindex handles different end dates)
# GPCC ends 2019-12, TWS ends 2019-07 → NaN for later months; others continue to 2024-12
mm_all = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='MS'))
mm_all['year']      = mm_all.index.year
mm_all['month']     = mm_all.index.month
mm_all['time']      = mm_all.index

mm_all['CGR']       = mm_cgr.set_index('time')['CGR'].reindex(mm_all.index)
mm_all['tmp_cru']   = mm_tmp_cru.set_index('time')['tmp'].reindex(mm_all.index)
mm_all['pre_cru']   = mm_pre_cru.set_index('time')['pre'].reindex(mm_all.index)
mm_all['pre_gpcc']  = mm_pre_gpcc.set_index('time')['pre'].reindex(mm_all.index)   # NaN after 2019-12
mm_all['tws_grace'] = mm_tws_grace.set_index('time')['tws'].reindex(mm_all.index)  # NaN after 2019-07
mm_all['nina34']    = mm_enso.set_index('time')['nina34'].reindex(mm_all.index)
mm_all['nina3']     = mm_enso.set_index('time')['nina3'].reindex(mm_all.index)
mm_all['nina4']     = mm_enso.set_index('time')['nina4'].reindex(mm_all.index)

# Label eruption periods
mm_all['eruption'] = 0
eruption_periods = {
    'Mount Agung': ('1962-01-01', '1963-12-31'),
    'El Chichón':  ('1982-01-01', '1982-12-31'),
    'Pinatubo':    ('1992-01-01', '1993-12-31')
}
for _, (start, end) in eruption_periods.items():
    mm_all.loc[(mm_all.index >= start) & (mm_all.index <= end), 'eruption'] = 1

# Create mask for normal years (non-eruption periods)
mm_normal = mm_all.mask(mm_all['eruption'] == 1)

print(f"mm_all: {mm_all.index[0].date()} → {mm_all.index[-1].date()}  (n={len(mm_all)})")
print(mm_all[['CGR','nina34','tmp_cru','pre_gpcc','tws_grace']].isna().sum().rename('NaN count'))


# %%

# #! Detrend all variables in one go
variables = ['CGR', 'tmp_cru', 'pre_cru', 'pre_gpcc', 'tws_grace', 'nina34', 'nina3', 'nina4']

x_idx = np.arange(len(mm_all))  # numeric index for LOWESS (avoids NaN in time column)

for var in variables:

    df_var = mm_all[[var, 'month', 'year']].copy()

    # Step 1: Seasonal cycle (computed on non-NaN values only)
    mean_seasonal_cycle = df_var.groupby('month')[var].mean().values
    full_seasonal_pattern = np.tile(mean_seasonal_cycle, len(df_var) // 12 + 1)[:len(df_var)]
    df_deseason_var = df_var[var] - full_seasonal_pattern

    # Step 2: Rolling sum (CGR) or mean (all others)
    if var == 'CGR':
        df_annual_var = df_deseason_var.rolling(window=12, center=False).sum()
    else:
        df_annual_var = df_deseason_var.rolling(window=12, center=False).mean()

    # Step 3: LOWESS detrend — fit on non-eruption points only (same as notebook)
    trend = sm.nonparametric.lowess(exog=mm_normal['time'], endog=df_annual_var, frac=4/5, return_sorted=False)
    mm_all[var] = df_annual_var - trend

# ========== Save the data ========== #
mm_all.to_csv('01-data/tropical_month_climate_carbon_atlas_mean.csv', index=True, header=True)

# %%
# Create mask for normal years (non-eruption periods)
mm_normal = mm_all.mask(mm_all['eruption'] == 1)

# %%
# Without moving average

print('TMP', pg.corr(mm_normal['CGR'], mm_normal['tmp_cru'].shift(0), method='pearson'))
print('PRE', pg.corr(mm_normal['CGR'], mm_normal['pre_gpcc'].shift(0), method='pearson'))
print('TWS', pg.corr(mm_normal['CGR'], mm_normal['tws_grace'].shift(0), method='pearson'))

# %%

# %% [markdown]
# # Annual data
#
# """ We just use that for reproduction of the publications

# %%

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Constants
CARBON_COEFFICIENT = 2.13
START_YEAR, END_YEAR = 1959, 2024  # extended from 2019 → 2024
ERUPTION_PERIODS = {
    'Mount Agung': ('1962-01-01', '1963-12-31'),
    'El Chichón':  ('1982-01-01', '1982-12-31'),
    'Pinatubo':    ('1991-01-01', '1993-12-31')
}
CLIMATE_DATASETS = {
    'pre_gpcc':  ('01-data/pre_gpcc.csv',  '1891-01-01', '2019-12-01', 'pre'),  # ends 2019
    'pre_cru':   ('01-data/pre_cru.csv',   '1901-01-01', '2024-12-01', 'pre'),
    'tmp_cru':   ('01-data/tmp_cru.csv',   '1901-01-01', '2024-12-01', 'tmp'),
    'tws_grace': ('01-data/tws_grace.csv', '1901-01-01', '2019-07-01', 'tws'),  # ends 2019-07
}

# Read Mauna Loa annual CGR data
yr_cgr = (pd.read_csv('01-data/co2_gr_mlo.csv', comment='#')
          [['year', 'ann inc']]
          .set_index('year')
          .rename(columns={'ann inc': 'CGR'}))
yr_cgr['CGR'] *= CARBON_COEFFICIENT

# Process climate datasets
yr_datasets = {}
for var_name, (file_path, start_date, end_date, column) in CLIMATE_DATASETS.items():
    try:
        df = pd.read_csv(file_path, index_col=0)
        df['time'] = pd.date_range(start=start_date, end=end_date, freq='MS')
        df['year'] = df['time'].dt.year
        if 'pre' in var_name:
            df[f'{column}_shifted'] = df[column].shift(6)
            yr_datasets[var_name] = df.groupby('year')[f'{column}_shifted'].sum().to_frame(name=column)
        elif 'tws' in var_name:
            yr_datasets[var_name] = df.groupby('year')[column].sum().to_frame(name=column)
        else:
            yr_datasets[var_name] = df.groupby('year')[column].mean().to_frame(name=column)
    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error processing {file_path}: {e}")
        continue

# Combine using reindex — GPCC/TWS get NaN for years beyond their range
yr_all = pd.DataFrame(index=pd.date_range(start=str(START_YEAR), end=str(END_YEAR), freq='AS'))
yr_all['year'] = yr_all.index.year
for var_name, df in yr_datasets.items():
    col = df.columns[0]
    yr_all[var_name] = df[col].reindex(yr_all['year'].values).values
yr_all['CGR'] = yr_cgr['CGR'].reindex(yr_all['year'].values).values

# Label eruption periods
yr_all['eruption'] = 0
for start, end in ERUPTION_PERIODS.values():
    yr_all.loc[start:end, 'eruption'] = 1

# Detrend function (NaN-safe)
def detrend_data(data, variable, method='lowess', frac=4/5):
    x = data.index.year.values
    y = data[variable].values
    mask = ~np.isnan(y)
    if method == 'lowess':
        trend_vals = sm.nonparametric.lowess(exog=x[mask], endog=y[mask], frac=frac, return_sorted=False)
        trend = np.full_like(y, np.nan, dtype=float)
        trend[mask] = trend_vals
    elif method == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x[mask].reshape(-1, 1), y[mask])
        trend = np.full_like(y, np.nan, dtype=float)
        trend[mask] = model.predict(x[mask].reshape(-1, 1))
    return y - trend

for var in [c for c in yr_all.columns if c not in ('eruption', 'year')]:
    yr_all[var] = detrend_data(yr_all, var, method='lowess')

# Save output
yr_all.to_csv('01-data/tropical_year_climate_carbon_atlas.csv')
print(f"yr_all: {START_YEAR}–{END_YEAR}  (n={len(yr_all)})")
print(yr_all[['CGR','tmp_cru','pre_gpcc','tws_grace']].isna().sum().rename('NaN count'))


# %%
# Create mask for normal years (non-eruption periods)
yr_normal = yr_all.mask(yr_all['eruption'] == 1)

# %%
print('TMP', pg.corr(yr_normal['CGR'], yr_normal['tmp_cru'].shift(0), method='pearson'))
print('PRE', pg.corr(yr_normal['CGR'], yr_normal['pre_cru'].shift(0), method='pearson'))
print('TWS', pg.corr(yr_normal['CGR'], yr_normal['tws_grace'].shift(0), method='pearson'))
