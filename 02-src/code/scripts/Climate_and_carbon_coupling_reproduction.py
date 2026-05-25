# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:percent
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
# Numerical computing
import pandas as pd
import numpy  as np

# Data visualization
import proplot as pplt

# Statistics
import pingouin as pg
import statsmodels.api as sm

# Warnings and errors
import warnings; warnings.filterwarnings("ignore")

# Path 
import os

# %%
# project path
proj_path = '~/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'
proj_path = os.path.expanduser(proj_path)

# change directory
os.chdir(proj_path)
print('We are locating in', os.getcwd())

# %%
df_all = pd.read_csv('01-data/tropical_month_climate_carbon_atlas_mean.csv', index_col=0)

# Create mask for normal years (non-eruption periods)
df_normal = df_all.mask(df_all['eruption'] == 1)

# %% [markdown]
# # Reproduce CGR 

# %% [markdown]
# Humphrey, V., Zscheischler, J., Ciais, P. et al. Sensitivity of atmospheric CO2 growth rate to observed changes in terrestrial water storage. Nature 560, 628–631 (2018). https://doi.org/10.1038/s41586-018-0424-4

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.5, journal='nat2')

# ------ CGR
ax.plot(
        pd.DatetimeIndex(df_normal.index), 
        df_normal['CGR'],
        '-o', ms=0, alpha=1, lw=2, 
        color='k',
)
ax.format(
    xlabel='',
    ylim=(4.5, -3.5),  # Note: reversed order to match the inverted axis
    xlim=(np.datetime64('1980-01-01'), np.datetime64('2016-12-31')),
    xlocator=('year', 5),
    ylocator=2,
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Carbon Growth Rate (PgC/yr)',  # Updated label to be more descriptive

)

# ------ Water-related variables
ax_twin = ax.twinx()

# Precipitation
ax_twin.plot(
        pd.DatetimeIndex(df_normal.index), 
        df_normal['pre_gpcc'].shift(9),
        '-o', ms=0, alpha=1, lw=2, 
        color='blue', label='(6-month lagged) Precipitation',
)

# TWS
ax_twin.plot(
        pd.DatetimeIndex(df_normal.index),
        df_normal['tws_grace'].shift(0),
        '-o', ms=0, alpha=1, lw=2,
        color='red', label='(Concurrent) TWS',
)

ax_twin.format(
    xlabel='',
    ylim=(-30,30),  # Note: reversed order to match the inverted axis
    xlim=(np.datetime64('1958-01-01'), np.datetime64('2020-12-31')),
    xlocator=('year', 5),
    ylocator=100,
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Lag Water-related Variables (mm)',  # Updated label to be more descriptive
)

ax.legend(ncol=3, loc='b', fontsize=10, frame=False)


# %% [markdown]
# # Reproduce Temperature Sensitivty

# %% [markdown]
# 1. Wang, X., Piao, S., Ciais, P. et al. A two-fold increase of carbon cycle sensitivity to tropical temperature variations. Nature 506, 212–215 (2014). https://doi.org/10.1038/nature12915
# 2. Zhang, W., Schurgers, G., Peñuelas, J. et al. Recent decrease of the impact of tropical temperature on the carbon cycle linked to increased precipitation. Nat Commun 14, 965 (2023). https://doi.org/10.1038/s41467-023-36727-2
# 3. He, B., Xie, X., & Guo, L. (2023). A shift from temperature to water as the primary driver for interannual variability of the tropical carbon cycle. Geophysical Research Letters, 50, e2023GL102812. https://doi.org/10.1029/2023GL102812
#

# %% [markdown]
# ## 1. Correlation 

# %%
def calculate_rolling_correlation(df, predictor='P', response='CGR', window_years=25, lag=0, min_samples=60):
    """
    Calculate rolling correlation between two variables using a specified window.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with predictor and response columns, and datetime index
    predictor : str, optional
        Predictor variable column name (default: 'P')
    response : str, optional
        Response variable column name (default: 'CGR')
    window_years : int, optional
        Size of the rolling window in years (default: 25)
    lag : int, optional
        Lag in months (negative means predictor leads response, default: 0)
    min_samples : int, optional
        Minimum samples required for correlation (default: 60)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with:
        - middle_date: center date of the window
        - corr: correlation coefficient
        - ci_lower: lower bound of 95% CI for correlation
        - ci_upper: upper bound of 95% CI for correlation
    """
    # Check if columns exist
    if predictor not in df.columns or response not in df.columns:
        raise ValueError(f"Predictor '{predictor}' or response '{response}' not found in DataFrame columns")

    window_size = window_years * 12
    num_windows = len(df) - window_size + 1
    result_list = []

    # Precompute middle dates
    middle_indices = np.array([start_idx + window_size // 2 for start_idx in range(num_windows)])
    middle_dates = df.index[middle_indices]

    for start_idx in range(num_windows):
        end_idx = start_idx + window_size
        # Select only relevant columns and drop NaN rows
        window_data = df.iloc[start_idx:end_idx][[predictor, response]].dropna()

        # Initialize result dictionary
        result = {'middle_date': middle_dates[start_idx], 'corr': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

        if len(window_data) < min_samples:
            result_list.append(result)
            continue

        # Apply lag and handle NaNs
        X_values = window_data[predictor].shift(lag)
        Y_values = window_data[response]
        paired_data = pd.DataFrame({'X': X_values, 'Y': Y_values}).dropna()

        # Check if enough paired samples remain after lag
        if len(paired_data) >= min_samples:
            corr_result = pg.corr(paired_data['X'], paired_data['Y'], method='pearson')
            result['corr'] = corr_result['r'].iloc[0]
            ci = corr_result['CI95%'].iloc[0]
            result['ci_lower'] = ci[0]
            result['ci_upper'] = ci[1]

        result_list.append(result)

    result_df = pd.DataFrame(result_list)
    result_df['middle_date'] = pd.to_datetime(result_df['middle_date'])
    return result_df


# %% [markdown]
# ### Temperature and CGR

# %%
# Calculate rolling correlations between temperature and CGR
result_tmp_25 = calculate_rolling_correlation(df=df_normal, predictor='tmp_cru', response='CGR', window_years=25, lag=0)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
        (result_tmp_25, 'r', '25-year'),
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['corr'],
                '-o', ms=0, alpha=1, lw=5, 
                color=color,
                label=f'Correlation with {window_label} window'
        )
        
        if color == 'r':

                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['ci_lower'],
                        result['ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    ylim=(0.3, 0.8),
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    ylocator=0.1,
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Correlation between CGR and T',
)

# %% [markdown]
# ### Precipitation and CGR

# %%
# Calculate rolling correlations between temperature and CGR
result_pre_25 = calculate_rolling_correlation(df=df_normal, predictor='pre_gpcc', response='CGR', window_years=25, lag=3)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [

        (result_pre_25, 'g', '25-year'),
     
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['corr'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'Correlation with {window_label} window'
        )
        
        if color == 'g':

                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['ci_lower'],
                        result['ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=True, xgrid=False,
    ylabel='Correlation between CGR and P',
)

# %% [markdown]
# ### TWS and CGR

# %%
df_normal

# %%
# Calculate rolling correlations between temperature and CGR
result_tws_25 = calculate_rolling_correlation(df=df_normal, predictor='tws_grace', response='CGR', window_years=25, lag=0)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [

        (result_tws_25, 'b', '25-year'),
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['corr'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'Correlation with {window_label} window'
        )
        
        if color == 'b':

                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['ci_lower'],
                        result['ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, )
                      
# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=True, xgrid=False,
    ylabel='Correlation between CGR and TWS',
)

# %% [markdown]
# ### ENSO and CGR

# %%
# Calculate rolling correlations between temperature and CGR
result_enso_25 = calculate_rolling_correlation(df=df_normal, predictor='nina34', response='CGR', window_years=25, lag=6)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
        
        (result_enso_25, 'g', '25-year'),

]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['corr'],
                '-o', ms=0, alpha=1, lw=2, 
                color=color,
                label=f'Correlation with {window_label} window'
        )
        
        if color == 'g':

                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['ci_lower'],
                        result['ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    # ylim=(0, 0.7), ylocator=0.1, 
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),  xlocator=('year', 3),  
    xformatter='%Y',  xminorlocator=('year', 1), 
    xrotation=45,  xgrid=False,
    ygrid=True,
    ylabel='Correlation between CGR and ENSO',
)


# %% [markdown]
# ## 2. Sensitivity

# %%
def calculate_rolling_regression(df, predictors=['pre_cru', 'tmp_cru'], response='CGR', window_years=25, lags=None, min_samples=10):
    """
    Calculate rolling regression with user-defined predictors using a specified window, with optional lags.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with response and predictor columns, and datetime index
    predictors : list of str, optional
        List of predictor variable column names (default: ['pre_cru', 'tmp_cru'])
    response : str, optional
        Response variable column name (default: 'CGR')
    window_years : int, optional
        Size of the rolling window in years (default: 25)
    lags : list of int, optional
        List of lags (in months) for each predictor (default: None, meaning no lags)
    min_samples : int, optional
        Minimum samples required for regression (default: 10)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with:
        - middle_date: center date of the window
        - r2: R-squared of the regression
        - {var}_coeff: regression coefficient for each predictor
        - {var}_ci_lower: lower bound of 95% CI for each predictor
        - {var}_ci_upper: upper bound of 95% CI for each predictor
    """
    # Check if columns exist
    missing_cols = [col for col in predictors + [response] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in DataFrame")

    # Check lags parameter
    if lags is None:
        lags = [0] * len(predictors)  # Default: no lags
    if len(lags) != len(predictors):
        raise ValueError("Length of lags must match length of predictors")

    window_size = window_years * 12
    num_windows = len(df) - window_size + 1
    result_list = []

    # Precompute middle dates
    middle_indices = np.array([start_idx + window_size // 2 for start_idx in range(num_windows)])
    middle_dates = df.index[middle_indices]

    for start_idx in range(num_windows):
        end_idx = start_idx + window_size
        # Select relevant columns
        columns = predictors + [response]
        window_data = df.iloc[start_idx:end_idx][columns]

        # Initialize result dictionary
        result = {'middle_date': middle_dates[start_idx], 'r2': np.nan}
        for var in predictors:
            result[f'{var}_coeff'] = np.nan
            result[f'{var}_ci_lower'] = np.nan
            result[f'{var}_ci_upper'] = np.nan

        # Apply lags to predictors and prepare data for regression
        X_dict = {}
        for var, lag in zip(predictors, lags):
            X_dict[var] = window_data[var].shift(lag)  # Apply lag to each predictor
        X = pd.DataFrame(X_dict)
        y = window_data[response]

        # Drop rows with NaN in any predictor or response
        paired_data = pd.concat([X, y.rename('y')], axis=1).dropna()

        if len(paired_data) < min_samples:
            result_list.append(result)
            continue

        # Prepare regression data
        X = paired_data[predictors]
        X = sm.add_constant(X)
        y = paired_data['y']

        # Fit regression model
        model = sm.OLS(y, X).fit()

        # Extract results
        result['r2'] = model.rsquared
        for var in predictors:
            result[f'{var}_coeff'] = model.params[var]
            ci = model.conf_int().loc[var]
            result[f'{var}_ci_lower'] = ci[0]
            result[f'{var}_ci_upper'] = ci[1]

        result_list.append(result)

    result_df = pd.DataFrame(result_list)
    result_df['middle_date'] = pd.to_datetime(result_df['middle_date'])
    return result_df


# %% [markdown]
# ### Temperature and CGR

# %%
result_tmp_25 = calculate_rolling_regression(df_normal, predictors=['tmp_cru'], response='CGR', window_years=25, lags=[0], min_samples=10)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [

        (result_tmp_25, 'r', '25-year'),
    
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['tmp_cru_coeff'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'CGR~f(T) with {window_label} window'
        )
        
        if color == 'r':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['tmp_cru_ci_lower'],
                        result['tmp_cru_ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    ylim=(1, 7),
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    ylocator=1,
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'Temperature sensitivity ($\gamma_{T}$)',
)

# %%
result_tmp_pre_25 = calculate_rolling_regression(df_normal, predictors=['tmp_cru', 'pre_gpcc'], response='CGR', window_years=25, lags=[0, 3],min_samples=10)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
        
        (result_tmp_pre_25, 'r', '25-year'),
] 

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['tmp_cru_coeff'],
                '-o', ms=0, alpha=1, lw=2, 
                color=color,
                label=f'CGR~f(T,P) with {window_label} window'
        )

        if color == 'r':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['tmp_cru_ci_lower'],
                        result['tmp_cru_ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'Temperature sensitivity ($\gamma_{T}$)',
)

# %% [markdown]
# ### Precipitation and CGR

# %%
result_pre_25 = calculate_rolling_regression(df_normal, predictors=['pre_gpcc'], response='CGR', window_years=25, lags=[3], min_samples=10)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
       
        (result_pre_25, 'g', '25-year'),     
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['pre_gpcc_coeff'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'CGR~f(P) with {window_label} window'
        )
        
        if color == 'g':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['pre_gpcc_ci_lower'],
                        result['pre_gpcc_ci_upper'],
                        lw=0, alpha=0.25,
                        color=color,
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'Precipitation sensitivity ($\gamma_{P}$)',
)

# %%

result_tmp_pre_25 = calculate_rolling_regression(df_normal, predictors=['tmp_cru', 'pre_gpcc'], response='CGR', window_years=25, lags=[0,3], min_samples=10)


# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
    
        (result_tmp_pre_25, 'g', '25-year'),
     
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['pre_gpcc_coeff'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'CGR~f(T,P) with {window_label} window'
        )
        
        if color == 'g':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['pre_gpcc_ci_lower'],
                        result['pre_gpcc_ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'Precipitation sensitivity ($\gamma_{P}$)',
)

# %% [markdown]
# ### TWS and CGR

# %%

result_tws_25 = calculate_rolling_regression(df_normal, predictors=['tws_grace'], response='CGR', window_years=25, lags=[0], min_samples=10)


# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
  
        (result_tws_25, 'b', '25-year'),
 
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['tws_grace_coeff'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'CGR~f(TWS) with {window_label} window'
        )
        
        if color == 'b':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['tws_grace_ci_lower'],
                        result['tws_grace_ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'TWS sensitivity ($\gamma_{TWS}$)',
)

# %%
result_tmp_tws_25 = calculate_rolling_regression(df_normal, predictors=['tmp_cru', 'tws_grace'], response='CGR', window_years=25, lags=[0,0], min_samples=10)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
      
        (result_tmp_tws_25, 'b', '25-year'),
       
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['tws_grace_coeff'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'CGR~f(TWS,T) with {window_label} window'
        )
        
        if color == 'b':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['tws_grace_ci_lower'],
                        result['tws_grace_ci_upper'],
                        lw=0, alpha=0.25,
                        color=color,
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'TWS sensitivity ($\gamma_{TWS}$)',
)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
      
        (result_tmp_tws_25, 'r', '25-year'),
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'], 
                result['tmp_cru_coeff'],
                '-o', ms=0, alpha=1, lw=3, 
                color=color,
                label=f'CGR~f(TWS,T) with {window_label} window'
        )
        
        if color == 'r':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['tmp_cru_ci_lower'],
                        result['tmp_cru_ci_upper'],
                        lw=0, alpha=0.25,
                        color=color,
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'T sensitivity ($\gamma_{T}$)',
)

# %% [markdown]
# ### ENSO and CGR

# %%

result_enso_25 = calculate_rolling_regression(df_normal, predictors=['nina34'], response='CGR', window_years=25, lags=[6], min_samples=10)


# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
  
        (result_enso_25, 'y', '25-year'),
 
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the temperature coefficient
        ax.plot(
                result['middle_date'],
                result['nina34_coeff'],
                '-o', ms=0, alpha=1, lw=3,
                color=color,
                label=f'CGR~f(ENSO) with {window_label} window'
        )
        
        if color == 'y':
                # Plot the confidence interval
                ax.fill_between(
                        result['middle_date'],
                        result['nina34_ci_lower'],
                        result['nina34_ci_upper'],
                        lw=0, alpha=0.25,
                        color=color,
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'ENSO sensitivity ($\gamma_{ENSO}$)',
)

# %% [markdown]
# # Reproduce Liu et al. (2023)

# %% [markdown]
# 1. Liu, L., Ciais, P., Wu, M. et al. Increasingly negative tropical water–interannual CO2 growth rate coupling. Nature 618, 755–760 (2023). https://doi.org/10.1038/s41586-023-06056-x
#

# %%
df_all = pd.read_csv('01-data/tropical_year_climate_carbon_atlas.csv', index_col=0)

# Create mask for normal years (non-eruption periods)
df_normal = df_all.mask(df_all['eruption'] == 1)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.5, journal='nat2')

# ------ CGR
ax.plot(
        pd.DatetimeIndex(df_normal.index), 
        df_normal['CGR'],
        '-o', ms=0, alpha=1, lw=2, 
        color='k',
)
ax.format(
    xlabel='',
    ylim=(4.5, -3.5),  # Note: reversed order to match the inverted axis
    xlim=(np.datetime64('1956-01-01'), np.datetime64('2020-12-31')),
    xlocator=('year', 5),
    ylocator=2,
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Carbon Growth Rate (PgC/yr)',  # Updated label to be more descriptive

)


# %% [markdown]
# ## Correlation

# %%
def calculate_annual_rolling_correlation(df, predictor='pre_cru', response='CGR', window_years=25, min_samples=10):
    """
    Calculate rolling correlation on annual data with a specified window.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with annual data, index as years
    predictor : str, optional
        Predictor variable column name (default: 'pre_cru')
    response : str, optional
        Response variable column name (default: 'CGR')
    window_years : int, optional
        Size of the rolling window in years (default: 25)
    min_samples : int, optional
        Minimum samples required for correlation (default: 20)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with:
        - middle_year: center year of the window
        - corr: correlation coefficient
        - ci_lower: lower bound of 95% CI for correlation
        - ci_upper: upper bound of 95% CI for correlation
    """
    num_windows = len(df) - window_years + 1
    result_list = []

    # Precompute middle years
    middle_indices = np.array([start_idx + window_years // 2 for start_idx in range(num_windows)])
    middle_years = df.index[middle_indices]

    for start_idx in range(num_windows):
        end_idx = start_idx + window_years
        window_data = df.iloc[start_idx:end_idx]

        # Drop rows with NaN
        window_data = window_data.dropna()

        # Initialize result dictionary
        result = {'middle_year': middle_years[start_idx], 'corr': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

        if len(window_data) < min_samples:
            result_list.append(result)
            continue

        # Prepare data for correlation
        X_values = window_data[predictor]
        Y_values = window_data[response]

        if len(X_values) >= min_samples:
            corr_result = pg.corr(X_values, Y_values, method='spearman')
            result['corr'] = corr_result['r'].iloc[0]
            ci = corr_result['CI95%'].iloc[0]
            result['ci_lower'] = ci[0]
            result['ci_upper'] = ci[1]

        result_list.append(result)

    result_df = pd.DataFrame(result_list)
    result_df['middle_year'] = pd.to_datetime(result_df['middle_year'])
    
    return result_df


# %%
result_pre_25_cru = calculate_annual_rolling_correlation(df_normal, predictor='pre_cru', response='CGR', window_years=25, min_samples=10)
result_pre_25_gpcc = calculate_annual_rolling_correlation(df_normal, predictor='pre_gpcc', response='CGR', window_years=25, min_samples=10)

result_tws_25_grace = calculate_annual_rolling_correlation(df_normal, predictor='tws_grace', response='CGR', window_years=25, min_samples=10)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
        (result_pre_25_cru, 'k', '25-year CRU'),
        (result_pre_25_gpcc, 'b', '25-year GPCC'),
        (result_tws_25_grace, 'r', '25-year GRACE')
        
]

# Loop through each result to plot
for result, color, window_label in window_results:
        # Plot the precipitation coefficient
        ax.plot(
                result['middle_year'], 
                result['corr'],
                '-o', ms=8, alpha=1, lw=2, 
                color=color,
                label=f'Correlation with {window_label} window'
        )
        
        if color == 'r':

                # Plot the confidence interval
                ax.fill_between(
                        result['middle_year'],
                        result['ci_lower'],
                        result['ci_upper'],
                        lw=0, alpha=0.25, 
                        color=color, 
                        label='_nolegend_'
                )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(#NINO3.4
    xlabel='',
    ylim=(0, -1),
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3),
    ylocator=0.2,
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Correlation between CGR and Water',
)


# %% [markdown]
# ## Sensitivity

# %%
def calculate_annual_rolling_regression(df, predictors=['pre_cru', 'tmp_cru'], response='CGR', window_years=25, min_samples=10):
    """
    Calculate rolling regression on annual data with a specified window.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with annual data, index as years
    predictors : list of str, optional
        List of predictor variable column names (default: ['pre_cru', 'tmp_cru'])
    response : str, optional
        Response variable column name (default: 'CGR')
    window_years : int, optional
        Size of the rolling window in years (default: 25)
    min_samples : int, optional
        Minimum samples required for regression (default: 20)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with:
        - middle_year: center year of the window
        - r2: R-squared of the regression
        - {var}_coeff: regression coefficient for each predictor
        - {var}_ci_lower: lower bound of 95% CI for each predictor
        - {var}_ci_upper: upper bound of 95% CI for each predictor
    """
    num_windows = len(df) - window_years + 1
    result_list = []

    # Precompute middle years
    middle_indices = np.array([start_idx + window_years // 2 for start_idx in range(num_windows)])
    middle_years = df.index[middle_indices]

    for start_idx in range(num_windows):
        end_idx = start_idx + window_years
        window_data = df.iloc[start_idx:end_idx]

        # Drop rows with NaN
        window_data = window_data.dropna()

        # Initialize result dictionary
        result = {'middle_year': middle_years[start_idx], 'r2': np.nan}
        for var in predictors:
            result[f'{var}_coeff'] = np.nan
            result[f'{var}_ci_lower'] = np.nan
            result[f'{var}_ci_upper'] = np.nan

        if len(window_data) < min_samples:
            result_list.append(result)
            continue

        # Prepare data for regression
        X = window_data[predictors]
        X = sm.add_constant(X)
        y = window_data[response]

        # Fit regression model
        model = sm.OLS(y, X).fit()

        # Extract results
        result['r2'] = model.rsquared
        for var in predictors:
            result[f'{var}_coeff'] = model.params[var]
            ci = model.conf_int().loc[var]
            result[f'{var}_ci_lower'] = ci[0]
            result[f'{var}_ci_upper'] = ci[1]

        result_list.append(result)

    result_df = pd.DataFrame(result_list)
    result_df['middle_year'] = pd.to_datetime(result_df['middle_year'])
    return result_df


# %%
result_pre_25_cru = calculate_annual_rolling_regression(df_normal, predictors=['pre_cru', 'tmp_cru'], response='CGR', window_years=25, min_samples=10)

result_pre_25_gpcc = calculate_annual_rolling_regression(df_normal, predictors=['pre_gpcc', 'tmp_cru'], response='CGR', window_years=25, min_samples=10)

result_tws_25_grace = calculate_annual_rolling_regression(df_normal, predictors=['tws_grace', 'tmp_cru'], response='CGR', window_years=25, min_samples=10)

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

# Define window results and styles
window_results = [
    (result_pre_25_cru, 'pre_cru', 'k', '25-year CRU'),
    (result_pre_25_gpcc, 'pre_gpcc', 'b', '25-year GPCC'),
    (result_tws_25_grace, 'tws_grace', 'r', '25-year TWS'),
]

# Loop through each result to plot
for result, var, color, window_label in window_results:
    # Plot the precipitation coefficient
    ax.plot(
        result['middle_year'], 
        result[var+'_coeff'],  
        '-o', ms=8, alpha=1, lw=2, 
        color=color,
        label=f'Regression with {window_label} window'
    )
    
    if color == 'r':
        # Plot the confidence interval
        ax.fill_between(
            result['middle_year'],
            result[var+'_ci_lower'],  
            result[var+'_ci_upper'], 
            lw=0, alpha=0.25, 
            color=color, 
            label='_nolegend_'
        )

# -------- Format
ax.legend(loc='ul', fontsize=8, ncols=1, frameon=False)

ax.format(
    xlabel='',
    ylim=(-0.020, 0.015),  
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2016-12-31')),
    xlocator=('year', 3),
    xformatter='%Y',
    xminorlocator=('year', 1),
    xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Regression coefficient of P on CGR', 
)


# %% [markdown]
# # Decompose correlation

# %%
def calculate_annual_rolling_std(df, variables, window_years=25, min_samples=10):
    """
    Calculate rolling standard deviation for one or more variables on annual data.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with annual data and datetime index
    variables : list of str
        Column names to compute rolling std for
    window_years : int
        Size of the rolling window in years (default: 25)
    min_samples : int
        Minimum non-NaN samples required (default: 10)

    Returns:
    --------
    pandas DataFrame
        DataFrame with columns: middle_year, {var}_std for each variable
    """
    num_windows = len(df) - window_years + 1
    result_list = []

    middle_indices = np.array([start_idx + window_years // 2 for start_idx in range(num_windows)])
    middle_years = df.index[middle_indices]

    for start_idx in range(num_windows):
        end_idx = start_idx + window_years
        window_data = df.iloc[start_idx:end_idx]

        result = {'middle_year': middle_years[start_idx]}
        for var in variables:
            col = window_data[var].dropna()
            result[f'{var}_std'] = col.std(ddof=1) if len(col) >= min_samples else np.nan

        result_list.append(result)

    result_df = pd.DataFrame(result_list)
    result_df['middle_year'] = pd.to_datetime(result_df['middle_year'])
    return result_df


# %% [markdown]
# ## Decompose: Corr = β × (σ_TWS / σ_CGR)
#
# The correlation identity allows us to decompose the trend in Corr(CGR, TWS) into two physically distinct drivers:
#
# - **β** (sensitivity): how strongly CGR responds per unit TWS change — reflects ecosystem processes  
# - **σ_TWS / σ_CGR** (variability ratio): amplitude of water forcing relative to carbon variability — reflects climate forcing

# %%
# ---- Step 1: Compute rolling std for TWS and CGR ----
result_std = calculate_annual_rolling_std(
    df_normal, variables=['tws_grace', 'CGR'], window_years=25, min_samples=10
)

# ---- Step 2: Compute β from simple OLS (CGR ~ TWS only) ----
# Note: use simple regression here so that β × (σ_TWS/σ_CGR) = Corr exactly
result_beta_tws = calculate_annual_rolling_regression(
    df_normal, predictors=['tws_grace'], response='CGR', window_years=25, min_samples=10
)
beta_df = result_beta_tws[['middle_year', 'tws_grace_coeff']].rename(columns={'tws_grace_coeff': 'beta'})

# ---- Step 3: Compute actual Corr(CGR, TWS) with Spearman ----
result_corr_tws = calculate_annual_rolling_correlation(
    df_normal, predictor='tws_grace', response='CGR', window_years=25, min_samples=10
)
corr_df = result_corr_tws[['middle_year', 'corr']].rename(columns={'corr': 'corr_actual'})

# ---- Merge all on middle_year ----
decomp = result_std.merge(beta_df, on='middle_year', how='inner').merge(corr_df, on='middle_year', how='inner')

# ---- Compute variability ratio and predicted correlation ----
decomp['sigma_ratio']    = decomp['tws_grace_std'] / decomp['CGR_std']
decomp['corr_predicted'] = decomp['beta'] * decomp['sigma_ratio']

decomp[['middle_year', 'beta', 'tws_grace_std', 'CGR_std', 'sigma_ratio', 'corr_predicted', 'corr_actual']].head(10)

# %% [markdown]
# ### Verification: does β × (σ_TWS / σ_CGR) ≈ Corr_actual?

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal='nat2')

ax.plot(
    decomp['middle_year'], decomp['corr_actual'],
    '-o', ms=6, lw=2, color='b', label='Corr actual (Spearman)',
)
ax.plot(
    decomp['middle_year'], decomp['corr_predicted'],
    '--', ms=0, lw=2, color='r', label=r'$\beta \times \sigma_{TWS}/\sigma_{CGR}$ (predicted)',
)

ax.format(
    xlabel='',
    xlim=(np.datetime64('1964-01-01'), np.datetime64('2013-12-31')),
    xlocator=('year', 3), xformatter='%Y', xminorlocator=('year', 1),
    xrotation=45, ygrid=True, xgrid=False,
    ylabel='Correlation (CGR ~ TWS)',
    title='Identity check: Corr = β × (σ_TWS / σ_CGR)',
)
ax.legend(loc='ll', fontsize=8, ncols=1, frameon=False)

# %% [markdown]
# ### Decomposition at each time point: Corr(t) = β(t) × σ_ratio(t)
#
# Around long-term means $\bar{\beta}$ and $\bar{\sigma}$, define $\Delta\beta(t) = \beta(t) - \bar{\beta}$ and $\Delta\sigma(t) = \sigma_{ratio}(t) - \bar{\sigma}$:
#
# $$\text{Corr}(t) = \underbrace{\bar{\beta}\,\bar{\sigma}}_{\text{baseline}} + \underbrace{\bar{\beta}\,\Delta\sigma(t)}_{\sigma\text{ contribution}} + \underbrace{\Delta\beta(t)\,\bar{\sigma}}_{\beta\text{ contribution}} + \underbrace{\Delta\beta(t)\,\Delta\sigma(t)}_{\text{interaction}}$$
#
# All four terms sum exactly to Corr(t) at every point.

# %%
from scipy import stats

# ---- Log-space static decomposition at each time point ----
# ln|Corr(t)| = ln|β(t)| + ln(σ_ratio(t))  — exact, unit-free

t = d['middle_year']
log_corr  = np.log(d['corr_actual'].abs())
log_beta  = np.log(d['beta'].abs())
log_sigma = np.log(d['sigma_ratio'])

# Verify identity
max_err = (log_corr - log_beta - log_sigma).abs().max()
print(f"Max identity error: {max_err:.2e}  (should be ~0)")

# ---- Linear trends ----
x = np.arange(len(d))
slope_corr,  icept_corr,  _, p_corr,  _ = stats.linregress(x, log_corr)
slope_beta,  icept_beta,  _, p_beta,  _ = stats.linregress(x, log_beta)
slope_sigma, icept_sigma, _, p_sigma, _ = stats.linregress(x, log_sigma)

# ---- Plot ----
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.2, journal='nat2')

ax.plot(t, log_corr,  'k-',  lw=2.5, label=r'$\ln|\mathrm{Corr}(t)|$')
ax.plot(t, log_beta,  '-',   lw=2.5, color='r',   label=r'$\ln|\beta(t)|$  (sensitivity)')
ax.plot(t, log_sigma, '-',   lw=2.5, color='b',   label=r'$\ln(\sigma_{\rm TWS}/\sigma_{\rm CGR})$  (variability ratio)')

# Trend lines (dashed)
ax.plot(t, slope_corr  * x + icept_corr,  'k--',  lw=1.2, alpha=0.7, label='_nolegend_')
ax.plot(t, slope_beta  * x + icept_beta,  '--',   lw=1.2, color='r', alpha=0.7, label='_nolegend_')
ax.plot(t, slope_sigma * x + icept_sigma, '--',   lw=1.2, color='b', alpha=0.7, label='_nolegend_')

ax.axhline(0, lw=0.8, color='gray', linestyle='-', alpha=0.3)

# Annotate trends
ax.text(0.02, 0.05,
        f'ln|Corr| trend: {slope_corr:+.4f}/window (p={p_corr:.3f})\n'
        f'ln|β|    trend: {slope_beta:+.4f}/window (p={p_beta:.3f})\n'
        f'ln(σ)   trend: {slope_sigma:+.4f}/window (p={p_sigma:.3f})',
        transform=ax.transAxes, va='bottom', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.format(
    xlabel='',
    xlim=(t.iloc[0], t.iloc[-1]),
    xlocator=('year', 3), xformatter='%Y', xminorlocator=('year', 1), xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'Log-space decomposition of $\ln|\mathrm{Corr}|$',
    title=r'$\ln|\mathrm{Corr}(t)| = \ln|\beta(t)| + \ln(\sigma_{\rm TWS}/\sigma_{\rm CGR})$',
)
ax.legend(loc='ur', fontsize=8, ncols=1, frameon=False)

# %%
from scipy import stats

# ---- Log-space static decomposition at each time point ----
# ln|Corr(t)| = ln|β(t)| + ln(σ_ratio(t))  — exact, unit-free

t = d['middle_year']
log_corr  = np.log(d['corr_actual'].abs())
log_beta  = np.log(d['beta'].abs())
log_sigma = np.log(d['sigma_ratio'])

# Verify identity
max_err = (log_corr - log_beta - log_sigma).abs().max()
print(f"Max identity error: {max_err:.2e}  (should be ~0)")

# ---- Linear trends ----
x = np.arange(len(d))
slope_corr,  icept_corr,  _, p_corr,  _ = stats.linregress(x, log_corr)
slope_beta,  icept_beta,  _, p_beta,  _ = stats.linregress(x, log_beta)
slope_sigma, icept_sigma, _, p_sigma, _ = stats.linregress(x, log_sigma)

# ---- Plot ----
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.2, journal='nat2')

ax.plot(t, log_corr,  'k-',  lw=2.5, label=r'$\ln|\mathrm{Corr}(t)|$')
ax.plot(t, log_beta,  '-',   lw=2.5, color='r',   label=r'$\ln|\beta(t)|$  (sensitivity)')
ax.plot(t, log_sigma, '-',   lw=2.5, color='b',   label=r'$\ln(\sigma_{\rm TWS}/\sigma_{\rm CGR})$  (variability ratio)')

# Trend lines (dashed)
ax.plot(t, slope_corr  * x + icept_corr,  'k--',  lw=1.2, alpha=0.7, label='_nolegend_')
ax.plot(t, slope_beta  * x + icept_beta,  '--',   lw=1.2, color='r', alpha=0.7, label='_nolegend_')
ax.plot(t, slope_sigma * x + icept_sigma, '--',   lw=1.2, color='b', alpha=0.7, label='_nolegend_')

ax.axhline(0, lw=0.8, color='gray', linestyle='-', alpha=0.3)

# Annotate trends
ax.text(0.02, 0.05,
        f'ln|Corr| trend: {slope_corr:+.4f}/window (p={p_corr:.3f})\n'
        f'ln|β|    trend: {slope_beta:+.4f}/window (p={p_beta:.3f})\n'
        f'ln(σ)   trend: {slope_sigma:+.4f}/window (p={p_sigma:.3f})',
        transform=ax.transAxes, va='bottom', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.format(
    xlabel='',
    xlim=(t.iloc[0], t.iloc[-1]),
    xlocator=('year', 3), xformatter='%Y', xminorlocator=('year', 1), xrotation=45,
    ygrid=False, xgrid=False,
    ylabel=r'Log-space decomposition of $\ln|\mathrm{Corr}|$',
    title=r'$\ln|\mathrm{Corr}(t)| = \ln|\beta(t)| + \ln(\sigma_{\rm TWS}/\sigma_{\rm CGR})$',
)
ax.legend(loc='ur', fontsize=8, ncols=1, frameon=False)

# %%

# %% [markdown]
# ### Relative contribution at each time point (%)
#
# At each point: divide each term by Corr(t) to get fractional contributions that sum to exactly 100%.
#
# - Positive = strengthening the coupling (same sign as Corr)  
# - Negative = dampening the coupling (opposite sign to Corr)

# %%
# ---- Relative contributions (%) at each time point ----
rel = d.copy()
for term in ['term_baseline', 'term_beta', 'term_sigma', 'term_interaction']:
    rel[term + '_pct'] = rel[term] / rel['corr_actual'] * 100

t = rel['middle_year']

fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.2, journal='nat2')

ax.stackplot(
    t,
    rel['term_baseline_pct'],
    rel['term_beta_pct'],
    rel['term_sigma_pct'],
    rel['term_interaction_pct'],
    labels=['Baseline ($\\bar{\\beta}\\,\\bar{\\sigma}$)',
            '$\\beta$ contribution',
            '$\\sigma$ contribution',
            'Interaction'],
    colors=['gray', 'r', 'b', 'orange'],
    alpha=0.75,
)

ax.axhline(0,   lw=0.8, color='k', linestyle='-',  alpha=0.4)
ax.axhline(100, lw=0.8, color='k', linestyle='--', alpha=0.3)

ax.format(
    xlabel='',
    xlim=(t.iloc[0], t.iloc[-1]),
    xlocator=('year', 3), xformatter='%Y', xminorlocator=('year', 1), xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Relative contribution (%)',
    title='Fractional contribution to Corr(CGR, TWS) at each time point',
)
ax.legend(loc='b', fontsize=8, ncols=5, frameon=False)

# %% [markdown]
# ### β vs σ: which drives the time-varying coupling?
#
# Focus on anomaly terms only (exclude constant baseline). Y-axis is in correlation units.

# %%
from scipy import stats

t = d['middle_year']
corr_anomaly = d['corr_actual'] - d['term_baseline']  # Corr(t) - baseline

fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.2, journal='nat2')

# --- Three anomaly terms ---
ax.plot(t, d['term_beta'],        '-', lw=2.5, color='r', label=r'$\beta$ contribution: $\Delta\beta\,\bar{\sigma}$')
ax.plot(t, d['term_sigma'],       '-', lw=2.5, color='b', label=r'$\sigma$ contribution: $\bar{\beta}\,\Delta\sigma$')
ax.plot(t, d['term_interaction'], '-', lw=1.5, color='orange', alpha=0.7, label='Interaction')
ax.plot(t, corr_anomaly,          'k-', lw=1.5, alpha=0.5, label='Total anomaly (Corr − baseline)')

ax.axhline(0, lw=0.8, color='k', linestyle='-', alpha=0.3)

# --- Overlay linear trends for β and σ terms ---
x = np.arange(len(d))
for y_vals, color in [(d['term_beta'], 'r'), (d['term_sigma'], 'b')]:
    slope, intercept, *_ = stats.linregress(x, y_vals)
    trend = slope * x + intercept
    ax.plot(t, trend, '--', lw=1.5, color=color, alpha=0.8)

# --- Annotate trend slopes ---
slope_beta,  _, _, p_beta,  _ = stats.linregress(x, d['term_beta'])
slope_sigma, _, _, p_sigma, _ = stats.linregress(x, d['term_sigma'])

ax.text(0.02, 0.97,
        f'β trend:  {slope_beta:+.4f}/window  (p={p_beta:.3f})\n'
        f'σ trend: {slope_sigma:+.4f}/window  (p={p_sigma:.3f})',
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.format(
    xlabel='',
    xlim=(t.iloc[0], t.iloc[-1]),
    xlocator=('year', 3), xformatter='%Y', xminorlocator=('year', 1), xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Contribution to Corr (correlation units)',
    title='Time-varying contributions of β and σ to CGR–TWS coupling',
)
ax.legend(loc='b', fontsize=8, ncols=5, frameon=False)

# %% [markdown]
# # Another Decomposition: Univariate (no temperature control)
#
# Using simple OLS (CGR ~ TWS only) gives an **exact** identity at every point:
#
# $$\text{Corr}_{\text{Pearson}}(\text{CGR}, \text{TWS}) \equiv \beta_{\text{simple}} \times \frac{\sigma_{\text{TWS}}}{\sigma_{\text{CGR}}}$$
#
# No approximation, no residual. This lets us cleanly attribute every unit of coupling to either ecosystem sensitivity (β) or climate variability (σ ratio).

# %%
from scipy import stats

# ---- Self-contained: reuse functions but not previous results ----
WINDOW = 25

# Annual data (already loaded as df_normal from Liu section)
# Pearson correlation — exact identity with simple OLS β
result_corr  = calculate_annual_rolling_correlation(
    df_normal, predictor='tws_grace', response='CGR',
    window_years=WINDOW, min_samples=10
)
# Change method to Pearson for exact identity
# (calculate_annual_rolling_correlation uses Spearman by default — recompute with Pearson)
num_windows = len(df_normal) - WINDOW + 1
pearson_list = []
for i in range(num_windows):
    window = df_normal.iloc[i:i+WINDOW][['tws_grace', 'CGR']].dropna()
    mid    = df_normal.index[i + WINDOW // 2]
    if len(window) >= 10:
        r, p = stats.pearsonr(window['tws_grace'], window['CGR'])
    else:
        r, p = np.nan, np.nan
    pearson_list.append({'middle_year': pd.to_datetime(mid), 'corr': r, 'p': p})
result_corr_pearson = pd.DataFrame(pearson_list)

# Simple OLS: CGR ~ TWS only (no temperature)
result_beta = calculate_annual_rolling_regression(
    df_normal, predictors=['tws_grace'], response='CGR',
    window_years=WINDOW, min_samples=10
)

# Rolling std for TWS and CGR
result_std = calculate_annual_rolling_std(
    df_normal, variables=['tws_grace', 'CGR'],
    window_years=WINDOW, min_samples=10
)

# ---- Merge ----
d2 = (result_std
      .merge(result_beta[['middle_year', 'tws_grace_coeff']], on='middle_year', how='inner')
      .merge(result_corr_pearson[['middle_year', 'corr']], on='middle_year', how='inner')
      .rename(columns={'tws_grace_coeff': 'beta', 'corr': 'corr_pearson'})
      .dropna())

# ---- Compute components ----
d2['sigma_ratio']    = d2['tws_grace_std'] / d2['CGR_std']
d2['corr_predicted'] = d2['beta'] * d2['sigma_ratio']

# Verify exact identity
max_err = (d2['corr_predicted'] - d2['corr_pearson']).abs().max()
print(f"Max identity error (Pearson ≡ β × σ_ratio): {max_err:.2e}  ✓ if < 1e-10")

# ---- Point-wise decomposition ----
beta_mean  = d2['beta'].mean()
sigma_mean = d2['sigma_ratio'].mean()

d2['delta_beta']  = d2['beta']        - beta_mean
d2['delta_sigma'] = d2['sigma_ratio'] - sigma_mean

d2['term_baseline']    = beta_mean * sigma_mean
d2['term_beta']        = d2['delta_beta']  * sigma_mean
d2['term_sigma']       = beta_mean         * d2['delta_sigma']
d2['term_interaction'] = d2['delta_beta']  * d2['delta_sigma']

# %%
t2 = d2['middle_year']
corr_anomaly2 = d2['corr_pearson'] - d2['term_baseline']

fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.2, journal='nat2')

ax.plot(t2, d2['term_beta'],        '-', lw=2.5, color='r',      label=r'$\beta$ contribution: $\Delta\beta\,\bar{\sigma}$')
ax.plot(t2, d2['term_sigma'],       '-', lw=2.5, color='b',      label=r'$\sigma$ contribution: $\bar{\beta}\,\Delta\sigma$')
ax.plot(t2, d2['term_interaction'], '-', lw=1.5, color='orange', alpha=0.7, label='Interaction')
ax.plot(t2, corr_anomaly2,          'k-', lw=1.5, alpha=0.5,    label='Total anomaly')

ax.axhline(0, lw=0.8, color='k', linestyle='-', alpha=0.3)

# Linear trends with annotations
x2 = np.arange(len(d2))
for y_vals, color, name in [
    (d2['term_beta'],  'r', 'β'),
    (d2['term_sigma'], 'b', 'σ'),
]:
    slope, intercept, _, p, _ = stats.linregress(x2, y_vals)
    ax.plot(t2, slope * x2 + intercept, '--', lw=1.5, color=color, alpha=0.8)

slope_b, _, _, p_b, _ = stats.linregress(x2, d2['term_beta'])
slope_s, _, _, p_s, _ = stats.linregress(x2, d2['term_sigma'])

ax.text(0.02, 0.97,
        f'β trend:  {slope_b:+.4f}/window  (p={p_b:.3f})\n'
        f'σ trend: {slope_s:+.4f}/window  (p={p_s:.3f})',
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.format(
    xlabel='',
    xlim=(t2.iloc[0], t2.iloc[-1]),
    xlocator=('year', 3), xformatter='%Y', xminorlocator=('year', 1), xrotation=45,
    ygrid=False, xgrid=False,
    ylabel='Contribution to Corr (Pearson, correlation units)',
    title='Univariate decomposition: β vs σ (no temperature control)',
)
ax.legend(loc='ll', fontsize=8, ncols=2, frameon=False)

# %% [markdown]
# # Model representation

# %%
# Load CMIP MRSO
ts_mrso_cmip = pd.read_csv(proj_path+'/'+'01-data/mrso_CMIP_processed.csv', index_col=0) 
ts_mrso_cmip.index = pd.to_datetime(ts_mrso_cmip.index)

ts_nbp_cmip = pd.read_csv(proj_path+'/'+'01-data/nbp_CMIP_processed.csv', index_col=0)

# %%
# Q1: Rolling correlation NBP ~ MRSO for each CMIP6 model
# Note: -NBP ≈ CGR (both increase when biosphere releases carbon)
# so Corr(-NBP, MRSO) is comparable to Corr(CGR, TWS) from observations

WINDOW = 25  # years

models = ts_mrso_cmip.columns.tolist()

# ---- Ensure DatetimeIndex (ts_nbp_cmip may not have been converted) ----
ts_mrso_cmip.index = pd.to_datetime(ts_mrso_cmip.index)
ts_nbp_cmip.index  = pd.to_datetime(ts_nbp_cmip.index)

# ---- Monthly → Annual mean ----
mrso_annual = ts_mrso_cmip.resample('YE').mean()
nbp_annual  = ts_nbp_cmip.resample('YE').mean()
mrso_annual.index = pd.to_datetime(mrso_annual.index.year.astype(str) + '-01-01')
nbp_annual.index  = pd.to_datetime(nbp_annual.index.year.astype(str)  + '-01-01')

# ---- Rolling Pearson correlation for each model ----
corr_models = {}   # model -> DataFrame with columns: middle_year, corr

for model in models:
    df_m = pd.DataFrame({
        'mrso': mrso_annual[model],
        'nbp':  -nbp_annual[model],   # flip sign → comparable to CGR
    }).dropna()

    num_windows = len(df_m) - WINDOW + 1
    rows = []
    for i in range(num_windows):
        win = df_m.iloc[i:i+WINDOW].dropna()
        mid = df_m.index[i + WINDOW // 2]
        if len(win) >= 10:
            r, p = stats.pearsonr(win['mrso'], win['nbp'])
        else:
            r, p = np.nan, np.nan
        rows.append({'middle_year': mid, 'corr': r, 'p': p})

    corr_models[model] = pd.DataFrame(rows)

print(f"Done: {len(models)} models processed")
print("Example (CanESM5):", corr_models['CanESM5'][['middle_year','corr']].dropna().head(3).to_string(index=False))

# %%
# ---- Observed rolling correlation (Pearson, for consistency with models) ----
obs_rows = []
df_obs = df_normal[['tws_grace', 'CGR']].dropna()
num_windows_obs = len(df_obs) - WINDOW + 1
for i in range(num_windows_obs):
    win = df_obs.iloc[i:i+WINDOW].dropna()
    mid = df_obs.index[i + WINDOW // 2]
    if len(win) >= 10:
        r, p = stats.pearsonr(win['tws_grace'], win['CGR'])
    else:
        r, p = np.nan, np.nan
    obs_rows.append({'middle_year': pd.to_datetime(mid), 'corr': r})
corr_obs = pd.DataFrame(obs_rows).dropna()

# ---- Build ensemble arrays ----
all_corr = pd.concat(
    {m: corr_models[m].set_index('middle_year')['corr'] for m in models},
    axis=1
)
ens_mean = all_corr.mean(axis=1)
ens_p10  = all_corr.quantile(0.10, axis=1)
ens_p90  = all_corr.quantile(0.90, axis=1)

# ---- Plot ----
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2.2, journal='nat2')

for model in models:
    d_m = corr_models[model].dropna()
    ax.plot(d_m['middle_year'], d_m['corr'],
            '-', lw=0.8, color='steelblue', alpha=0.3, label='_nolegend_')

ax.fill_between(ens_mean.index, ens_p10, ens_p90,
                color='steelblue', alpha=0.25, lw=0, label='Model 10–90%')

ax.plot(ens_mean.index, ens_mean,
        '-', lw=2.5, color='steelblue', label='Model ensemble mean')

ax.plot(corr_obs['middle_year'], corr_obs['corr'],
        '-', lw=2.5, color='k', label='Observation (TWS–CGR)')

ax.axhline(0, lw=0.8, color='gray', linestyle='--', alpha=0.5)

ax.format(
    xlabel='',
    xlim=(np.datetime64('1970-01-01'), np.datetime64('2014-01-01')),
    xlocator=('year', 5), xformatter='%Y', xminorlocator=('year', 1), xrotation=45,
    ylim=(-1,0),
    ygrid=False, xgrid=False,
    ylabel='Pearson Correlation (MRSO ~ −NBP  /  TWS ~ CGR)',
    title=f'Rolling {WINDOW}-year correlation: models vs observation',
)
ax.legend(loc='ll', fontsize=8, ncols=1, frameon=False)

# %% [markdown]
# ## Q2: Decompose — is the model bias from β or σ?
#
# Since Corr = β × (σ_MRSO / σ_NBP), we compare each component separately:
# - **β**: Does the model capture the ecosystem sensitivity trend?
# - **σ ratio**: Does the model capture the variability ratio trend?

# %%
from scipy import stats

# ---- Rolling β and σ for each model ----
beta_models     = {}   # model -> DataFrame: middle_year, beta
sigma_models    = {}   # model -> DataFrame: middle_year, sigma_mrso, sigma_nbp, sigma_ratio

for model in models:
    df_m = pd.DataFrame({
        'mrso': mrso_annual[model],
        'nbp':  -nbp_annual[model],
    }).dropna()

    num_windows = len(df_m) - WINDOW + 1
    beta_rows  = []
    sigma_rows = []

    for i in range(num_windows):
        win = df_m.iloc[i:i+WINDOW].dropna()
        mid = df_m.index[i + WINDOW // 2]

        beta_row  = {'middle_year': mid, 'beta': np.nan}
        sigma_row = {'middle_year': mid, 'sigma_mrso': np.nan, 'sigma_nbp': np.nan, 'sigma_ratio': np.nan}

        if len(win) >= 10:
            # β: simple OLS -NBP ~ MRSO
            slope, _, _, _, _ = stats.linregress(win['mrso'], win['nbp'])
            beta_row['beta'] = slope

            # σ
            s_mrso = win['mrso'].std(ddof=1)
            s_nbp  = win['nbp'].std(ddof=1)
            sigma_row['sigma_mrso']  = s_mrso
            sigma_row['sigma_nbp']   = s_nbp
            sigma_row['sigma_ratio'] = s_mrso / s_nbp if s_nbp > 0 else np.nan

        beta_rows.append(beta_row)
        sigma_rows.append(sigma_row)

    beta_models[model]  = pd.DataFrame(beta_rows)
    sigma_models[model] = pd.DataFrame(sigma_rows)

print("Done.")

# ---- Observed β and σ (from d2 computed in univariate decomposition) ----
obs_beta  = d2[['middle_year', 'beta']].copy()
obs_sigma = d2[['middle_year', 'sigma_ratio', 'tws_grace_std', 'CGR_std']].copy()

# ---- Build ensemble DataFrames ----
def ensemble_stats(model_dict, col):
    df = pd.concat(
        {m: model_dict[m].set_index('middle_year')[col] for m in models},
        axis=1
    )
    return df.mean(axis=1), df.quantile(0.10, axis=1), df.quantile(0.90, axis=1)

ens_beta_mean,  ens_beta_p10,  ens_beta_p90  = ensemble_stats(beta_models,  'beta')
ens_sigma_mean, ens_sigma_p10, ens_sigma_p90 = ensemble_stats(sigma_models, 'sigma_ratio')

# %%
fig, axes = pplt.subplots(ncols=1, nrows=2, refaspect=2.2, journal='nat2', share=False, hspace=5)

XLIM = (np.datetime64('1970-01-01'), np.datetime64('2014-01-01'))
FMT  = dict(xlocator=('year', 5), xformatter='%Y', xminorlocator=('year', 1),
            xrotation=45, ygrid=False, xgrid=False, xlabel='')

# ============================================================
# Panel A: β (sensitivity)
# ============================================================
ax1 = axes[0]

for model in models:
    d_m = beta_models[model].dropna()
    ax1.plot(d_m['middle_year'], d_m['beta'],
             '-', lw=0.8, color='steelblue', alpha=0.3, label='_nolegend_')

ax1.fill_between(ens_beta_mean.index, ens_beta_p10, ens_beta_p90,
                 color='steelblue', alpha=0.25, lw=0, label='Model 10–90%')
ax1.plot(ens_beta_mean.index, ens_beta_mean,
         '-', lw=2.5, color='steelblue', label='Model ensemble mean')
ax1.plot(obs_beta['middle_year'], obs_beta['beta'],
         '-', lw=2.5, color='k', label='Observation')

ax1.axhline(0, lw=0.8, color='gray', linestyle='--', alpha=0.5)
ax1.format(xlim=XLIM, ylabel=r'Sensitivity $\beta$ (−NBP / MRSO  or  CGR / TWS)',
           title='(a) Ecosystem sensitivity β', **FMT)
ax1.legend(loc='ll', fontsize=8, ncols=1, frameon=False,)

# ============================================================
# Panel B: σ ratio
# ============================================================
ax2 = axes[1]

for model in models:
    d_m = sigma_models[model].dropna()
    ax2.plot(d_m['middle_year'], d_m['sigma_ratio'],
             '-', lw=0.8, color='darkorange', alpha=0.3, label='_nolegend_')

ax2.fill_between(ens_sigma_mean.index, ens_sigma_p10, ens_sigma_p90,
                 color='darkorange', alpha=0.25, lw=0, label='Model 10–90%')
ax2.plot(ens_sigma_mean.index, ens_sigma_mean,
         '-', lw=2.5, color='darkorange', label='Model ensemble mean')
ax2.plot(obs_sigma['middle_year'], obs_sigma['sigma_ratio'],
         '-', lw=2.5, color='k', label='Observation')

ax2.axhline(0, lw=0.8, color='gray', linestyle='--', alpha=0.5)
ax2.format(xlim=XLIM, ylabel=r'Variability ratio $\sigma_{MRSO} / \sigma_{NBP}$',
           title='(b) Variability ratio σ', **FMT)
ax2.legend(loc='ll', fontsize=8, ncols=1, frameon=False)


# %% [markdown]
# ## Q2 (Method A): Normalized comparison — β and σ trends
#
# Normalize each component by its own long-term mean: β_norm(t) = β(t) / |β̄|, σ_norm(t) = σ_ratio(t) / σ̄_ratio
#
# - Values > 1: stronger than long-term average
# - Values < 1: weaker than long-term average  
# - Trend direction and relative magnitude are directly comparable across obs and models

# %%
def normalize_series(s):
    """Divide by absolute long-term mean → dimensionless relative change."""
    return s / s.abs().mean()

# ---- Normalize each model's β and σ_ratio ----
beta_norm_models  = {}
sigma_norm_models = {}

for model in models:
    b = beta_models[model].set_index('middle_year')['beta'].dropna()
    s = sigma_models[model].set_index('middle_year')['sigma_ratio'].dropna()
    beta_norm_models[model]  = normalize_series(b)
    sigma_norm_models[model] = normalize_series(s)

# ---- Ensemble stats on normalized series ----
def ens_from_dict(d):
    df = pd.concat(d, axis=1)
    return df.mean(axis=1), df.quantile(0.10, axis=1), df.quantile(0.90, axis=1)

ens_beta_norm_mean,  ens_beta_norm_p10,  ens_beta_norm_p90  = ens_from_dict(beta_norm_models)
ens_sigma_norm_mean, ens_sigma_norm_p10, ens_sigma_norm_p90 = ens_from_dict(sigma_norm_models)

# ---- Normalize observations ----
obs_beta_norm  = normalize_series(obs_beta.set_index('middle_year')['beta'])
obs_sigma_norm = normalize_series(obs_sigma.set_index('middle_year')['sigma_ratio'])

# ---- Plot ----
fig, axes = pplt.subplots(ncols=1, nrows=2, refaspect=2.2, journal='nat2', share=False, hspace=5)

XLIM = (np.datetime64('1970-01-01'), np.datetime64('2014-01-01'))
FMT  = dict(xlocator=('year', 5), xformatter='%Y', xminorlocator=('year', 1),
            xrotation=45, ygrid=False, xgrid=False, xlabel='')

for ax, ens_mean, ens_p10, ens_p90, norm_models, obs_norm, color, title, ylabel in [
    (axes[0],
     ens_beta_norm_mean,  ens_beta_norm_p10,  ens_beta_norm_p90,
     beta_norm_models,  obs_beta_norm,
     'steelblue', r'(a) Sensitivity $\beta$ (normalized)', r'$\beta(t)\,/\,|\bar{\beta}|$'),
    (axes[1],
     ens_sigma_norm_mean, ens_sigma_norm_p10, ens_sigma_norm_p90,
     sigma_norm_models, obs_sigma_norm,
     'darkorange', r'(b) Variability ratio $\sigma$ (normalized)', r'$\sigma_{ratio}(t)\,/\,\bar{\sigma}_{ratio}$'),
]:
    # Individual models
    for model in models:
        s = norm_models[model]
        ax.plot(s.index, s.values, '-', lw=0.8, color=color, alpha=0.25, label='_nolegend_')

    # Ensemble spread and mean
    ax.fill_between(ens_mean.index, ens_p10, ens_p90,
                    color=color, alpha=0.25, lw=0, label='Model 10–90%')
    ax.plot(ens_mean.index, ens_mean,
            '-', lw=2.5, color=color, label='Model ensemble mean')

    # Observation
    ax.plot(obs_norm.index, obs_norm.values,
            '-', lw=2.5, color='k', label='Observation')

    ax.axhline(1, lw=0.8, color='gray', linestyle='--', alpha=0.6)

    ax.format(xlim=XLIM, ylabel=ylabel, title=title, **FMT)
    ax.legend(loc='ll', fontsize=8, ncols=1, frameon=False)

# %% [markdown]
# ## Q3: Point-wise decomposition for each model
#
# Apply the same decomposition as observations to each CMIP6 model:
#
# $$\text{Corr}(t) = \bar{\beta}\bar{\sigma} + \Delta\beta\,\bar{\sigma} + \bar{\beta}\,\Delta\sigma + \Delta\beta\,\Delta\sigma$$
#
# Then express β and σ contributions as % of Corr(t) at each point.  
# Compare the ensemble-mean contribution profiles against observations.

# %%
# ---- Point-wise decomposition for each model ----
# For each model: merge corr, beta, sigma_ratio → compute 4 terms → compute % contributions

decomp_pct_models = {}   # model -> DataFrame with middle_year, pct_beta, pct_sigma, pct_interaction

for model in models:
    # Merge the three rolling quantities on middle_year
    corr_s  = corr_models[model].set_index('middle_year')['corr']
    beta_s  = beta_models[model].set_index('middle_year')['beta']
    sigma_s = sigma_models[model].set_index('middle_year')['sigma_ratio']

    dm = pd.DataFrame({'corr': corr_s, 'beta': beta_s, 'sigma_ratio': sigma_s}).dropna()
    if len(dm) < 5:
        continue

    # Long-term means
    b_mean = dm['beta'].mean()
    s_mean = dm['sigma_ratio'].mean()

    # Anomalies
    dm['delta_beta']  = dm['beta']        - b_mean
    dm['delta_sigma'] = dm['sigma_ratio'] - s_mean

    # Four terms
    dm['term_baseline']    = b_mean           * s_mean
    dm['term_beta']        = dm['delta_beta'] * s_mean
    dm['term_sigma']       = b_mean           * dm['delta_sigma']
    dm['term_interaction'] = dm['delta_beta'] * dm['delta_sigma']

    # % contributions
    dm['pct_beta']        = dm['term_beta']        / dm['corr'] * 100
    dm['pct_sigma']       = dm['term_sigma']        / dm['corr'] * 100
    dm['pct_interaction'] = dm['term_interaction']  / dm['corr'] * 100
    dm['pct_baseline']    = dm['term_baseline']     / dm['corr'] * 100

    decomp_pct_models[model] = dm[['pct_beta', 'pct_sigma', 'pct_interaction', 'pct_baseline']]

# ---- Ensemble mean of % contributions ----
def ens_pct(col):
    df = pd.concat({m: decomp_pct_models[m][col] for m in decomp_pct_models}, axis=1)
    return df.mean(axis=1), df.quantile(0.10, axis=1), df.quantile(0.90, axis=1)

ens_pct_beta_mean,  ens_pct_beta_p10,  ens_pct_beta_p90  = ens_pct('pct_beta')
ens_pct_sigma_mean, ens_pct_sigma_p10, ens_pct_sigma_p90 = ens_pct('pct_sigma')
ens_pct_int_mean,   ens_pct_int_p10,   ens_pct_int_p90   = ens_pct('pct_interaction')

# ---- Observed % contributions (from d2) ----
obs_pct_beta        = d2['term_beta']        / d2['corr_pearson'] * 100
obs_pct_sigma       = d2['term_sigma']        / d2['corr_pearson'] * 100
obs_pct_interaction = d2['term_interaction']  / d2['corr_pearson'] * 100
obs_t = d2['middle_year']

print("Obs mean contributions:   β={:.1f}%  σ={:.1f}%  interaction={:.1f}%".format(
    obs_pct_beta.mean(), obs_pct_sigma.mean(), obs_pct_interaction.mean()))
print("Model ens mean (mean t):  β={:.1f}%  σ={:.1f}%  interaction={:.1f}%".format(
    ens_pct_beta_mean.mean(), ens_pct_sigma_mean.mean(), ens_pct_int_mean.mean()))

# %%
fig, axes = pplt.subplots(ncols=1, nrows=2, refaspect=2.2, journal='nat2', share=False, hspace=0.6)

XLIM = (np.datetime64('1970-01-01'), np.datetime64('2014-01-01'))
FMT  = dict(xlocator=('year', 5), xformatter='%Y', xminorlocator=('year', 1),
            xrotation=45, ygrid=False, xgrid=False, xlabel='')

for ax, ens_mean, ens_p10, ens_p90, obs_pct, color, title, ylabel in [
    (axes[0],
     ens_pct_beta_mean,  ens_pct_beta_p10,  ens_pct_beta_p90,
     obs_pct_beta,  'steelblue',
     r'(a) $\beta$ contribution (%)',
     r'$\beta$ contribution to Corr (%)'),
    (axes[1],
     ens_pct_sigma_mean, ens_pct_sigma_p10, ens_pct_sigma_p90,
     obs_pct_sigma, 'darkorange',
     r'(b) $\sigma$ contribution (%)',
     r'$\sigma$ contribution to Corr (%)'),
]:
    ax.fill_between(ens_mean.index, ens_p10, ens_p90,
                    color=color, alpha=0.25, lw=0, label='Model 10–90%')
    ax.plot(ens_mean.index, ens_mean,
            '-', lw=2.5, color=color, label='Model ensemble mean')
    ax.plot(obs_t, obs_pct.values,
            '-', lw=2.5, color='k', label='Observation')

    ax.axhline(0, lw=0.8, color='gray', linestyle='--', alpha=0.5)

    ax.format(xlim=XLIM, ylabel=ylabel, title=title, **FMT)
    ax.legend(loc='ll', fontsize=8, ncols=1, frameon=False)


# %% [markdown]
# ## Log-space trend decomposition: obs vs each model
#
# $$\Delta\ln|\text{Corr}| = \Delta\ln|\beta| + \Delta\ln(\sigma_{\text{ratio}})$$
#
# Fit linear trends on log-transformed series → fractional contribution of β and σ to the total trend in |Corr|.

# %%
def log_decomp(corr_s, beta_s, sigma_s):
    """
    Log-space decomposition of trend in |Corr|.
    Returns (contrib_beta_pct, contrib_sigma_pct, slope_corr, p_corr)
    """
    dm = pd.DataFrame({
        'corr':        corr_s,
        'beta':        beta_s,
        'sigma_ratio': sigma_s,
    }).dropna()

    if len(dm) < 10:
        return np.nan, np.nan, np.nan, np.nan

    dm['log_corr']  = np.log(dm['corr'].abs())
    dm['log_beta']  = np.log(dm['beta'].abs())
    dm['log_sigma'] = np.log(dm['sigma_ratio'])

    x = np.arange(len(dm))
    slope_corr,  _, _, p_corr,  _ = stats.linregress(x, dm['log_corr'])
    slope_beta,  _, _, p_beta,  _ = stats.linregress(x, dm['log_beta'])
    slope_sigma, _, _, p_sigma, _ = stats.linregress(x, dm['log_sigma'])

    if slope_corr == 0:
        return np.nan, np.nan, slope_corr, p_corr

    contrib_beta  = slope_beta  / slope_corr * 100
    contrib_sigma = slope_sigma / slope_corr * 100
    return contrib_beta, contrib_sigma, slope_corr, p_corr


# ---- Observations (using d2: pure Pearson + simple OLS) ----
obs_cb, obs_cs, obs_slope, obs_p = log_decomp(
    d2['corr_pearson'],
    d2['beta'],
    d2['sigma_ratio'],
)
print(f"Observation:  β={obs_cb:+.1f}%  σ={obs_cs:+.1f}%  "
      f"(total trend slope={obs_slope:+.4f}, p={obs_p:.3f})")

# ---- Each CMIP6 model ----
results = []
for model in models:
    corr_s  = corr_models[model].set_index('middle_year')['corr']
    beta_s  = beta_models[model].set_index('middle_year')['beta']
    sigma_s = sigma_models[model].set_index('middle_year')['sigma_ratio']

    cb, cs, slope, p = log_decomp(corr_s, beta_s, sigma_s)
    results.append({'model': model, 'contrib_beta': cb, 'contrib_sigma': cs,
                    'slope_corr': slope, 'p_corr': p})

df_results = pd.DataFrame(results).set_index('model')
print("\nModel log-decomposition results:")
print(df_results[['contrib_beta', 'contrib_sigma', 'slope_corr', 'p_corr']].round(1).to_string())

# %%
# ---- Bar chart: obs + each model ----
labels  = ['OBS'] + df_results.index.tolist()
cb_vals = [obs_cb] + df_results['contrib_beta'].tolist()
cs_vals = [obs_cs] + df_results['contrib_sigma'].tolist()

x_pos = np.arange(len(labels))
width = 0.38

fig, ax = pplt.subplots(refaspect=3.5, journal='nat2')

bars_b = ax.bar(x_pos - width/2, cb_vals, width=width,
                color='steelblue', alpha=0.85, label=r'$\beta$ contribution (%)')
bars_s = ax.bar(x_pos + width/2, cs_vals, width=width,
                color='darkorange', alpha=0.85, label=r'$\sigma$ contribution (%)')

# Highlight observation bar edges
ax.bar(x_pos[0] - width/2, cb_vals[0], width=width,
       color='steelblue', edgecolor='k', lw=1.5, label='_nolegend_')
ax.bar(x_pos[0] + width/2, cs_vals[0], width=width,
       color='darkorange', edgecolor='k', lw=1.5, label='_nolegend_')

ax.axhline(0,   lw=0.8, color='k',    linestyle='-',  alpha=0.3)
ax.axhline(100, lw=0.8, color='gray', linestyle='--', alpha=0.5)

# Ensemble mean lines
ens_cb_mean = df_results['contrib_beta'].mean()
ens_cs_mean = df_results['contrib_sigma'].mean()
ax.axhline(ens_cb_mean,  lw=1.5, color='steelblue',  linestyle='--', alpha=0.8)
ax.axhline(ens_cs_mean,  lw=1.5, color='darkorange', linestyle='--', alpha=0.8)

ax.format(
    xticks=x_pos,
    xticklabels=labels,
    xrotation=45,
    ylabel='Contribution to trend in ln|Corr| (%)',
    title='Log-space decomposition: β vs σ contribution to CGR–water coupling trend',
    ygrid=False, xgrid=False,
)
ax.legend(loc='ur', fontsize=8, ncols=1, frameon=False)

# %%
