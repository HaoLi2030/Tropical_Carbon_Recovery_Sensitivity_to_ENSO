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
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# ===============================
#          Load packages
# ===============================

# Numerical computing
import pandas as pd
import numpy as np

# Scientific plotting
import proplot as pplt
import seaborn as sns

# Arial font for plots
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import stats

plt.rcParams["font.family"] = "Arial"
# Statistical analysis
import pingouin as pg
import statsmodels.api as sm

# Warnings and errors
import warnings

warnings.filterwarnings("ignore")
# Xarray utilities for temporal trends
import xarray as xr
from xarrayutils.utils import linear_trend

# Path operations
import os
import glob

# %% [markdown]
# # Setup of the Project

# %%
# Set project path
proj_path = '/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate'
os.chdir(proj_path)

# %% [markdown]
# ## Carbon and climate in observations

# %%
## Load Climate-Carbon Dataset

# Load tropical monthly climate and carbon data
df_all = pd.read_csv('01-data/tropical_month_climate_carbon_atlas_mean.csv', index_col=0)
df_all.index = pd.to_datetime(df_all.index)

# %%
## Data Filtering

# Analysis period: 1959-03-01 to 2019-07-31
# Rationale:
# - CGR data starts from 1959-03-01 (data preprocessing limitation)
# - GRACE TWS data ends at 2019-07-31 (data availability limitation)

start_date = '1959-03-01'
end_date = '2019-07-31'

df_all = df_all.loc[start_date:end_date]

# Create dataset excluding volcanic eruption periods
df_normal = df_all.mask(df_all['eruption'] == 1)

print(f"Total months: {len(df_all)}")
print(f"Eruption months: {(df_all['eruption'] == 1).sum()}")
print(f"Normal months: {(df_all['eruption'] == 0).sum()}")

# %%
# CGR observations
ts_cgr_obs = df_all['CGR']
# SST observations
ts_sst_obs = df_all['nina34']

# %%
corr_obs = pg.corr(ts_cgr_obs, ts_sst_obs, method='pearson', alpha=0.05)

print(corr_obs)


# %%
# ===============================
# Statistical Analysis Functions
# ===============================

# ------ 1. Pearson Correlation
def calculate_correlation(X_data, Y_data, min_samples=10, alpha=0.05):
    """
    Calculate Pearson correlation between X and Y variables

    Parameters:
    -----------
    X_data : array-like
        Predictor variable(s)
    Y_data : array-like
        Response variable
    conf_level : float, optional
        Confidence level for intervals (default=0.95)
    min_samples : int, optional
        Minimum valid data points required (default=10)

    Returns:
    --------
    dict or None
        - 'r': correlation coefficient
        - 'p-val': p-value for significance
        - 'CI95%': confidence interval [lower, upper]
        - 'n_samples': number of valid points
        Returns None if insufficient data
    """
    # Input validation for the same size
    if len(X_data) != len(Y_data) or len(X_data) < min_samples:
        return None

    # Remove NaN values and ensure paired data
    valid_mask = ~(np.isnan(X_data.values) | np.isnan(Y_data.values))
    X_clean, Y_clean = X_data[valid_mask], Y_data[valid_mask]

    try:
        result = pg.corr(X_clean, Y_clean, method="pearson", alpha=alpha)
        return {
            "r": result["r"].values[0],
            "p-val": result["p-val"].values[0],
            "CI95%": [result["CI95%"][0][0], result["CI95%"][0][1]],
            "n_samples": len(X_clean),
        }
    except Exception as e:
        print(f"Correlation error: {e}")
        return None

# ------ 2. Linear Regression
def calculate_regression(X_data, Y_data, conf_level=0.95, min_samples=10):
    """
    Calculate linear regression: Y = β0 + β1·X + ε

    Parameters:
    -----------
    X_data : array-like
        Predictor variable
    Y_data : array-like
        Response variable(s)
    conf_level : float, optional
        Confidence level for intervals (default=0.95)
    min_samples : int, optional
        Minimum valid data points required (default=10)

    Returns:
    --------
    dict
        - 'coef': regression slope coefficient
        - 'pvalue': p-value for slope significance
        - 'r2': coefficient of determination
        - 'ci_lower': lower confidence bound for slope
        - 'ci_upper': upper confidence bound for slope
        - 'n_samples': number of valid points
        Returns NaN values if calculation fails
    """
    # Input validation
    if len(X_data) != len(Y_data) or len(X_data) < min_samples:
        return {
            k: np.nan
            for k in ["coef", "pvalue", "r2", "ci_lower", "ci_upper", "n_samples"]
        }

    # Remove NaN values and ensure paired data
    valid_mask = ~(np.isnan(X_data) | np.isnan(Y_data))
    X_clean, Y_clean = X_data[valid_mask], Y_data[valid_mask]

    try:
        X_with_const = sm.add_constant(X_clean)
        model = sm.OLS(Y_clean, X_with_const).fit()

        if len(model.params) >= 2:
            alpha = 1 - conf_level
            conf_int = model.conf_int(alpha=alpha)
            return {
                "coef": model.params[1],  # slope
                "pvalue": model.pvalues[1],  # p-value for slope
                "r2": model.rsquared,  # R-squared
                "ci_lower": conf_int[1, 0],  # lower CI
                "ci_upper": conf_int[1, 1],  # upper CI
                "n_samples": len(X_clean),  # sample size
            }
    except Exception as e:
        print(f"Regression error: {e}")

    return {
        k: np.nan for k in ["coef", "pvalue", "r2", "ci_lower", "ci_upper", "n_samples"]
    }


# %%

# %% [markdown]
# ## SST and NBP in CMIP models

# %%
# Load CMIP NBP
ts_nbp_cmip = pd.read_csv(proj_path+'/'+'01-data/nbp_CMIP_processed.csv', index_col=0) 

# =======================================
# IMPORTANT: Sign Convention
# =======================================
# NBP (Net Biome Production): NBP > 0 means land absorbs carbon (sink)
# CGR (Carbon Growth Rate): CGR > 0 means atmospheric CO2 increases (source)
# Relationship: CGR ≈ -NBP (ignoring ocean and fossil fuel)
#
# To compare with CGR observations, we convert NBP → CGR equivalent:
ts_nbp_cmip = -ts_nbp_cmip  # NBP → CGR equivalent
#
# After this conversion:
# - All correlations, sensitivities automatically in CGR convention
# - Labels use: -γ_NBP^SSTA (equivalent to γ_CGR^SSTA)
# =======================================

ts_nbp_cmip.index = pd.to_datetime(ts_nbp_cmip.index)
# Select the time period
ts_nbp_cmip = ts_nbp_cmip[(ts_nbp_cmip.index >= start_date) & (ts_nbp_cmip.index <= end_date)]

# %% [markdown]
# ### 📌 Sign Convention: NBP vs CGR
#
# #### Definitions
#
# **NBP (Net Biome Production)**:
# - NBP > 0: Land **absorbs** carbon (carbon sink)
# - NBP < 0: Land **releases** carbon (carbon source)
#
# **CGR (Carbon Growth Rate)**:
# - CGR > 0: Atmospheric CO₂ **increases** (net source to atmosphere)
# - CGR < 0: Atmospheric CO₂ **decreases** (net sink from atmosphere)
#
# #### Relationship
#
# ```
# CGR ≈ -NBP
# ```
#
# (Ignoring ocean uptake and fossil fuel emissions variability)
#
# #### Implementation in This Notebook
#
# **At data loading** (Cell above):
# ```python
# ts_nbp_cmip = -ts_nbp_cmip  # Convert NBP → CGR equivalent
# ```
#
# **After conversion**:
# - ✅ All CMIP model data in **CGR convention**
# - ✅ Direct comparison with **observational CGR** data
# - ✅ No need for sign flips in analysis
# - ✅ Labels use: $-\gamma_{SSTA}^{NBP}$ (equivalent to $\gamma_{SSTA}^{CGR}$)
#
# #### Example
#
# | Scenario | Original NBP | After Conversion | Interpretation |
# |----------|--------------|------------------|----------------|
# | La Niña recovery | NBP increases (+2 Gt C) | -NBP decreases (-2 Gt C) | Atmosphere loses carbon (recovery) |
# | El Niño stress | NBP decreases (-2 Gt C) | -NBP increases (+2 Gt C) | Atmosphere gains carbon (stress) |
#
# ---
#

# %%
# Load CMIP SST
ts_sst_cmip = pd.read_csv(proj_path+'/'+'01-data/nina34_CMIP_processed.csv', index_col=0) 
ts_sst_cmip.index = pd.to_datetime(ts_sst_cmip.index)
# Select the time period
ts_sst_cmip = ts_sst_cmip[(ts_sst_cmip.index >= start_date) & (ts_sst_cmip.index <= end_date)]

# %%
# Load CMIP MRSO
ts_mrso_cmip = pd.read_csv(proj_path+'/'+'01-data/mrso_CMIP_processed.csv', index_col=0) 
ts_mrso_cmip.index = pd.to_datetime(ts_mrso_cmip.index)
# Select the time period
ts_mrso_cmip = ts_mrso_cmip[(ts_mrso_cmip.index >= start_date) & (ts_mrso_cmip.index <= end_date)]

# %%

# %% [markdown]
# ## Correlation between SST and NBP

# %%
# =============================================
# Calculate | NBP vs SST Correlation Analysis
# =============================================

# Set the store list 
correlation_results = []

# calculate correlation for each CMIP model
for model in ts_sst_cmip.columns:
    
    # calculate correlation (SST | NBP)
    correlation_result = calculate_correlation(ts_sst_cmip[model], ts_nbp_cmip[model], min_samples=10)

    if correlation_result is not None:
        correlation_results.append({
            'model': model,
            'correlation': correlation_result['r'],
            'p_value': correlation_result['p-val'],
            'ci_lower': correlation_result['CI95%'][0],
            'ci_upper': correlation_result['CI95%'][1],
            'n_samples': correlation_result['n_samples'],
            'significant': correlation_result['p-val'] < 0.05
        })
        
    else:
        print(f"  Failed to calculate correlation")

# Convert to DataFrame
nbp_sst_correlation_df = pd.DataFrame(correlation_results)

# Print correlation results
nbp_sst_correlation_df

# %%
# =============================================
# Plot | NBP vs SST Correlation Analysis
# =============================================

fig, axes = pplt.subplots(
    ncols=1, nrows=1, 
    refaspect=3, 
    sharey=False, sharex=True, 
    journal='nat2'
)

BAR_COLOR = "#B22222"  # Brick red
EDGE_COLOR = "#000000"  # Black
OBS_ALPHA = 0.9
MODEL_ALPHA = 0.5

ax_corr = axes[0]

# ============== Prepare Model Data ==============
models = nbp_sst_correlation_df['model'].values
# Note: NBP already converted to CGR equivalent at data loading
correlations = nbp_sst_correlation_df['correlation'].values
ci_lower = nbp_sst_correlation_df['ci_lower'].values
ci_upper = nbp_sst_correlation_df['ci_upper'].values
significant = nbp_sst_correlation_df['significant'].values

# ============== Observational Data ==============
obs_r = corr_obs['r'].values[0] 
obs_ci_lower = corr_obs['CI95%'][0][0]
obs_ci_upper = corr_obs['CI95%'][0][1]

# ============== Combine: Observation First ==============
models_all = np.concatenate([['Observation'], models])
correlations_all = np.concatenate([[obs_r], correlations])
ci_lower_all = np.concatenate([[obs_ci_lower], ci_lower])
ci_upper_all = np.concatenate([[obs_ci_upper], ci_upper])

# ============== Check Sign Consistency ==============
# Models with opposite sign to observation are marked as non-significant
obs_sign = np.sign(obs_r)
model_signs = np.sign(correlations)
sign_consistent = (model_signs == obs_sign)

# Combine significance: must be both statistically significant AND sign-consistent
significant_all = np.concatenate([[True], significant & sign_consistent])

# ============== Calculate Error Bars ==============
corr_yerr_lower = correlations_all - ci_lower_all
corr_yerr_upper = ci_upper_all - correlations_all

# ============== X-axis Positions ==============
n_total = len(models_all)
model_positions = np.arange(1, n_total + 1)

# ============== Plot Bars ==============
for i, (model, corr, sig) in enumerate(zip(models_all, correlations_all, significant_all)):
    if np.isnan(corr):
        continue
    
    # Observation (position 1, leftmost)
    if model == 'Observation':
        ax_corr.bar(
            model_positions[i], corr, 
            color=BAR_COLOR, 
            alpha=OBS_ALPHA,
            edgecolor=EDGE_COLOR,
            linewidth=1, 
            width=1,
            zorder=5
        )
    else:
        # Models: significant (dark) vs non-significant (white)
        bar_color = BAR_COLOR if sig else "#FFFFFF"
        ax_corr.bar(
            model_positions[i], corr, 
            color=bar_color, 
            alpha=MODEL_ALPHA,
            edgecolor=EDGE_COLOR, 
            linewidth=1, 
            width=1,
            zorder=5
        )

# ============== Plot Error Bars ==============
for i, (model, corr) in enumerate(zip(models_all, correlations_all)):
    if np.isnan(corr):
        continue
    
    # Observation error bar (red, thick)
    if model == 'Observation':
        ax_corr.errorbar(
            model_positions[i], 
            corr, 
            yerr=[[corr_yerr_lower[i]], [corr_yerr_upper[i]]],
            fmt='o', 
            color=BAR_COLOR, 
            capsize=0, 
            capthick=0,
            markersize=5, 
            markerfacecolor='white',
            linestyle='none', 
            linewidth=1, 
            zorder=10
        )
    else:
        # Model error bars (black, thin)
        ax_corr.errorbar(
            model_positions[i], 
            corr, 
            yerr=[[corr_yerr_lower[i]], [corr_yerr_upper[i]]],
            fmt='o', 
            color=BAR_COLOR, 
            capsize=5, 
            capthick=0,
            markersize=5, 
            markerfacecolor='white',
            linestyle='none', 
            linewidth=1, 
            zorder=10
        )

# ============== Reference Lines ==============
# Horizontal line at zero
ax_corr.axhline(0, color='k', linestyle='-', linewidth=1, zorder=1)

# Vertical separator after observation
ax_corr.axvline(
    model_positions[0] + 0.5,  # After observation (position 1)
    color='gray', 
    linestyle=':', 
    linewidth=1.5, 
    alpha=0.7,
    zorder=0
)

# ============== Format Axes ==============
ax_corr.format(
    xlim=(model_positions.min() - 1, model_positions.max() + 1), 
    ylim=(-1, 1), 
    ylocator=0.3, 
    xlocator=model_positions,
    xlabel='', 
    ylabel=r'Correlation between carbon and ENSO',
    ygrid=False,
    xgrid=True,
    gridcolor='gray',
    gridlinestyle=':',
    xtickminor=False, 
    ytickminor=True, 
    xticklabels=models_all.tolist(), 
    xrotation=90
)

# ============== Print Summary ==============
n_models = len(models)
n_sig_models = np.sum(significant_all[1:])  # Exclude observation
n_consistent = np.sum(sign_consistent)

print(f"\n=== Correlation Analysis Summary ===")
print(f"Observation: r = {obs_r:.3f}, 95% CI = [{obs_ci_lower:.3f}, {obs_ci_upper:.3f}]")
print(f"Total models: {n_models}")
print(f"Models with consistent sign: {n_consistent}/{n_models} ({100*n_consistent/n_models:.1f}%)")
print(f"Significant & consistent models: {n_sig_models}/{n_models} ({100*n_sig_models/n_models:.1f}%)")

# %%
# Select the significant correlation models
corr_models_mask = nbp_sst_correlation_df['correlation'] > 0.3
corr_models = nbp_sst_correlation_df['model'][corr_models_mask]

corr_models


# %% [markdown]
# # Analysis on carbon recovery periods

# %% [markdown]
# ## Identify carbon recoevry periods

# %%
# =======================================
# IMPORTANT: Recovery Period Detection for CGR-equivalent Data
# =======================================
# Since we converted NBP → CGR equivalent at data loading (ts_nbp_cmip = -ts_nbp_cmip),
# the recovery pattern has REVERSED:
#
# Original NBP recovery: negative (source) → positive (sink)
# CGR-equivalent recovery: POSITIVE (source) → NEGATIVE (sink)
#
# Therefore, we need to:
# 1. Find POSITIVE peaks (max_indices) as recovery START
# 2. Find NEGATIVE troughs (min_indices) as recovery END
# 3. Require NEGATIVE slope (decreasing trend)
# 4. Check: start > 0, end < 0
# =======================================

def find_cmip_recovery_periods(model_nbp, model_name, min_duration=10, max_duration=36, min_gap_months=3):
   
    from scipy.stats import linregress
    from scipy.signal import find_peaks
    
    # Define volcanic eruption periods
    volcanic_periods = [
        ('1963-03-01', '1964-12-31'),  # Mount Agung
        ('1982-03-01', '1983-12-31'),  # El Chichón  
        ('1991-06-01', '1993-12-31')   # Pinatubo
    ]
    volcanic_dates = []
    for start_str, end_str in volcanic_periods:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        volcanic_dates.extend(pd.date_range(start, end, freq='M'))
    
    # Step 1: Identify peaks and troughs
    # For CGR-equivalent data, recovery goes from POSITIVE peak → NEGATIVE trough
    max_indices, _ = find_peaks(model_nbp.values, prominence=0.5, distance=6)   # Positive peaks (START)
    min_indices, _ = find_peaks(-model_nbp.values, prominence=0.5, distance=6)  # Negative troughs (END)
    
    recovery_periods = []
    last_end_idx = None
    
    for i, max_idx in enumerate(max_indices):
        next_mins = min_indices[min_indices > max_idx]
        if len(next_mins) == 0:
            continue
        
        min_idx = next_mins[0]
        duration = min_idx - max_idx + 1
        
        # === Key modification: split if too long ===
        if duration > max_duration:
            # Re-identify peaks and troughs within this long interval
            sub_data = model_nbp.iloc[max_idx:min_idx+1]
            sub_max_idx, _ = find_peaks(sub_data.values, prominence=0.25, distance=3)    # Positive peaks
            sub_min_idx, _ = find_peaks(-sub_data.values, prominence=0.25, distance=3)   # Negative troughs
            
            # Iterate over recovery events within the sub-interval
            for sub_i, sub_max in enumerate(sub_max_idx):
                next_sub_mins = sub_min_idx[sub_min_idx > sub_max]
                if len(next_sub_mins) == 0:
                    continue
                
                sub_min = next_sub_mins[0]
                sub_duration = sub_min - sub_max + 1
                
                if sub_duration < min_duration or sub_duration > max_duration:
                    continue
                
                # Convert back to global indices
                global_start_idx = max_idx + sub_max
                global_end_idx = max_idx + sub_min
                
                # === Volcanic impact check and truncation ===
                start_date = model_nbp.index[global_start_idx]
                end_date = model_nbp.index[global_end_idx]
                
                # Check if volcanic eruptions occurred during the recovery period
                period_dates = pd.date_range(start_date, end_date, freq='M')
                volcanic_in_period = [d for d in period_dates if d in volcanic_dates]
                
                if volcanic_in_period:
                    # Truncate before the first volcanic date
                    first_volcanic_date = min(volcanic_in_period)
                    # Find the index of the month before the volcanic date
                    truncate_date = first_volcanic_date - pd.DateOffset(months=1)
                    
                    if truncate_date <= start_date:
                        continue  # Volcanic eruption too early, skip this recovery period
                    
                    # Update end date and index
                    try:
                        global_end_idx = model_nbp.index.get_loc(truncate_date)
                        end_date = truncate_date
                        sub_duration = global_end_idx - global_start_idx + 1
                        
                        if sub_duration < min_duration:
                            continue  # Too short after truncation, skip
                    except KeyError:
                        continue  # Corresponding date not found, skip
                
                # Check gap
                if last_end_idx is not None and (global_start_idx - last_end_idx) < min_gap_months:
                    continue
                
                # Extract data and validate
                period_data = model_nbp.iloc[global_start_idx:global_end_idx+1]
                
                # === Monotonicity test (CORRECTED for CGR-equivalent) ===
                x = np.arange(len(period_data))
                slope, _, r_value, _, _ = linregress(x, period_data.values)
                if slope >= 0 or r_value**2 < 0.1:  # Require NEGATIVE slope (decreasing)
                    continue
                
                # Amplitude test (CORRECTED for CGR-equivalent)
                start_nbp = period_data.iloc[:5].max()  # Should be POSITIVE (source)
                end_nbp = period_data.iloc[-5:].min()   # Should be NEGATIVE (sink)
                if start_nbp < 0 or end_nbp > 0:        # Skip if pattern is wrong
                    continue
                
                # Passed all tests, add to results
                recovery_periods.append({
                    'model': model_name,
                    'group_id': len(recovery_periods) + 1,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_months': len(period_data),
                    'start_nbp': start_nbp,
                    'end_nbp': end_nbp,
                    'nbp_change': end_nbp - start_nbp,  # Should be NEGATIVE (recovery = decrease)
                    'data': period_data
                })
                
                last_end_idx = global_end_idx
        
        else:
            # Short interval, process directly
            if duration < min_duration:
                continue
            
            # Check gap
            if last_end_idx is not None and (max_idx - last_end_idx) < min_gap_months:
                continue
            
            # === Volcanic impact check and truncation ===
            start_date = model_nbp.index[max_idx]
            end_date = model_nbp.index[min_idx]
            
            # Check if there are volcanic eruptions during the recovery period
            period_dates = pd.date_range(start_date, end_date, freq='M')
            volcanic_in_period = [d for d in period_dates if d in volcanic_dates]
            
            if volcanic_in_period:
                # Truncate before the first volcanic date
                first_volcanic_date = min(volcanic_in_period)
                truncate_date = first_volcanic_date - pd.DateOffset(months=1)
                
                if truncate_date <= start_date:
                    continue  # Volcanic eruption too early, skip this recovery period
                
                # Update end date and index
                try:
                    min_idx = model_nbp.index.get_loc(truncate_date)
                    end_date = truncate_date
                    duration = min_idx - max_idx + 1
                    
                    if duration < min_duration:
                        continue  # Too short after truncation, skip
                except KeyError:
                    continue  # Corresponding date not found, skip
            
            period_data = model_nbp.iloc[max_idx:min_idx+1]
            
            # Monotonicity test (CORRECTED for CGR-equivalent)
            x = np.arange(len(period_data))
            slope, _, r_value, _, _ = linregress(x, period_data.values)
            if slope >= 0 or r_value**2 < 0.1:  # Require NEGATIVE slope
                continue
            
            # Amplitude test (CORRECTED for CGR-equivalent)
            start_nbp = period_data.iloc[:5].max()  # Should be POSITIVE
            end_nbp = period_data.iloc[-5:].min()   # Should be NEGATIVE
            if start_nbp < 0 or end_nbp > 0:        # Skip if wrong pattern
                continue
            
            recovery_periods.append({
                'model': model_name,
                'group_id': len(recovery_periods) + 1,
                'start_date': start_date,
                'end_date': end_date,
                'duration_months': duration,
                'start_nbp': start_nbp,
                'end_nbp': end_nbp,
                'nbp_change': end_nbp - start_nbp,  # Should be NEGATIVE
                'data': period_data
            })
            
            last_end_idx = min_idx
    
    return recovery_periods

#-----------------------------------
#----- Main -------# 
#-----------------------------------
print("=== Processing CMIP models (CGR-equivalent data) ===")

cmip_recovery_data_unified = {}

for model in corr_models:
    model_nbp = ts_nbp_cmip[model].dropna()
    # Find recovery periods for each model
    recovery_periods = find_cmip_recovery_periods(model_nbp, model)

    if recovery_periods:
        cmip_recovery_data_unified[model] = recovery_periods  
    else:
        print(f"  No valid recovery periods found")

# Generate summary table
summary_data_unified = []
for model, periods in cmip_recovery_data_unified.items():
    for period in periods:
        summary_data_unified.append({
            'model': period['model'],
            'group_id': period['group_id'],
            'start_date': period['start_date'],
            'end_date': period['end_date'],
            'duration_months': period['duration_months'],
            'start_nbp': period['start_nbp'],
            'end_nbp': period['end_nbp'],
            'nbp_change': period['nbp_change']
        })

cmip_recovery_summary_unified = pd.DataFrame(summary_data_unified)
print(f"\n=== CGR-equivalent recovery results ===")
print(f"Total number of recovery periods: {len(cmip_recovery_summary_unified)}")
print(f"Number of models with recovery periods: {cmip_recovery_summary_unified['model'].nunique()}")

cmip_recovery_summary_unified

# %%

# %% [markdown]
# ## Find best lags between SST and NBP

# %%
# =======================================
# CMIP Recovery Periods - Optimal Lag Analysis
# =======================================

# Configuration
LAG_RANGE = range(0, -24, -1)  # [0, -1, -2, ..., -24] months lag

# Storage for results
cmip_recovery_lag_results = []

# Process each model that has recovery periods
for model in cmip_recovery_data_unified.keys():
    nbp_data = ts_nbp_cmip[model]
    sst_data = ts_sst_cmip[model]
    sst_data.index = pd.to_datetime(sst_data.index)
    nbp_data.index = pd.to_datetime(nbp_data.index)

    # Process each recovery period for this model
    for period in cmip_recovery_data_unified[model]:
        group_id = period['group_id']
        event_start_date = pd.to_datetime(period['start_date'])
        event_end_date = pd.to_datetime(period['end_date'])

        Y_data = nbp_data.loc[event_start_date:event_end_date]

        best_lag, best_corr, best_result = None, None, None

        for lag in LAG_RANGE:
            sst_start = event_start_date - pd.DateOffset(months=-lag)
            sst_end = event_end_date - pd.DateOffset(months=-lag)

            if sst_start < sst_data.index[0] or sst_end > sst_data.index[-1]:
                continue

            X_data = sst_data.loc[sst_start:sst_end]

            if len(X_data) != len(Y_data):
                continue

            result = calculate_correlation(X_data, Y_data, min_samples=5)

            if result is not None:
                if best_result is None or result['r'] > best_corr:
                    best_corr = result['r']
                    best_lag = lag
                    best_result = result

        if best_result is not None:
            cmip_recovery_lag_results.append({
                'model': model,
                'group_id': group_id,
                'start_date': event_start_date,
                'end_date': event_end_date,
                'duration': len(Y_data),
                'optimal_lag': best_lag,
                'correlation': best_corr,
                'p_value': best_result['p-val'],
                'ci_lower': best_result['CI95%'][0],
                'ci_upper': best_result['CI95%'][1],
                'n_samples': best_result['n_samples'],
                'significant': best_result['p-val'] < 0.05,
            })
        else:
            print(f"  No valid correlation found for {model} group {group_id}")

# Convert to DataFrame
cmip_recovery_lag_df = pd.DataFrame(cmip_recovery_lag_results)

print(f"\n=== CMIP Recovery Lag Analysis Summary ===")
print(f"Total recovery periods analyzed: {len(cmip_recovery_lag_df)}")
print(f"Models with valid correlations: {cmip_recovery_lag_df['model'].nunique()}")
print(f"Significant correlations (p<0.05): {cmip_recovery_lag_df['significant'].sum()}")

# Display results
cmip_recovery_lag_df

# %%

# %%
# Filter for significant positive correlations (p < 0.001)
cmip_recovery_lag_df_filtered = cmip_recovery_lag_df[
    (cmip_recovery_lag_df['correlation'] > 0) &
    (cmip_recovery_lag_df['p_value'] < 0.05)
].copy()

# ============== Renumber group_id ==============
# Renumber group_id for each model starting from 1
cmip_recovery_lag_df_filtered['group_id'] = (
    cmip_recovery_lag_df_filtered
    .groupby('model')
    .cumcount() + 1  # cumcount() starts from 0, so add 1
)

cmip_recovery_lag_df_filtered

# %%
# =============================================
# Plot All CMIP Models Recovery Periods (Two Columns)
# =============================================

# Time periods and eruption events
ERUPTION_PERIODS = {
    'Mount Agung': ('1962-01-01', '1963-12-31'),
    'El Chichón': ('1982-01-01', '1982-12-31'),
    'Pinatubo': ('1992-01-01', '1993-12-31')
}

# Get all available models
all_models = list(corr_models)
n_models = len(all_models)

# Calculate grid dimensions for 2 columns
n_cols = 2
n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

# Create figure with 2-column layout
fig, axes = pplt.subplots(
    nrows=n_rows, 
    ncols=n_cols, 
    refaspect=3,
    journal='nat2', 
    share=0
)

# Time series data
time_cmip = pd.to_datetime(ts_nbp_cmip.index)

# Track which labels have been added for legend
legend_added = {'nbp': False, 'ssta': False, 'recovery': False, 'volcano': False}

# Define middle row for adding labels
middle_row = n_rows // 2

# Plot each model
for i, model in enumerate(all_models):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    # ============== NBP Time Series (Black Line) ==============
    nbp_label = 'NBP' if (row == middle_row and col == 0 and not legend_added['nbp']) else ''
    if nbp_label:
        legend_added['nbp'] = True
    
    ax.plot(
        time_cmip, 
        -ts_nbp_cmip[model].values, 
        '-', 
        color='black', 
        alpha=0.8, 
        lw=1.5, 
        zorder=5,
        label=nbp_label
    )
    
    # ============== SSTA Time Series (Red Line, Right Y-axis) ==============
    ax_sst = ax.twinx()
    
    ssta_label = 'SSTA (Niño 3.4)' if (row == middle_row and col == 1 and not legend_added['ssta']) else ''
    if ssta_label:
        legend_added['ssta'] = True
    
    ax_sst.plot(
        time_cmip,
        ts_sst_cmip[model].values,
        "-",
        color="red",
        alpha=0.7,
        lw=1.5,
        zorder=4,
        label=ssta_label
    )
    
    # ============== Calculate Balanced Y-axis Limits ==============
    # NBP: Calculate symmetric ylim
    nbp_data = ts_nbp_cmip[model].values # here we convert back to NBP for limit calculation
    nbp_lim = np.ceil(np.nanmax(np.abs(nbp_data)))
    
    # SSTA: Calculate symmetric ylim
    sst_data = ts_sst_cmip[model].values
    sst_lim = np.ceil(np.nanmax(np.abs(sst_data)))
    
    # ============== Format SSTA Axis (Right) ==============
    ax_sst.format(
        ylabel="SSTA [°C]",
        ylim=(-sst_lim, sst_lim), 
        ylocator=sst_lim / 2,
        grid=False,
        ycolor='red',
        yminorticks=False   
    )
    
    # ============== Add Volcanic Eruption Periods (Gray Shading) ==============
    for j, (eruption, (start, end)) in enumerate(ERUPTION_PERIODS.items()):
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        volcano_label = 'Volcanic eruptions' if (row == middle_row and col == 0 and j == 0 and not legend_added['volcano']) else ''
        if volcano_label:
            legend_added['volcano'] = True
        
        ax.axvspan(
            start_date, end_date, 
            color='gray', 
            alpha=0.25, 
            zorder=1, 
            edgecolor=None, 
            linewidth=0,
            label=volcano_label
        )
    
    # ============== Add Carbon Recovery Periods (Green Shading) ==============
    # Use FILTERED recovery events (after correlation/sensitivity screening)
    model_filtered_periods = cmip_recovery_lag_df_filtered[cmip_recovery_lag_df_filtered['model'] == model]
    
    if not model_filtered_periods.empty:
        for j, (idx, period) in enumerate(model_filtered_periods.iterrows()):
            event_start_date = pd.to_datetime(period["start_date"])
            event_end_date = pd.to_datetime(period["end_date"])
            
            recovery_label = 'Carbon recovery periods' if (row == middle_row and col == 0 and j == 0 and not legend_added['recovery']) else ''
            if recovery_label:
                legend_added['recovery'] = True
            
            ax.axvspan(
                event_start_date,
                event_end_date,
                color="green",
                alpha=0.15,
                edgecolor=None,
                linewidth=0,
                zorder=2,
                label=recovery_label
            )
    
    # ============== Format NBP Axis (Left) ==============
    ax.format(
        ylabel='NBP [Gt C]',
        xlabel='Year' if row == n_rows - 1 else '',
        ylim=(nbp_lim, -nbp_lim),
        xlim=(time_cmip[0], time_cmip[-1]),
        xlocator=('year', 10), 
        xformatter='%Y', 
        xminorlocator=('year', 5), 
        xticklabels=[] if row < n_rows - 1 else None,
        xrotation=90,
        grid=False,
        ylocator=nbp_lim / 2,
        title=model,
        titleweight='bold',
        titlesize=9,
        yminorticks=False   
    )

# ============== Hide Unused Subplots ==============
if n_models % n_cols != 0:
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

# ============== Add Figure Legend ==============
fig.legend(
    loc='b', 
    ncols=4, 
    frameon=False, 
    fontsize=8
)

# ============== Save Figure ==============
fig.savefig("03-res/figs/SP_CMIP_carbon_recovery_episodes.png", dpi=600)

# %%
# =======================================
# CMIP Recovery Periods - Sensitivity Analysis
# =======================================

cmip_recovery_sensitivity_results = []

for _, row in cmip_recovery_lag_df_filtered.iterrows():
    model = row["model"]
    group_id = row["group_id"]
    start_date = row["start_date"]
    end_date = row["end_date"]
    optimal_lag = row["optimal_lag"]  # Best lag from previous analysis
    
    # ============== Get Data ==============
    nbp_data = ts_nbp_cmip[model]
    ssta_data = ts_sst_cmip[model]  # SSTA
    
    # ============== Apply Optimal Lag ==============
    # Shift SSTA time window by optimal lag
    # Example: if optimal_lag = -3, SSTA leads NBP by 3 months
    ssta_start = start_date - pd.DateOffset(months=-optimal_lag)
    sst_end = end_date - pd.DateOffset(months=-optimal_lag)
    
    try:
        # ============== Extract Data ==============
        X_data = ssta_data.loc[ssta_start:sst_end].values  # SSTA (predictor)
        Y_data = nbp_data.loc[start_date:end_date].values   # NBP (response)
        
        # Check data length match
        if len(X_data) != len(Y_data):
            print(f"  Data length mismatch for {model} group {group_id}: SSTA={len(X_data)}, NBP={len(Y_data)}")
            continue
        
        # ============== Calculate Regression ==============
        # Linear regression: NBP = β₀ + β₁ × SSTA + ε
        # Sensitivity = β₁ (slope)
        min_samples = max(3, len(Y_data) // 3)
        reg_result = calculate_regression(
            X_data, Y_data, 
            conf_level=0.95, 
            min_samples=min_samples
        )
        
        # ============== Store Results ==============
        if not np.isnan(reg_result["coef"]):
            cmip_recovery_sensitivity_results.append({
                "model": model,
                "group_id": group_id,
                "start_date": start_date,
                "end_date": end_date,
                "duration": len(Y_data),
                "optimal_lag": optimal_lag,
                "sensitivity": reg_result["coef"],  # β₁: NBP sensitivity to SSTA
                "sensitivity_p": reg_result["pvalue"],
                "sensitivity_ci_lower": reg_result["ci_lower"],
                "sensitivity_ci_upper": reg_result["ci_upper"],
                "r_squared": reg_result["r2"],
                "n_samples": reg_result["n_samples"],
                "significant": reg_result["pvalue"] < 0.05
            })
        else:
            print(f"  Failed to calculate sensitivity for {model} group {group_id}")
            
    except Exception as e:
        print(f"  Error processing {model} group {group_id}: {e}")
        continue

# ============== Convert to DataFrame ==============
cmip_recovery_sensitivity_df = pd.DataFrame(cmip_recovery_sensitivity_results)

# ============== Print Summary ==============
cmip_recovery_sensitivity_df

# %%

# %%
# =======================================
# CMIP Recovery Sensitivity Line Plots for Each Model
# =======================================

# ============== Prepare Data ==============
# Get all models with sensitivity data
models_with_data = cmip_recovery_sensitivity_df["model"].unique()
n_models = len(models_with_data)

# ============== Grid Layout ==============
# Calculate grid dimensions (2 columns)
n_cols = 2
n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

# ============== Create Figure ==============
fig, axes = pplt.subplots(
    nrows=n_rows, 
    ncols=n_cols, 
    refaspect=2, 
    share=0,  # Independent axes for each subplot
    journal="nat2"
)

# ============== Color Scheme ==============
POINT_COLOR = "#B22222"      # Brick red for points
EDGE_COLOR = "#000000"       # Black edges
ERROR_COLOR = "#B22222"      # Brick red for error bars
POINT_ALPHA = 0.8
ERROR_ALPHA = 0.6

# ============== Plot Each Model ==============
for i, model in enumerate(models_with_data):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    # Get sensitivity data for this model
    model_data = cmip_recovery_sensitivity_df[
        cmip_recovery_sensitivity_df["model"] == model
    ].copy()
    model_data = model_data.sort_values("group_id")

    # Extract data
    group_ids = model_data["group_id"].values
    sensitivities = model_data["sensitivity"].values
    ci_lower = model_data["sensitivity_ci_lower"].values
    ci_upper = model_data["sensitivity_ci_upper"].values
    
    # Calculate error bar sizes
    yerr_lower = sensitivities - ci_lower
    yerr_upper = ci_upper - sensitivities

    # ========== Plot Error Bars ==========
    ax.errorbar(
        group_ids,
        sensitivities,
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        color=ERROR_COLOR,
        alpha=ERROR_ALPHA,
        capsize=0,
        capthick=1,
        linewidth=1,
        zorder=3
    )
    
    # ========== Plot Points ==========
    ax.scatter(
        group_ids,
        sensitivities,
        s=30,
        color=POINT_COLOR,
        edgecolor=EDGE_COLOR,
        alpha=POINT_ALPHA,
        marker="o",
        linewidth=1,
        zorder=10
    )

    # ========== Add Zero Reference Line ==========
    ax.axhline(
        y=0, 
        color="gray", 
        linestyle="--", 
        linewidth=0.8, 
        alpha=0.5,
        zorder=1
    )

    # ========== Format Subplot ==========
    ax.format(
        title=model,
        titlesize=8,
        titleweight="bold",
        grid=False,
        xlim=(group_ids.min() - 0.5, group_ids.max() + 0.5),
        ylim=(0, None),  # 0 min, auto max
        xlocator=1,
        xtickminor=False,
        ytickminor=False
    )

# ============== Hide Unused Subplots ==============
if n_models % n_cols != 0:
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")

# ============== Add Common Axis Labels ==============
# Add centered X and Y labels for the entire figure
fig.format(
    xlabel="Recovery Episode",
    ylabel=r"Recovery Sensitivity" + "\n" + r"$-\gamma_{SSTA}^{NBP}$ [Gt C per °C]",
    xlabelloc="bottom",  # X label at bottom center
    ylabelloc="left"     # Y label at left center
)

# ============== Print Summary ==============

total_episodes = len(cmip_recovery_sensitivity_df)
print(f"Total recovery episodes: {total_episodes}")
print(f"Average episodes per model: {total_episodes/n_models:.1f}")

# %%

# %% [markdown]
# # Sensitivity analysis among carbon recovery 

# %% [markdown]
# ### Trend of sensitivity

# %%
# =======================================
# CMIP vs Observations | Trend of Sensitivity
# =======================================

print("=== CMIP Recovery Sensitivity Trend Analysis ===")

cmip_recovery_sensitivity_trend_results = []

for model in cmip_recovery_sensitivity_df["model"].unique():

    # Get sensitivity data for this model
    model_data = cmip_recovery_sensitivity_df[
        cmip_recovery_sensitivity_df["model"] == model
    ].copy()
    model_data = model_data.sort_values("group_id")

    # Extract sensitivity values
    sensitivities = model_data["sensitivity"].values
    group_ids = model_data["group_id"].values

    if len(sensitivities) > 5:  # Need at least 3 points for trend analysis

        # Calculate linear regression trend
        X_data = np.arange(1, len(sensitivities) + 1)  # Sequential event order
        Y_data = sensitivities

        # Use our existing regression function
        trend_result = calculate_regression(
            X_data, Y_data, conf_level=0.95, min_samples=5
        )

        if not np.isnan(trend_result["coef"]):
            # Store trend results
            cmip_recovery_sensitivity_trend_results.append(
                {
                    "model": model,
                    "n_events": len(sensitivities),
                    "trend_slope": trend_result[
                        "coef"
                    ],  # Change in sensitivity per event
                    "trend_p_value": trend_result["pvalue"],
                    "trend_r_squared": trend_result["r2"],
                    "trend_ci_lower": trend_result["ci_lower"],
                    "trend_ci_upper": trend_result["ci_upper"],
                    "trend_significant": trend_result["pvalue"] < 0.10,
                    "mean_sensitivity": np.mean(sensitivities),
                    "std_sensitivity": np.std(sensitivities, ddof=1),
                    "min_sensitivity": np.min(sensitivities),
                    "max_sensitivity": np.max(sensitivities),
                    "sensitivity_range": np.max(sensitivities) - np.min(sensitivities),
                }
            )

    else:
        print(f"  Insufficient data for trend analysis (need ≥3 events)")

        # Still store basic statistics
        cmip_recovery_sensitivity_trend_results.append(
            {
                "model": model,
                "n_events": len(sensitivities),
                "trend_slope": np.nan,
                "trend_p_value": np.nan,
                "trend_r_squared": np.nan,
                "trend_ci_lower": np.nan,
                "trend_ci_upper": np.nan,
                "trend_significant": False,
                "mean_sensitivity": (
                    np.mean(sensitivities) if len(sensitivities) > 0 else np.nan
                ),
                "std_sensitivity": (
                    np.std(sensitivities, ddof=1) if len(sensitivities) > 1 else np.nan
                ),
                "min_sensitivity": (
                    np.min(sensitivities) if len(sensitivities) > 0 else np.nan
                ),
                "max_sensitivity": (
                    np.max(sensitivities) if len(sensitivities) > 0 else np.nan
                ),
                "sensitivity_range": (
                    np.max(sensitivities) - np.min(sensitivities)
                    if len(sensitivities) > 0
                    else np.nan
                ),
            }
        )

# Convert to DataFrame
cmip_sensitivity_trend_df = pd.DataFrame(cmip_recovery_sensitivity_trend_results)

# Display results
cmip_sensitivity_trend_df

# %%

# %% [markdown]
# ### Avergae of sensitivity

# %%
# =======================================
# CMIP vs Observations | Mean Sensitivity
# =======================================

# ============== Create Figure ==============
from matplotlib.pyplot import ylabel


fig, ax = pplt.subplots(
    ncols=1, nrows=1, 
    refaspect=3, 
    sharey=4, sharex=1, 
    journal='nat2'
)

# ============== Prepare Model Data ==============
models = cmip_sensitivity_trend_df['model'].values
means = cmip_sensitivity_trend_df['mean_sensitivity'].values
stds = cmip_sensitivity_trend_df['std_sensitivity'].values

# ============== Observational Data ==============
# Individual recovery period sensitivities
obs_sensitivities = np.array([
    1.52088972, 2.54632606, 1.85509519, 2.62254683, 
    4.35694912, 1.28471424, 1.15832473, 1.99992818, 
    1.85927278, 2.18403324, 1.68014934, 1.60739798
])

# Calculate mean and std (negated for CGR sign convention)
obs_mean = np.mean(obs_sensitivities)
obs_std = np.std(obs_sensitivities, ddof=1)

# ============== X-axis Positions ==============
# Position 0: Observation, Positions 1+: Models
model_positions = np.arange(0, len(models) + 1)

# ============== Color Scheme ==============
BAR_COLOR = "#B22222"  # Brick red
EDGE_COLOR = "#000000"  # Black
OBS_ALPHA = 0.9
MODEL_ALPHA = 0.5

# ============== Plot Observation ==============
ax.bar(
    model_positions[0], 
    obs_mean, 
    color=BAR_COLOR, 
    alpha=OBS_ALPHA, 
    edgecolor=EDGE_COLOR,  # 添加黑色边框
    linewidth=1.5, 
    width=1,
    zorder=5,
    label=r"$\gamma_{SSTA}^{CGR}$"
)

# Observation error bar
ax.errorbar(
    model_positions[0], 
    obs_mean, 
    yerr=obs_std, 
    fmt='o', 
    color=BAR_COLOR, 
    capsize=5, 
    capthick=0, 
    markersize=5, 
    markerfacecolor='white', 
    linestyle='none', 
    linewidth=2, 
    zorder=10
)

# ============== Plot Models ==============
for i, (model, mean, std) in enumerate(zip(models, means, stds)):
    if np.isnan(mean):
        continue  # Skip invalid data
    
    pos = model_positions[i + 1]
    
    ax.bar(
        pos, 
        mean, 
        color=BAR_COLOR, 
        alpha=MODEL_ALPHA, 
        edgecolor=EDGE_COLOR,  
        linewidth=1, 
        width=1,
        zorder=5,
        label=r"$-\gamma_{SSTA}^{NBP}$" if i == 0 else None  # Label only once
    )

# ============== Plot Model Error Bars ==============
valid_mask = ~np.isnan(means)

ax.errorbar(
    model_positions[1:][valid_mask], 
    means[valid_mask], 
    yerr=stds[valid_mask],
    fmt='o', 
    color=BAR_COLOR, 
    capsize=5, 
    capthick=0,
    markersize=5, 
    markerfacecolor='white',
    linestyle='none', 
    linewidth=2, 
    zorder=10
)

# ============== Reference Lines ==============
# Horizontal line at zero
ax.axhline(
    y=0, 
    color='black', 
    linestyle='-', 
    linewidth=0.8, 
    alpha=0.6,
    zorder=1
)

# Vertical separator between observation and models
ax.axvline(
    x=model_positions[0] + 0.5,  # Between position 0 and 1
    color='gray', 
    linestyle=':', 
    linewidth=1.5, 
    alpha=0.7,
    zorder=0
)

# ============== Format Axes ==============
ax.legend(loc='ur', ncols=1, frameon=False)
ax.format(
    xlim=(model_positions.min() - 1, model_positions.max() + 1), 
    ylim=(0, 13), 
    ylocator=2, 
    xlocator=model_positions,
    xlabel='', 
    ylabel="Recovery sensitivity [Gt C per °C]",
    grid=False, 
    xgrid=True,
    gridcolor='gray',
    gridlinestyle=':',
    xtickminor=False, 
    ytickminor=False, 
    xticklabels=['Observation'] + models.tolist(), 
    xrotation=90
)

# ============== Print Summary ==============
n_models = len(models)
model_mean_avg = np.nanmean(means)
model_mean_std = np.nanstd(means)

print(f"\n=== Mean Sensitivity Summary ===")
print(f"Observation: {obs_mean:.2f} ± {obs_std:.2f} Gt C per °C")
print(f"Models ({n_models} total):")
print(f"  Average: {model_mean_avg:.2f} ± {model_mean_std:.2f} Gt C per °C")
print(f"  Range: [{np.nanmin(means):.2f}, {np.nanmax(means):.2f}] Gt C per °C")


# %%

# %% [markdown]
# # ENSO legacy impacts on sensitivity

# %% [markdown]
# ## Preprocess

# %%
def find_enso_boundary_model(sst_series, recovery_start, max_lookback_months=24):
    """
    Find ENSO boundary by excluding La Niña months.
    Returns the month after 3 consecutive negative SSTA months.
    """
    current_date = recovery_start
    consecutive_months = 0
    required_consecutive = 3

    for i in range(max_lookback_months):
        current_date = current_date - pd.DateOffset(months=1)
        
        if current_date < sst_series.index[0]:
            break
        
        try:
            sst_value = sst_series.loc[current_date]
            
            if sst_value < 0:
                consecutive_months += 1
                if consecutive_months >= required_consecutive:
                    # Current_date is the 2nd consecutive negative month
                    # Return the next month (forward in time) to exclude these 2
                    return current_date + pd.DateOffset(months=required_consecutive)
            else:
                consecutive_months = 0
                
        except (KeyError, IndexError):
            consecutive_months = 0
            continue
    
    return recovery_start - pd.DateOffset(months=12)


# %%
# =======================================
# Antecedent Period Analysis: ENSO + Water Storage
# =======================================
# 
# Purpose: For each filtered recovery event, analyze the antecedent period
#          (from ENSO boundary to recovery start) to calculate:
#          1. El Niño strength (warm phase ENSO)
#          2. Water storage anomalies (TWS)
#          3. Recovery period NBP changes
#
# Scientific Logic:
#   Antecedent El Niño → Antecedent TWS changes → Recovery sensitivity
#
# Note: The find_enso_boundary_model function stops at La Niña onset,
#       so the antecedent period is dominated by warm phase ENSO (SSTA > 0)
# =======================================

print("\n" + "="*70)
print("ANTECEDENT PERIOD ANALYSIS")
print("="*70)

antecedent_results = []

# Iterate over filtered recovery events
for idx, (_, row) in enumerate(cmip_recovery_lag_df_filtered.iterrows(), 1):
    model = row['model']
    group_id = row['group_id']
    recovery_start = pd.to_datetime(row["start_date"])
    recovery_end = pd.to_datetime(row["end_date"])
    
    # Progress indicator
    if idx % 50 == 0:
        print(f"Processing event {idx}/{len(cmip_recovery_lag_df_filtered)}...")
    
    try:
        # ============================================================
        # Step 1: Get Time Series Data
        # ============================================================
        sst_data = ts_sst_cmip[model]
        mrso_data = ts_mrso_cmip[model]
        nbp_data = ts_nbp_cmip[model]
        
        # ============================================================
        # Step 2: Find ENSO Boundary (stops at La Niña onset)
        # ============================================================
        extended_start = find_enso_boundary_model(
            sst_data, recovery_start, max_lookback_months=24
        )
        
        # ============================================================
        # Step 3: Extract Time Windows
        # ============================================================
        # Antecedent period: from ENSO boundary to recovery start
        antecedent_enso = sst_data.loc[extended_start:recovery_start]
        antecedent_mrso = mrso_data.loc[extended_start:recovery_start]
        
        # Recovery period: from recovery start to end
        recovery_nbp = nbp_data.loc[recovery_start:recovery_end]
        
        # Skip if data is insufficient
        if len(antecedent_enso) < 5:
            print(f"  Insufficient data for {model} group {group_id}")
            continue
        
        # ============================================================
        # Step 4: Calculate ENSO Metrics (El Niño strength)
        # ============================================================
        
        # ENSO statistics
        enso_avg = antecedent_enso[antecedent_enso>0].mean()  # antecedent_enso.mean()
        enso_count = len(antecedent_enso[antecedent_enso>0])
        enso_total_months = len(antecedent_enso)
        enso_fraction = enso_count / enso_total_months if enso_total_months > 0 else 0
        enso_sum = enso_avg * enso_count if not np.isnan(enso_avg) else np.nan
        
        # ============================================================
        # Step 5: Calculate Water Storage Metrics (TWS)
        # ============================================================
        # TWS statistics during antecedent period
        mrso_sum_total = antecedent_mrso.sum()
        mrso_mean = antecedent_mrso.mean()
        mrso_std = antecedent_mrso.std()
        mrso_min = antecedent_mrso.min()
        mrso_max = antecedent_mrso.max()
        
        # TWS change: end - start
        mrso_start_val = antecedent_mrso.iloc[0]
        mrso_end_val = antecedent_mrso.iloc[-1]
        mrso_change = mrso_end_val - mrso_start_val
        
        # ============================================================
        # Step 6: Calculate Recovery NBP Metrics
        # ============================================================
        # NBP change during recovery
        nbp_start_val = recovery_nbp.iloc[0]
        nbp_end_val = recovery_nbp.iloc[-1]
        nbp_change = nbp_end_val - nbp_start_val  # Should be negative (recovery)
        
        # Recovery duration
        recovery_duration = len(recovery_nbp)
        
        # ============================================================
        # Step 7: Calculate Lookback Duration
        # ============================================================
        actual_lookback_months = (
            (recovery_start.year - extended_start.year) * 12 + 
            (recovery_start.month - extended_start.month)
        )
        
        # ============================================================
        # Step 8: Store All Results
        # ============================================================
        antecedent_results.append({
            # -------------------- Identifiers --------------------
            'model': model,
            'group_id': group_id,
            
            # -------------------- Time Periods --------------------
            'antecedent_start': extended_start,
            'recovery_start': recovery_start,
            'recovery_end': recovery_end,
            'antecedent_months': enso_total_months,
            'recovery_months': recovery_duration,
            'actual_lookback_months': actual_lookback_months,
            
            # -------------------- ENSO Metrics --------------------
            'enso_avg': enso_avg,                    # Mean El Niño intensity (°C)
            'enso_count': enso_count,                # Number of warm months
            'enso_sum': enso_sum,                    # Cumulative warm anomaly
            'enso_fraction': enso_fraction,          # Fraction of warm months
            
            # -------------------- Water Storage Metrics --------------------
            'mrso_sum_total': mrso_sum_total,        # Total antecedent TWS
            'mrso_mean': mrso_mean,                  # Mean antecedent TWS
            'mrso_std': mrso_std,                    # Std of antecedent TWS
            'mrso_min': mrso_min,                    # Min antecedent TWS
            'mrso_max': mrso_max,                    # Max antecedent TWS
            'mrso_change': mrso_change,              # TWS change (end - start)
            'mrso_start': mrso_start_val,            # TWS at antecedent start
            'mrso_end': mrso_end_val,                # TWS at recovery start
            
            # -------------------- Recovery NBP Metrics --------------------
            'nbp_change': nbp_change,                # NBP change during recovery
            'nbp_start': nbp_start_val,              # NBP at recovery start
            'nbp_end': nbp_end_val,                  # NBP at recovery end
        })
        
    except Exception as e:
        print(f"  ERROR processing {model} group {group_id}: {e}")
        continue

# ============================================================
# Convert to DataFrame
# ============================================================
cmip_antecedent_df = pd.DataFrame(antecedent_results)

# Display DataFrame
cmip_antecedent_df

# %%
# =======================================
# Comprehensive Analysis - Merge All Variables
# =======================================
#
# Purpose: Integrate all analyzed variables into one DataFrame:
#          1. Recovery characteristics (from filtered events)
#          2. CGR-ENSO sensitivity (from regression analysis)
#          3. Antecedent ENSO + Water metrics (from antecedent analysis)
#
# Final dataset for pathway analysis:
#   Antecedent El Niño → Antecedent TWS → Recovery Sensitivity
# =======================================

print("\n" + "="*70)
print("COMPREHENSIVE DATA INTEGRATION")
print("="*70)

# ============================================================
# Step 1: Start with Filtered Recovery Events
# ============================================================
cmip_comprehensive_df = cmip_recovery_lag_df_filtered.copy()
print(f"\nStarting with {len(cmip_comprehensive_df)} filtered recovery events")

# ============================================================
# Step 2: Merge Sensitivity Data (CGR-ENSO regression)
# ============================================================
cmip_comprehensive_df = cmip_comprehensive_df.merge(
    cmip_recovery_sensitivity_df[[
        'model', 'group_id', 
        'sensitivity',                # β₁: CGR sensitivity to SSTA
        'sensitivity_p',              # p-value
        'sensitivity_ci_lower',       # 95% CI lower
        'sensitivity_ci_upper',       # 95% CI upper
        'r_squared'                   # R²
    ]],
    on=['model', 'group_id'],
    how='left',
    suffixes=('', '_sens')
)

sensitivity_matched = cmip_comprehensive_df['sensitivity'].notna().sum()
print(f"After merging sensitivity:       {sensitivity_matched} events matched")

# ============================================================
# Step 3: Merge Antecedent Data (ENSO + Water in one!)
# ============================================================
cmip_comprehensive_df = cmip_comprehensive_df.merge(
    cmip_antecedent_df[[
        'model', 'group_id',
        # Time periods
        'antecedent_start', 'antecedent_months', 'recovery_months',
        # ENSO metrics
        'enso_avg', 'enso_count', 'enso_sum', 'enso_fraction',
        # Water metrics
        'mrso_sum_total', 'mrso_mean', 'mrso_std', 'mrso_change',
        'mrso_start', 'mrso_end',
        # Recovery NBP
        'nbp_change', 'nbp_start', 'nbp_end'
    ]],
    on=['model', 'group_id'],
    how='left',
    suffixes=('', '_ant')
)

antecedent_matched = cmip_comprehensive_df['enso_avg'].notna().sum()
print(f"After merging antecedent data:   {antecedent_matched} events matched")

# ============================================================
# Step 4: Clean Up
# ============================================================
# Remove any duplicate rows
cmip_comprehensive_df_complete = cmip_comprehensive_df.drop_duplicates(
    subset=['model', 'group_id']
)

# %%
cmip_comprehensive_df_complete

# %%

# %% [markdown]
# ## ENSO Intensity vs recovery sensitivity

# %%
# =======================================
# Analysis 1: ENSO Strength → CGR-ENSO relationship
# =======================================

# Calculate correlation between ENSO strength and water storage cumulative sum
cmip_enso_cgr_correlation_results = []

for model in cmip_comprehensive_df_complete['model'].unique():
    model_data = cmip_comprehensive_df_complete[cmip_comprehensive_df_complete['model'] == model]
    
    if len(model_data) >= 5:  # Need at least 5 points for correlation
        # Extract variables
        enso_avg = model_data['enso_avg'].values
        cgr_enso = model_data['sensitivity'].values
        
        # Calculate correlation: ENSO → Water Storage
        corr_result = calculate_correlation(
            pd.Series(enso_avg), 
            pd.Series(cgr_enso), 
            min_samples=5,
        )
        
        if corr_result is not None:
            cmip_enso_cgr_correlation_results.append({
                'model': model,
                'n_events': len(model_data),
                'correlation': corr_result['r'],
                'p_value': corr_result['p-val'],
                'ci_lower': corr_result['CI95%'][0],
                'ci_upper': corr_result['CI95%'][1],
                'significant': corr_result['p-val'] < 0.05,
                'mean_enso_avg': np.mean(enso_avg),
                'mean_cgr_enso': np.mean(cgr_enso)
            })

# Convert to DataFrame
cmip_enso_cgr_corr_df = pd.DataFrame(cmip_enso_cgr_correlation_results)

print(f"\n=== ENSO Strength vs CGR-ENSO Correlation Results ===")
print(f"Models with correlation analysis: {len(cmip_enso_cgr_corr_df)}")

cmip_enso_cgr_corr_df

# %%
# =======================================
# Visualization: ENSO Strength vs NBP-ENSO Sensitivity (Bar Chart)
# =======================================

# === Configure ===
# Observation configuration
OBSERVATION_DATA = {
    'correlation': -0.79,
    'p_value': 0.001,
    'ci_lower': -0.73,
    'ci_upper': -0.85
}

# Plotting configuration
PLOT_CONFIG = {
    'colors': {
        'observation': '#B22222',
        'model_significant': '#B22222',
        'model_non_significant': 'white'
    },
    'significance_threshold': 0.05,
    'alpha_values': {
        'observation': 0.9,
        'model': 0.5
    }
}

# === Create the figure ===
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=3, journal='nat2')

# === Data extraction ===
models = cmip_enso_cgr_corr_df['model'].values
correlations = cmip_enso_cgr_corr_df['correlation'].values # NBP already converted to CGR at loading
p_values = cmip_enso_cgr_corr_df['p_value'].values
ci_lower = cmip_enso_cgr_corr_df['ci_lower'].values
ci_upper = cmip_enso_cgr_corr_df['ci_upper'].values

# X-axis positions (0 for observation, 1~N for models)
model_positions = np.arange(0, len(models) + 1)

# === Plot observation ===
obs_corr = OBSERVATION_DATA['correlation']
obs_p = OBSERVATION_DATA['p_value']
obs_ci_low = OBSERVATION_DATA['ci_lower']
obs_ci_high = OBSERVATION_DATA['ci_upper']

# Observation bar (significant, solid red with higher alpha)
ax.bar(
    model_positions[0], obs_corr, 
    color=PLOT_CONFIG['colors']['observation'], 
    edgecolor=PLOT_CONFIG['colors']['observation'], ew=1, 
    alpha=PLOT_CONFIG['alpha_values']['observation'], 
    width=1, zorder=10,
    label='Observation'
)

# Observation error bar
obs_yerr_lower = obs_corr - obs_ci_low
obs_yerr_upper = obs_ci_high - obs_corr
ax.errorbar(
    model_positions[0], obs_corr, 
    yerr=[[obs_yerr_lower], [obs_yerr_upper]],
    fmt='o', color=PLOT_CONFIG['colors']['observation'], 
    capsize=0, markersize=5, markerfacecolor='white', 
    linestyle='none', linewidth=1, zorder=10
)

# === Plot model data ===
for i, (model_name, correlation, p_val, ci_low, ci_high) in enumerate(
    zip(models, correlations, p_values, ci_lower, ci_upper)
):
    x_position = model_positions[i+1]  # Skip position 0 (observation)
    
    # Check data validity
    if np.isnan(correlation):
        continue

    # Calculate error bar components
    yerr_lower = correlation - ci_low if not np.isnan(ci_low) else 0
    yerr_upper = ci_high - correlation if not np.isnan(ci_high) else 0
    
    # Determine significance
    is_significant = not np.isnan(p_val) and p_val < PLOT_CONFIG['significance_threshold']

    # Choose bar style based on significance
    if is_significant:
        # Significant: solid bar
        ax.bar(
            x_position, correlation, 
            color=PLOT_CONFIG['colors']['model_significant'], 
            alpha=PLOT_CONFIG['alpha_values']['model'], 
            width=1, zorder=10
        )
    else:
        # Not significant: hollow bar
        ax.bar(
            x_position, correlation, 
            color=PLOT_CONFIG['colors']['model_non_significant'], 
            alpha=PLOT_CONFIG['alpha_values']['model'], 
            edgecolor=PLOT_CONFIG['colors']['model_significant'], 
            linewidth=1, width=1, zorder=10
        )
    
    # Add error bar
    ax.errorbar(
        x_position, correlation, 
        yerr=[[yerr_lower], [yerr_upper]], 
        fmt='o', color=PLOT_CONFIG['colors']['model_significant'], 
        capsize=0, markersize=5, markerfacecolor='white', 
        linestyle='none', linewidth=1, zorder=10
    )

# === Figure formatting ===
ax.format(
    xlim=(model_positions.min()-1, model_positions.max()+1), 
    ylim=(-1.1, 1.1), 
    ylocator=0.3, 
    xlocator=model_positions,
    xlabel='', 
    ylabel='Correlation between El Niño intensity \n and recovery sensitivity',
    grid=False, 
    xgrid=True,
    ygrid=False,
    gridcolor='gray',
    xtickminor=False, 
    ytickminor=False, 
    xticklabels=['Observation'] + models.tolist(), 
    xrotation=90
)

# Add zero baseline
ax.axhline(
    y=0, color='black', linestyle='-', 
    linewidth=0.8, alpha=0.6, zorder=1
)

# Vertical separator between observation and models
ax.axvline(
    x=model_positions[0] + 0.5,  # Between position 0 and 1
    color='gray', 
    linestyle=':', 
    linewidth=1.5, 
    alpha=0.7,
    zorder=0
)

fig.savefig('03-res/figs/constraint_diagnostic_in_models.png', dpi=600)

# %%

# %%
# =======================================
# Visualization: ENSO Strength vs Water Storage Cumulative (Scatter Plot Grid)
# =======================================

# === Configuration parameters ===
# Figure layout configuration
GRID_CONFIG = {
    'n_cols': 3,
    'refaspect': 2,
    'share': 0,
    'journal': "nat2"
}

# Scatter plot style configuration
SCATTER_CONFIG = {
    'marker_size': 50,
    'marker_color': '#B22222',
    'marker_alpha': 0.5,
    'marker_style': 'o',
    'zorder': 10
}

# Text annotation configuration
TEXT_CONFIG = {
    'position': (0.3, 0.85),  # (x, y) in axes coordinates
    'ha': "center",
    'va': "bottom",
    'fontsize': 8,
    'significant_color': '#B22222',
    'non_significant_color': 'gray'
}

# Axis label configuration
AXIS_CONFIG = {
    'title_size': 8,
    'title_weight': 'bold',
    'xlabel': 'Antecedent El Niño intensity [°C]',
    'ylabel': 'Recovery sensitivity \n [Gt C per °C]'
}

# === Data preparation ===
# Get models with correlation data
available_models = cmip_enso_cgr_corr_df['model'].unique()
n_models = len(available_models)

# Calculate grid layout
n_cols = GRID_CONFIG['n_cols']
n_rows = (n_models + n_cols - 1) // n_cols

# === Create figure ===
fig, axes = pplt.subplots(
    nrows=n_rows, 
    ncols=n_cols, 
    refaspect=GRID_CONFIG['refaspect'], 
    share=GRID_CONFIG['share'], 
    journal=GRID_CONFIG['journal']
)

# === Draw scatter plot for each model ===
for i, model_name in enumerate(available_models):
    # Calculate subplot position
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    # Extract data for this model
    model_data = cmip_comprehensive_df_complete[
        cmip_comprehensive_df_complete['model'] == model_name
    ]
    correlation_data = cmip_enso_cgr_corr_df[
        cmip_enso_cgr_corr_df['model'] == model_name
    ]
    
    # Check data availability
    if len(model_data) == 0:
        print(f"Warning: No data available for model {model_name}")
        continue
    
    # Draw scatter plot
    ax.scatter(
        model_data['enso_avg'], 
        model_data['sensitivity'], 
        s=SCATTER_CONFIG['marker_size'],
        color=SCATTER_CONFIG['marker_color'],
        alpha=SCATTER_CONFIG['marker_alpha'],
        marker=SCATTER_CONFIG['marker_style'],
        zorder=SCATTER_CONFIG['zorder']
    )
    
    # === Add correlation annotation ===
    if len(correlation_data) > 0:
        # Extract correlation statistics
        correlation_coeff = correlation_data['correlation'].iloc[0]
        p_value = correlation_data['p_value'].iloc[0]
        is_significant = correlation_data['significant'].iloc[0]
        
        # Choose text color based on significance
        text_color = (TEXT_CONFIG['significant_color'] if is_significant 
                     else TEXT_CONFIG['non_significant_color'])
        
        # Add correlation info text
        ax.text(
            TEXT_CONFIG['position'][0], 
            TEXT_CONFIG['position'][1],
            f"R = {correlation_coeff:.3f}, P = {p_value:.3f}",
            ha=TEXT_CONFIG['ha'],
            va=TEXT_CONFIG['va'],
            transform=ax.transAxes,
            fontsize=TEXT_CONFIG['fontsize'],
            color=text_color
        )
    
    # === Subplot formatting ===
    # Show axis labels only on the edge subplots
    show_xlabel = (row == n_rows - 1)
    show_ylabel = (col == 0)
    
    ax.format(
        title=model_name,
        titlesize=AXIS_CONFIG['title_size'],
        titleweight=AXIS_CONFIG['title_weight'],
        grid=False,
        xlabel=AXIS_CONFIG['xlabel'] if show_xlabel else '',
        ylabel=AXIS_CONFIG['ylabel'] if show_ylabel else ''
    )

# === Hide unused subplots ===
# If the number of models does not fill the grid, hide extra subplots
for i in range(n_models, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].axis('off')


# %%

# %% [markdown]
# ## ENSO Intensity vs water Anomalies

# %%
# =======================================
# Analysis 1: ENSO Strength → Water Storage Cumulative
# =======================================

# Calculate correlation between ENSO strength and water storage cumulative sum
cmip_enso_water_correlation_results = []

for model in cmip_comprehensive_df_complete['model'].unique():
    model_data = cmip_comprehensive_df_complete[cmip_comprehensive_df_complete['model'] == model]
    
    if len(model_data) >= 3:  # Need at least 3 points for correlation
        # Extract variables
        enso_avg = model_data['enso_avg'].values
        water_avg = model_data['mrso_mean'].values
        
        # Calculate correlation: ENSO → Water Storage
        corr_result = calculate_correlation(
            pd.Series(enso_avg), 
            pd.Series(water_avg), 
            min_samples=3
        )
        
        if corr_result is not None:
            cmip_enso_water_correlation_results.append({
                'model': model,
                'n_events': len(model_data),
                'correlation': corr_result['r'],
                'p_value': corr_result['p-val'],
                'ci_lower': corr_result['CI95%'][0],
                'ci_upper': corr_result['CI95%'][1],
                'significant': corr_result['p-val'] < 0.10,
                'mean_enso_avg': np.mean(enso_avg),
                'mean_water_avg': np.mean(water_avg)
            })

# Convert to DataFrame
cmip_enso_water_corr_df = pd.DataFrame(cmip_enso_water_correlation_results)

print(f"\n=== ENSO Strength vs Water Storage Correlation Results ===")
print(f"Models with correlation analysis: {len(cmip_enso_water_corr_df)}")
print(f"Significant correlations: {cmip_enso_water_corr_df['significant'].sum()}")

cmip_enso_water_corr_df

# %%
# =======================================
# Visualization: ENSO Strength vs Water Storage
# =======================================

# === Configure ===
# Observation configuration
OBSERVATION_DATA = {
    'correlation': -0.701,
    'p_value': 0.011,
    'ci_lower': -0.65,
    'ci_upper': -0.75
}

# Plotting configuration
PLOT_CONFIG = {
    'colors': {
        'observation': '#FF6B35',
        'model_significant': '#FF6B35',
        'model_non_significant': 'white'
    },
    'significance_threshold': 0.10,
    'alpha_values': {
        'observation': 0.75,
        'model': 0.5
    }
}

# === Create the figure ===
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=3, journal='nat2')

# === Data extraction ===
models = cmip_enso_water_corr_df['model'].values
correlations = cmip_enso_water_corr_df['correlation'].values
p_values = cmip_enso_water_corr_df['p_value'].values
ci_lower = cmip_enso_water_corr_df['ci_lower'].values
ci_upper = cmip_enso_water_corr_df['ci_upper'].values

# X-axis positions (0 for observation, 1~N for models)
model_positions = np.arange(0, len(models) + 1)

# === Plot observation ===
obs_corr = OBSERVATION_DATA['correlation']
obs_p = OBSERVATION_DATA['p_value']
obs_ci_low = OBSERVATION_DATA['ci_lower']
obs_ci_high = OBSERVATION_DATA['ci_upper']

# Observation bar (significant, solid orange)
ax.bar(
    model_positions[0], obs_corr, 
    color=PLOT_CONFIG['colors']['observation'], 
    edgecolor='k', ew=1.25, 
    alpha=PLOT_CONFIG['alpha_values']['observation'], 
    width=1, label='Observation', zorder=10
)

# Observation error bar
obs_yerr_lower = obs_corr - obs_ci_low
obs_yerr_upper = obs_ci_high - obs_corr
ax.errorbar(
    model_positions[0], obs_corr, 
    yerr=[[obs_yerr_lower], [obs_yerr_upper]],
    fmt='o', color=PLOT_CONFIG['colors']['observation'], 
    capsize=0, markersize=5, markerfacecolor='white', 
    alpha=PLOT_CONFIG['alpha_values']['observation'],
    linestyle='none', linewidth=1, zorder=10
)

# === Plot model data ===
for i, (model_name, correlation, p_val, ci_low, ci_high) in enumerate(
    zip(models, correlations, p_values, ci_lower, ci_upper)
):
    x_position = model_positions[i+1]  # Skip position 0 (observation)
    
    # Check data validity
    if np.isnan(correlation):
        continue
    
    # Determine significance
    is_significant = not np.isnan(p_val) and p_val < PLOT_CONFIG['significance_threshold']
    
    # Choose bar style based on significance
    if is_significant:
        # Significant: solid bar
        bar_color = PLOT_CONFIG['colors']['model_significant']
        edge_color = PLOT_CONFIG['colors']['model_significant']
        ax.bar(
            x_position, correlation, 
            color=bar_color, alpha=PLOT_CONFIG['alpha_values']['model'], 
            edgecolor=edge_color, linewidth=1, width=1, zorder=10
        )
    else:
        # Not significant: hollow bar
        ax.bar(
            x_position, correlation, 
            color=PLOT_CONFIG['colors']['model_non_significant'], 
            alpha=PLOT_CONFIG['alpha_values']['model'], 
            edgecolor=PLOT_CONFIG['colors']['model_significant'], 
            linewidth=1, width=1, zorder=10
        )
    
    # Add error bar
    yerr_lower = correlation - ci_low
    yerr_upper = ci_high - correlation
    ax.errorbar(
        x_position, correlation, 
        yerr=[[yerr_lower], [yerr_upper]], 
        fmt='o', color=PLOT_CONFIG['colors']['model_significant'], 
        capsize=0, alpha=PLOT_CONFIG['alpha_values']['model'],
        markersize=5, markerfacecolor='white', 
        linestyle='none', linewidth=1, zorder=10
    )

# === Figure formatting ===
ax.format(
    xlim=(model_positions.min()-1, model_positions.max()+1), 
    ylim=(-1.1, 1.1), 
    ylocator=0.3, 
    xlocator=model_positions,
    xlabel='', 
    ylabel=r'$R^{ENSO}_{MRSO}$ or $R^{ENSO}_{TWS}$',
    grid=False, 
    xgrid=True,
    ygrid=False,
    gridcolor='gray',
    xtickminor=False, 
    ytickminor=False, 
    xticklabels=['Observation'] + models.tolist(), 
    xrotation=90
)

# Add zero baseline
ax.axhline(
    y=0, color='black', linestyle='--', 
    linewidth=0.8, alpha=0.6, zorder=1
)

# %%
# =======================================
# Visualization: ENSO Strength vs Water Storage (Scatter Plot Grid)
# =======================================

# === Configuration parameters ===
# Figure layout configuration
GRID_CONFIG = {
    'n_cols': 3,
    'refaspect': 2,
    'share': 0,
    'journal': "nat2"
}

# Scatter plot style configuration
SCATTER_CONFIG = {
    'marker_size': 50,
    'marker_color': '#FF6B35',
    'marker_alpha': 0.5,
    'marker_style': 'o',
    'zorder': 10
}

# Text annotation configuration
TEXT_CONFIG = {
    'position': (0.3, 0.05),  # (x, y) in axes coordinates
    'ha': "center",
    'va': "bottom",
    'fontsize': 8,
    'significant_color': '#FF6B35',
    'non_significant_color': 'gray'
}

# Axis label configuration
AXIS_CONFIG = {
    'title_size': 8,
    'title_weight': 'bold',
    'xlabel': 'El Niño intensity [°C]',
    'ylabel': 'Water anomalies [mm]'
}

# === Data preparation ===
# Get models with correlation data
available_models = cmip_enso_water_corr_df['model'].unique()
n_models = len(available_models)

# Calculate grid layout
n_cols = GRID_CONFIG['n_cols']
n_rows = (n_models + n_cols - 1) // n_cols

# === Create figure ===
fig, axes = pplt.subplots(
    nrows=n_rows, 
    ncols=n_cols, 
    refaspect=GRID_CONFIG['refaspect'], 
    share=GRID_CONFIG['share'], 
    journal=GRID_CONFIG['journal']
)

# === Draw scatter plot for each model ===
for i, model_name in enumerate(available_models):
    # Calculate subplot position
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    # Extract data for this model
    model_data = cmip_comprehensive_df_complete[
        cmip_comprehensive_df_complete['model'] == model_name
    ]
    correlation_data = cmip_enso_water_corr_df[
        cmip_enso_water_corr_df['model'] == model_name
    ]
    
    # Check data availability
    if len(model_data) == 0:
        print(f"Warning: No data available for model {model_name}")
        continue
    
    # Draw scatter plot
    ax.scatter(
        model_data['enso_avg'], 
        model_data['mrso_mean'], 
        s=SCATTER_CONFIG['marker_size'],
        color=SCATTER_CONFIG['marker_color'],
        alpha=SCATTER_CONFIG['marker_alpha'],
        marker=SCATTER_CONFIG['marker_style'],
        zorder=SCATTER_CONFIG['zorder']
    )
    
    # === Add correlation annotation ===
    if len(correlation_data) > 0:
        # Extract correlation statistics
        correlation_coeff = correlation_data['correlation'].iloc[0]
        p_value = correlation_data['p_value'].iloc[0]
        is_significant = correlation_data['significant'].iloc[0]
        
        # Choose text color based on significance
        text_color = (TEXT_CONFIG['significant_color'] if is_significant 
                     else TEXT_CONFIG['non_significant_color'])
        
        # Add correlation info text
        ax.text(
            TEXT_CONFIG['position'][0], 
            TEXT_CONFIG['position'][1],
            f"R = {correlation_coeff:.3f}, P = {p_value:.3f}",
            ha=TEXT_CONFIG['ha'],
            va=TEXT_CONFIG['va'],
            transform=ax.transAxes,
            fontsize=TEXT_CONFIG['fontsize'],
            color=text_color
        )
    
    # === Subplot formatting ===
    # Show axis labels only on the edge subplots
    show_xlabel = (row == n_rows - 1)
    show_ylabel = (col == 0)
    
    ax.format(
        title=model_name,
        titlesize=AXIS_CONFIG['title_size'],
        titleweight=AXIS_CONFIG['title_weight'],
        grid=False,
        xlabel=AXIS_CONFIG['xlabel'] if show_xlabel else '',
        ylabel=AXIS_CONFIG['ylabel'] if show_ylabel else ''
    )

# === Hide unused subplots ===
# If the number of models does not fill the grid, hide extra subplots
for i in range(n_models, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].axis('off')

# %%

# %% [markdown]
# ## Water Anomalies vs recovery sensitiivty

# %%
# =======================================
# Analysis 2: Water Storage Cumulative → NBP-ENSO Sensitivity
# =======================================

# Calculate correlation between water storage cumulative sum and NBP-ENSO sensitivity
cmip_water_sensitivity_correlation_results = []

for model in cmip_comprehensive_df_complete['model'].unique():
    model_data = cmip_comprehensive_df_complete[cmip_comprehensive_df_complete['model'] == model]
    
    if len(model_data) >= 3:  # Need at least 3 points for correlation
        # Extract variables
        water_sum = model_data['mrso_mean'].values
        nbp_sensitivity = model_data['sensitivity'].values
        
        # Calculate correlation: Water Storage → NBP Sensitivity
        corr_result = calculate_correlation(
            pd.Series(water_sum), 
            pd.Series(nbp_sensitivity), 
            min_samples=3
        )
        
        if corr_result is not None:
            cmip_water_sensitivity_correlation_results.append({
                'model': model,
                'n_events': len(model_data),
                'correlation': corr_result['r'],
                'p_value': corr_result['p-val'],
                'ci_lower': corr_result['CI95%'][0],
                'ci_upper': corr_result['CI95%'][1],
                'significant': corr_result['p-val'] < 0.05,
                'mean_water_sum': np.mean(water_sum),
                'mean_nbp_sensitivity': np.mean(nbp_sensitivity)
            })

# Convert to DataFrame
cmip_water_sensitivity_corr_df = pd.DataFrame(cmip_water_sensitivity_correlation_results)

print(f"\n=== Water Storage vs NBP-ENSO Sensitivity Correlation Results ===")
print(f"Models with correlation analysis: {len(cmip_water_sensitivity_corr_df)}")
print(f"Significant correlations: {cmip_water_sensitivity_corr_df['significant'].sum()}")

cmip_water_sensitivity_corr_df

# %%
# =======================================
# Visualization: Water Storage vs NBP-ENSO Sensitivity (Bar Chart)
# =======================================

# === Configure ===
# Observation configuration
OBSERVATION_DATA = {
    'correlation': 0.608,
    'p_value': 0.036,
    'ci_lower': 0.55,
    'ci_upper': 0.85
}

# Plotting configuration
PLOT_CONFIG = {
    'colors': {
        'observation': '#22A1B2',
        'model_significant': '#22A1B2',
        'model_non_significant': 'white'
    },
    'significance_threshold': 0.05,
    'alpha_values': {
        'observation': 0.75,
        'model': 0.5
    }
}

# === Create the figure ===
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=3, journal='nat2')

# === Data extraction ===
models = cmip_water_sensitivity_corr_df['model'].values
correlations = cmip_water_sensitivity_corr_df['correlation'].values
p_values = cmip_water_sensitivity_corr_df['p_value'].values
ci_lower = cmip_water_sensitivity_corr_df['ci_lower'].values
ci_upper = cmip_water_sensitivity_corr_df['ci_upper'].values

# X-axis positions (0 for observation, 1~N for models)
model_positions = np.arange(0, len(models) + 1)

# === Plot observation ===
obs_corr = OBSERVATION_DATA['correlation']
obs_p = OBSERVATION_DATA['p_value']
obs_ci_low = OBSERVATION_DATA['ci_lower']
obs_ci_high = OBSERVATION_DATA['ci_upper']

# Observation bar (significant, solid blue)
ax.bar(
    model_positions[0], obs_corr, 
    color=PLOT_CONFIG['colors']['observation'], 
    edgecolor='k', ew=1.25, 
    alpha=PLOT_CONFIG['alpha_values']['observation'], 
    width=1, label='Observation', zorder=10
)

# Observation error bar
obs_yerr_lower = obs_corr - obs_ci_low
obs_yerr_upper = obs_ci_high - obs_corr
ax.errorbar(
    model_positions[0], obs_corr, 
    yerr=[[obs_yerr_lower], [obs_yerr_upper]],
    fmt='o', color=PLOT_CONFIG['colors']['observation'], 
    capsize=0, markersize=5, markerfacecolor='white', 
    alpha=PLOT_CONFIG['alpha_values']['observation'],
    linestyle='none', linewidth=1, zorder=10
)

# === Plot model data ===
for i, (model_name, correlation, p_val, ci_low, ci_high) in enumerate(
    zip(models, correlations, p_values, ci_lower, ci_upper)
):
    x_position = model_positions[i+1]  # Skip position 0 (observation)
    
    # Check data validity
    if np.isnan(correlation):
        continue
    
    # Determine significance
    is_significant = not np.isnan(p_val) and p_val < PLOT_CONFIG['significance_threshold']
    
    # Choose bar style based on significance
    if is_significant:
        # Significant: solid bar
        bar_color = PLOT_CONFIG['colors']['model_significant']
        edge_color = PLOT_CONFIG['colors']['model_significant']
        ax.bar(
            x_position, correlation, 
            color=bar_color, alpha=PLOT_CONFIG['alpha_values']['model'], 
            edgecolor=edge_color, linewidth=1, width=1, zorder=10
        )

    else:
        # Not significant: hollow bar
        ax.bar(
            x_position, correlation, 
            color=PLOT_CONFIG['colors']['model_non_significant'], 
            alpha=PLOT_CONFIG['alpha_values']['model'], 
            edgecolor=PLOT_CONFIG['colors']['model_significant'], 
            linewidth=1, width=1, zorder=10
        )
    
    # Add error bar
    yerr_lower = correlation - ci_low
    yerr_upper = ci_high - correlation
    ax.errorbar(
        x_position, correlation, 
        yerr=[[yerr_lower], [yerr_upper]], 
        fmt='o', color=PLOT_CONFIG['colors']['model_significant'], 
        capsize=0, alpha=PLOT_CONFIG['alpha_values']['model'],
        markersize=5, markerfacecolor='white', 
        linestyle='none', linewidth=1, zorder=10
    )

# === Figure formatting ===
ax.format(
    xlim=(model_positions.min()-1, model_positions.max()+1), 
    ylim=(-1.1, 1.1), 
    ylocator=0.3, 
    xlocator=model_positions,
    xlabel='', 
    ylabel=r'$R^{MRSO}_{-\gamma^{NBP}_{ENSO}}$',  # CGR equivalent
    grid=False, 
    xgrid=True,
    ygrid=False,
    gridcolor='gray',
    xtickminor=False, 
    ytickminor=False, 
    xticklabels=['Observation'] + models.tolist(), 
    xrotation=90
)

# Add zero baseline
ax.axhline(
    y=0, color='black', linestyle='-', 
    linewidth=0.8, alpha=0.6, zorder=1
)

# %%
# =======================================
# Visualization: Water Storage Cumulative vs NBP-ENSO Sensitivity (Scatter Plot Grid)
# =======================================

# === 配置参数 ===
# 图形布局配置
GRID_CONFIG = {
    'n_cols': 3,
    'refaspect': 2,
    'share': 0,
    'journal': "nat2"
}

# 绘图样式配置
SCATTER_CONFIG = {
    'marker_size': 50,
    'marker_color': '#22A1B2',
    'marker_alpha': 0.5,
    'marker_style': 'o',
    'zorder': 10
}

# 文本标注配置
TEXT_CONFIG = {
    'position': (0.3, 0.05),  # (x, y) in axes coordinates
    'ha': "center",
    'va': "bottom",
    'fontsize': 8,
    'bbox_style': dict(facecolor="white", alpha=1, boxstyle="round,pad=0.3"),
    'significant_color': '#22A1B2',
    'non_significant_color': 'gray'
}

# 轴标签配置
AXIS_CONFIG = {
    'title_size': 8,
    'title_weight': 'bold',
    'xlabel': 'Antecedent water anomalies [mm]',
    'ylabel': 'Recovery sensitivity \n [Gt C per °C]'
}

# === 数据准备 ===
# 获取有相关性数据的模型
available_models = cmip_water_sensitivity_corr_df['model'].unique()
n_models = len(available_models)

# 计算网格布局
n_cols = GRID_CONFIG['n_cols']
n_rows = (n_models + n_cols - 1) // n_cols

# === 创建图形 ===
fig, axes = pplt.subplots(
    nrows=n_rows, 
    ncols=n_cols, 
    refaspect=GRID_CONFIG['refaspect'], 
    share=GRID_CONFIG['share'], 
    journal=GRID_CONFIG['journal'],
)

# === 绘制各模型的散点图 ===
for i, model_name in enumerate(available_models):
    # 计算子图位置
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    
    # 提取该模型的数据
    model_data = cmip_comprehensive_df_complete[
        cmip_comprehensive_df_complete['model'] == model_name
    ]
    correlation_data = cmip_water_sensitivity_corr_df[
        cmip_water_sensitivity_corr_df['model'] == model_name
    ]
    
    # 检查数据可用性
    if len(model_data) == 0:
        print(f"警告: 模型 {model_name} 没有可用数据")
        continue
    
    # 绘制散点图
    ax.scatter(
        model_data['mrso_mean'], 
        model_data['sensitivity'], 
        s=SCATTER_CONFIG['marker_size'],
        color=SCATTER_CONFIG['marker_color'],
        alpha=SCATTER_CONFIG['marker_alpha'],
        marker=SCATTER_CONFIG['marker_style'],
        zorder=SCATTER_CONFIG['zorder']
    )
    
    # === 添加相关性信息标注 ===
    if len(correlation_data) > 0:
        # 提取相关性统计量
        correlation_coeff = correlation_data['correlation'].iloc[0]
        p_value = correlation_data['p_value'].iloc[0]
        is_significant = correlation_data['significant'].iloc[0]
        
        # 根据显著性选择文本颜色
        text_color = (TEXT_CONFIG['significant_color'] if is_significant 
                     else TEXT_CONFIG['non_significant_color'])
        
        # 添加相关性信息文本（多行显示）
        ax.text(
            TEXT_CONFIG['position'][0], 
            TEXT_CONFIG['position'][1],
            f"R = {correlation_coeff:.3f}, P = {p_value:.3f}",
            ha=TEXT_CONFIG['ha'],
            va=TEXT_CONFIG['va'],
            transform=ax.transAxes,
            fontsize=TEXT_CONFIG['fontsize'],
            color=text_color,
        )
    
    # === 子图格式设置 ===
    # 判断是否显示轴标签（只在边缘显示）
    show_xlabel = (row == n_rows - 1)
    show_ylabel = (col == 0)
    
    ax.format(
        title=model_name,
        titlesize=AXIS_CONFIG['title_size'],
        titleweight=AXIS_CONFIG['title_weight'],
        grid=False,
        xlabel=AXIS_CONFIG['xlabel'] if show_xlabel else '',
        ylabel=AXIS_CONFIG['ylabel'] if show_ylabel else ''
    )

# === 隐藏未使用的子图 ===
# 如果模型数量不能完全填满网格，隐藏多余的子图
for i in range(n_models, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].axis('off')

# %%

# %%

# %% [markdown]
# # TEST

# %%
# 合并两步的数据
# IMPORTANT: 符号转换说明
# - 观测使用 CGR (Carbon Growth Rate): CGR↑ = 大气CO2增长↑ = 陆地吸收↓
# - 模型使用 NBP (Net Biome Production): NBP↑ = 陆地吸收↑
# - 关系: CGR ≈ -NBP
# - 因此: γ_CGR^ENSO ≈ -γ_NBP^ENSO
# - 为了与观测比较，模型的 step2 (TWS → γ_NBP^ENSO) 需要取负号
#   转换为等效的 TWS → γ_CGR^ENSO

pathway_data = []

for model in cmip_enso_water_corr_df['model'].unique():
    # 第一步：ENSO -> TWS
    step1 = cmip_enso_water_corr_df[cmip_enso_water_corr_df['model'] == model]
    
    # 第二步：TWS -> 敏感性
    step2 = cmip_water_sensitivity_corr_df[cmip_water_sensitivity_corr_df['model'] == model]
    
    if len(step1) > 0 and len(step2) > 0:
        # 提取相关系数
        enso_to_tws = step1['correlation'].values[0]
        tws_to_sens = step2['correlation'].values[0]
        
        # 计算标准误（用置信区间估算）
        enso_to_tws_se = (step1['ci_upper'].values[0] - step1['ci_lower'].values[0]) / (2 * 1.96)
        tws_to_sens_se = (step2['ci_upper'].values[0] - step2['ci_lower'].values[0]) / (2 * 1.96)
        
        pathway_data.append({
            'model': model,
            'step1_enso_to_tws': enso_to_tws,
            'step1_se': enso_to_tws_se,
            'step2_tws_to_sens': tws_to_sens, 
            'step2_se': tws_to_sens_se,  # SE保持正值（标准误始终为正）
            'step1_sig': step1['significant'].values[0],
            'step2_sig': step2['significant'].values[0]
        })

pathway_df = pd.DataFrame(pathway_data)

print(f"\n=== Pathway Decomposition Data ===")
print(f"Models with complete pathway data: {len(pathway_df)}")
print(f"Both steps significant: {((pathway_df['step1_sig']) & (pathway_df['step2_sig'])).sum()}")

# =======================================
# 观测数据
# =======================================

# Step 1: ENSO -> TWS
obs_step1_r = -0.719
obs_step1_ci = [-0.92, -0.25]
obs_step1_se = (obs_step1_ci[1] - obs_step1_ci[0]) / (2 * 1.96)

# Step 2: TWS -> Sensitivity
# Step 2: TWS -> γ_CGR^ENSO (观测：CGR对ENSO的敏感性)
obs_step2_r = 0.594  # 观测值（保持原始符号）
obs_step2_ci = [0.03, 0.87]  # 观测的95%置信区间
obs_step2_se = (obs_step2_ci[1] - obs_step2_ci[0]) / (2 * 1.96)

obs_enso_to_tws = obs_step1_r
obs_tws_to_sens = obs_step2_r

# %%
# Create the figure
fig, ax = pplt.subplots(ncols=1, nrows=1, journal="nat2", refaspect=1)

ax.format(
    xlabel='Correlation between antecedent TWS anomalies and $\gamma^{NBP}_{SSTA}$',
    ylabel='Correlation between antecedent ENSO intensity and TWS anomalies',
    xlim=(-1, 1), xlocator=0.3,
    ylim=(-1, 1), ylocator=0.3,
    grid=False
)

# -------------- Plot Observations --------------
if True: 

    # --------- CI95%
    # CI95% Y direction
    ax.axhspan(
        obs_step1_r - obs_step1_se, 
        obs_step1_r + obs_step1_se,
        color='#FF6B35', 
        alpha=0.25,
        zorder=1,
        linewidth=0,
    )

    # CI95% X direction
    ax.axvspan(
        obs_step2_r - obs_step2_se,
        obs_step2_r + obs_step2_se,
        color='#22A1B2', 
        alpha=0.25,
        zorder=1,
        linewidth=0,
    )

    # ------------ Observation point
    ax.plot(
        obs_tws_to_sens,
        obs_enso_to_tws,
        marker='o',
        markersize=10,
        color='black',
        markeredgecolor='black',
        markeredgewidth=1,
        alpha=0.75,
        zorder=12
    )

    # ------------ Observation error bars
    ## Y direction
    ax.errorbar(
        obs_tws_to_sens,
        obs_enso_to_tws,
        xerr=0,
        yerr=obs_step1_se,
        fmt='none',
        color='#FF6B35',
        alpha=1,
        capsize=5,
        capthick=0,
        elinewidth=1.5,
        zorder=10
    )
    ## X direction
    ax.errorbar(
        obs_tws_to_sens,
        obs_enso_to_tws,
        xerr=obs_step2_se,
        yerr=0,
        fmt='none',
        color='#22A1B2',
        alpha=1,
        capsize=5,
        capthick=0,
        elinewidth=1.5,
        label='Observation',
        zorder=10
    )

# -------------- Plot Models --------------
if True:
    legend_added = {'sig': False, 'nonsig': False}
    texts = []
    x_coords = []
    y_coords = []
    
    for i, row in pathway_df.iterrows():
        # Significance (p < 0.05)
        both_sig = row['step1_sig'] and row['step2_sig']
        
        if both_sig:
            color = 'black'
            edgecolor = 'black'
            alpha = 0.75
            marker_size = 10
        else:
            color = 'gray'
            edgecolor = 'black'
            alpha = 0.75
            marker_size = 10
        
        x_val = row['step2_tws_to_sens']
        y_val = row['step1_enso_to_tws']
        
        # Error bar colors
        # Y direction error bar (Step 1: ENSO → TWS)
        if row['step1_sig']:
            yerr_color = '#FF6B35' 
        else:
            yerr_color = 'none' 
        
        # ------------- Error bars -------------
        # X direction error bar (Step 2: TWS → Sensitivity)
        if row['step2_sig']:
            xerr_color = '#22A1B2' 
        else:
            xerr_color = 'none'
        
        # Draw Y direction error bar
        ax.errorbar(
            x_val, y_val,
            yerr=row['step1_se'],
            xerr=0, 
            fmt='none',
            ecolor=yerr_color,
            alpha=1,
            capsize=5,
            capthick=0,
            elinewidth=1.5,
            zorder=4
        )
        
        # Draw X direction error bar
        ax.errorbar(
            x_val, y_val,
            xerr=row['step2_se'],
            yerr=0, 
            fmt='none',
            ecolor=xerr_color,
            alpha=1,
            capsize=5,
            capthick=0,
            elinewidth=1.5,
            zorder=4
        )
        
        # ----------- Plot the model point ------------
        ax.plot(
            x_val, y_val,
            'o',
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            markersize=marker_size,
            zorder=5
        )
        
        # ------------- Add texts for model names -------------
        model_name = row['model'].replace('_', '-')
        # Biased directions for text offsets
        offset_x = 0  # X direction offset
        offset_y = 0  # Y direction offset
        
        txt = ax.text(
            x_val + offset_x, y_val + offset_y,  # Add offset
            model_name,
            fontsize=8,
            color='black',
            alpha=1,
            zorder=100)
        texts.append(txt)
        x_coords.append(x_val-offset_x)
        y_coords.append(y_val-offset_y)


if True:
    # Observation annotation
    ax.text(
        obs_tws_to_sens, obs_enso_to_tws,
        'Observation',
        fontsize=8,
        color='red',
        zorder=15,
    )
    # Adjust text positions to avoid overlap
    adjust_text(
        texts,
        x=x_coords,
        y=y_coords,
        arrowprops=dict(arrowstyle='-', color='k', lw=0.5, alpha=0.5),
        expand_points=(2, 2),
        expand_text=(3, 3),
        force_points=1,
        force_text=1,
        avoid_points=True,
        avoid_self=True,
        lim=300,
        ax=ax,
    )

# # 添加零参考线
# ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=1, zorder=0)
# ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=1, zorder=0)

# 格式化 - 使用不同颜色的轴标签

# 设置X轴和Y轴标签的颜色
ax.xaxis.label.set_color('#1f77b4')  # 蓝色 - 匹配X轴阴影
ax.yaxis.label.set_color('#d62728')  # 红色 - 匹配Y轴阴影

# %%
# 输出统计信息
print(f"\n=== Model Statistics ===")
print(f"Mean Step1 (ENSO→TWS): {pathway_df['step1_enso_to_tws'].mean():.3f} ± {pathway_df['step1_enso_to_tws'].std():.3f}")
print(f"Mean Step2 (TWS→Sens): {pathway_df['step2_tws_to_sens'].mean():.3f} ± {pathway_df['step2_tws_to_sens'].std():.3f}")
print(f"\nObservation:")
print(f"Step1 (ENSO→TWS): {obs_enso_to_tws:.3f} (95% CI: [{obs_step1_ci[0]:.2f}, {obs_step1_ci[1]:.2f}])")
print(f"Step2 (TWS→Sens): {obs_tws_to_sens:.3f} (95% CI: [{obs_step2_ci[0]:.2f}, {obs_step2_ci[1]:.2f}])")

# 统计有多少模型落在观测的95% CI内
models_in_ci = 0
models_in_ci_list = []
for i, row in pathway_df.iterrows():
    if (obs_step2_ci[0] <= row['step2_tws_to_sens'] <= obs_step2_ci[1] and
        obs_step1_ci[0] <= row['step1_enso_to_tws'] <= obs_step1_ci[1]):
        models_in_ci += 1
        models_in_ci_list.append(row['model'])

print(f"\nModels within Obs. 95% CI: {models_in_ci}/{len(pathway_df)} ({100*models_in_ci/len(pathway_df):.1f}%)")
if models_in_ci > 0:
    print("Models in CI:", ', '.join(models_in_ci_list))


# %%

from scipy.stats import gaussian_kde
# =======================================
# 绘图
# =======================================

# Create the figure
fig, ax = pplt.subplots(ncols=1, nrows=1, journal="nat2", refaspect=1)

# -------------- Plot Observations --------------
# CI95% Y direction
ax.axhspan(
    obs_step1_r - obs_step1_se, 
    obs_step1_r + obs_step1_se,
    color='#FF6B35', 
    alpha=0.25,
    zorder=1,
    linewidth=0,
)

# CI95% X direction
ax.axvspan(
    obs_step2_r - obs_step2_se,
    obs_step2_r + obs_step2_se,
    color='#22A1B2', 
    alpha=0.25,
    zorder=1,
    linewidth=0,
)

# Observation point
ax.plot(
    obs_tws_to_sens,
    obs_enso_to_tws,
    marker='o',
    markersize=10,
    color='black',
    markeredgecolor='w',
    markeredgewidth=1,
    alpha=1,
    zorder=12
)

# Observation error bars - Y direction
ax.errorbar(
    obs_tws_to_sens,
    obs_enso_to_tws,
    xerr=0,
    yerr=obs_step1_se,
    fmt='none',
    color='#FF6B35',
    alpha=1,
    capsize=5,
    capthick=0,
    elinewidth=1.5,
    zorder=10
)

# Observation error bars - X direction
ax.errorbar(
    obs_tws_to_sens,
    obs_enso_to_tws,
    xerr=obs_step2_se,
    yerr=0,
    fmt='none',
    color='#22A1B2',
    alpha=1,
    capsize=5,
    capthick=0,
    elinewidth=1.5,
    label='Observation',
    zorder=10
)

# add the x and y lines
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.6, zorder=0)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.6, zorder=0)

# -------------- Plot Models --------------
for i, row in pathway_df.iterrows():
    # Significance (p < 0.05)
    both_sig = row['step1_sig'] and row['step2_sig']
    
    if both_sig:
        color = 'white'
        edgecolor = 'black'
        alpha = 1
        marker_size = 10
    else:
        color = 'white'
        edgecolor = 'black'
        alpha = 1
        marker_size = 10
    
    x_val = row['step2_tws_to_sens']
    y_val = row['step1_enso_to_tws']
    
    # Error bar colors
    yerr_color = '#FF6B35' if row['step1_sig'] else 'none'
    xerr_color = '#22A1B2' if row['step2_sig'] else 'none'
    
    # Draw Y direction error bar
    ax.errorbar(
        x_val, y_val,
        yerr=row['step1_se'],
        xerr=0, 
        fmt='none',
        ecolor=yerr_color,
        alpha=1,
        capsize=5,
        capthick=0,
        elinewidth=1.5,
        zorder=4
    )
    
    # Draw X direction error bar
    ax.errorbar(
        x_val, y_val,
        xerr=row['step2_se'],
        yerr=0, 
        fmt='none',
        ecolor=xerr_color,
        alpha=1,
        capsize=5,
        capthick=0,
        elinewidth=1.5,
        zorder=4
    )
    
    # Plot the model point
    ax.plot(
        x_val, y_val,
        'o',
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        markersize=marker_size,
        zorder=5
    )

# ----------- Format Main Plot -----------------
# Set the colors of X and Y axis labels
ax.xaxis.label.set_color('#22A1B2') 
ax.yaxis.label.set_color('#FF6B35')

ax.format(
    xlabel='Correlation between antecedent water anomalies and recovery sensitivity',
    ylabel='Correlation between antecedent El Niño intensity and water anomalies',
    labelsize=10, fontweight='bold',
    xlim=(-1, 1), xlocator=0.3,
    ylim=(-1, 1), ylocator=0.3,
    grid=False,
    ytickloc='right',  # ❗️Y轴刻度移到右侧
    ylabelloc='right'  # ❗️Y轴标签也移到右侧
)

# ------------- Density Distribution Inset --------------
# 创建左上角的小图
ax_kde = ax.inset([0.0, 0.65, 0.67, 0.35], zoom=False)  # 改为左上角 [x, y, width, height]

# 准备数据
x_data_step2 = pathway_df["step2_tws_to_sens"].values
x_data_step1 = pathway_df["step1_enso_to_tws"].values

# 创建密度估计
x_range = np.linspace(-1, 1, 200)
kde_step2 = gaussian_kde(x_data_step2)
kde_step1 = gaussian_kde(x_data_step1)
density_step2 = kde_step2(x_range)
density_step1 = kde_step1(x_range)

# 绘制Step 2的KDE曲线（蓝色）
ax_kde.fill_between(
    x_range, 0, density_step2,
    color="#22A1B2",
    alpha=0.3,
    linewidth=0
)
ax_kde.plot(
    x_range, density_step2,
    color="#22A1B2",
    linewidth=2,
)

# 绘制Step 1的KDE曲线（红色）
ax_kde.fill_between(
    x_range, 0, density_step1,
    color="#FF6B35",
    alpha=0.3,
    linewidth=0
)
ax_kde.plot(
    x_range, density_step1,
    color="#FF6B35",
    linewidth=2,
)


# Step 1 (橙色)
step1_mean = x_data_step1.mean()
step1_density = kde_step1(step1_mean)[0]
ax_kde.plot([step1_mean, step1_mean], [0, step1_density], 
            color="#FF6B35", linestyle="--", linewidth=1.5, alpha=0.8)
# Step 2 (蓝色)
step2_mean = x_data_step2.mean()
step2_density = kde_step2(step2_mean)[0]
ax_kde.plot([step2_mean, step2_mean], [0, step2_density], 
            color="#22A1B2", linestyle="--", linewidth=1.5, alpha=0.8)

# 格式化
ax_kde.format(
    xlim=(-1, 1),
    ylim=(0, 2),
    xlabel="Correlation",
    ylabel="Density",
    grid=False,
    xlocator=0.3,
    ylocator=0.3,
    ytickloc='left',  # ❗️Y轴刻度移到右侧
    ylabelloc='left'  # ❗️Y轴标签也移到右侧
)

fig.savefig('03-res/figs/path_decompostion_diagnostic.png', dpi=600)
