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

"""
We here explored the sensitivity of CO2 Growth Rate (CGR) to ENSO phase shifts during carbon recovery periods. 
"""


# %% [markdown]
# # Setup of the Project
#

# %%
# ===============================
#          Load packages
# ===============================

# Numerical computing
import pandas as pd
import numpy as np

# Scientific plotting
import ultraplot as pplt
import seaborn as sns

# Statistical analysis
import pingouin as pg
import statsmodels.api as sm

# Spatial data operations
import xarray as xr
from xarrayutils.utils import linear_trend

# Warnings and errors
import warnings

warnings.filterwarnings("ignore")

# Arial font for plots
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

# Path operations
import os


# %%
# Set project path
proj_path = "/Users/hao/01-RESEARCH/PROGRESS/Carbon_Recovery_and_Climate"
os.chdir(proj_path)

print(f"Working directory: {os.getcwd()}")


# %%
# Load tropical monthly climate and carbon data
# See details in the folder "prep"
df_all = pd.read_csv("01-data/tropical_month_climate_carbon_atlas_mean.csv", index_col=0)
df_all.index = pd.to_datetime(df_all.index)


# %%
# ===============================
#     缺失值标记：2025年后用999
# ===============================
# 将2025-01-01之后的缺失值标记为999
mask_2025_after = df_all.index >= '2025-01-01'
df_all.loc[mask_2025_after] = df_all.loc[mask_2025_after].fillna(999)

print("\n【数据处理】2025-01-01后的缺失值已标记为999")
print("=" * 70)


# %%
df_all


# %%
# ===============================
#         Data Filtering
# ===============================
"""
# Analysis period: 1959-03-01 to 2026-03-01
# Rationale:
# - CGR data starts from 1959-03-01 (data preprocessing limitation)
# - Extended to 2026-02 for robustness check
# - 2025-01后的缺失值标记为999（表示数据不可用）
"""
# Filter the dataset for the specified period
df_all = df_all.loc["1959-03-01":"2026-03-01"]

# Exclude volcanic eruption periods
df_normal = df_all.mask(df_all["eruption"] == 1)

# Print the number of months in each category
print(f"Total months.  : {len(df_all)}")
print(f"Eruption months: {(df_all['eruption'] == 1).sum()}")
print(f"Normal months  : {(df_all['eruption'] == 0).sum()}")


# %%

# %% [markdown]
# # Climatological Climate-carbon Coupling
#

# %% [markdown]
# ## Function definitions
#

# %%
# ===============================
# Statistical Analysis Functions
# ===============================

# ------ 1. Pearson Correlation
def calculate_correlation(X_data, Y_data, conf_level=0.95, min_samples=10):
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
        result = pg.corr(X_clean, Y_clean, method="pearson", confidence=conf_level)
        return {
            "r": result["r"].values[0],
            "p-val": result["p_val"].values[0],
            "CI95%": [result["CI95"][0][0], result["CI95"][0][1]],
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



# %% [markdown]
# ## Correlation Analysis
#

# %%
# ============================================
#   Climatological Lag Correlation Analysis
# ============================================

# Configuration
LAG_RANGE = range(0, -24, -1)  # 0 to -23 months lag
CLIMATE_VARS = {
    "ENSO": "nina34",
    "TMP": "tmp_cru",
    "TWS": "tws_grace",
    "PRE": "pre_gpcc",
}

# Store results
climatology_correlation_results = []

# Calculate correlations for each variable and lag
for var_name, var_col in CLIMATE_VARS.items():

    for lag in LAG_RANGE:
        # Prepare lagged X data and Y data
        """
        When lag = 0: shift(0) -> concurrent (no lead/lag)
        When lag = -1: shift(1) -> climate variable leads CGR by 1 month
        When lag = -23: shift(23) -> climate variable leads CGR by 23 months
        """
        X_data = df_normal[var_col].shift(-lag)
        Y_data = df_normal["CGR"]

        # Calculate correlation
        result = calculate_correlation(X_data, Y_data, conf_level=0.95, min_samples=10)

        # Store results (successful or failed)
        if result is not None:
            climatology_correlation_results.append(
                {
                    "Variable": var_name,
                    "Lag": lag,
                    "Correlation": result["r"],
                    "P_value": result["p-val"],
                    "CI_lower": result["CI95%"][0],
                    "CI_upper": result["CI95%"][1],
                    "N_samples": result["n_samples"],
                }
            )
        else:
            # Data insufficient - record as NaN
            climatology_correlation_results.append(
                {
                    "Variable": var_name,
                    "Lag": lag,
                    "Correlation": float("nan"),
                    "P_value": float("nan"),
                    "CI_lower": float("nan"),
                    "CI_upper": float("nan"),
                    "N_samples": 0,
                }
            )

# Convert to DataFrame
climatology_correlation_results = pd.DataFrame(climatology_correlation_results)

climatology_correlation_results


# %%
# Create figure for climatological lag correlation analysis
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=2, journal="nat2")

CLIMATE_COLORS = {
    "ENSO": "#B22222",
    "TMP": "#DAA520",
    "TWS": "#4682B4",
    "PRE": "#2E8B57",
}

# Plot lines for each variable
for var_name in CLIMATE_VARS.keys():

    # Filter data for the current variable
    var_data = climatology_correlation_results[
        climatology_correlation_results["Variable"] == var_name
    ]

    # Extract data for plotting
    lags = var_data["Lag"].values
    correlations = var_data["Correlation"].values
    ci_lower = var_data["CI_lower"].values
    ci_upper = var_data["CI_upper"].values

    # Plot correlation line
    ax.plot(
        lags, correlations, color=CLIMATE_COLORS[var_name], label=var_name, linewidth=2
    )

    # Add confidence interval as shaded area
    ax.fill_between(
        lags,
        ci_lower,
        ci_upper,
        color=CLIMATE_COLORS[var_name],
        alpha=0.2,
        linewidth=0,
        zorder=1,
    )

    # Find and mark optimal lag
    if var_name in [
        "PRE",
        "TWS",
    ]:  # Precipitation and TWS take most negative correlation
        best_idx = var_data["Correlation"].idxmin()
    else:  # ENSO and TMP take most positive correlation
        best_idx = var_data["Correlation"].idxmax()

    best_row = var_data.loc[best_idx]

    # Mark optimal lag with large circle
    ax.scatter(
        best_row["Lag"],
        best_row["Correlation"],
        color=CLIMATE_COLORS[var_name],
        s=100,
        marker="o",
        edgecolor="black",
        linewidth=2,
        zorder=5,
    )

    # Add text annotation for optimal lag
    ax.text(
        best_row["Lag"],
        best_row["Correlation"] + 0.05,
        f'{best_row["Lag"]}',
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=CLIMATE_COLORS[var_name],
    )

ax.legend(
    loc="t",
    ncols=4,
    frame=False,
)

# Format axes
ax.format(
    xlabel="Time lag (months)",
    ylabel="Correlation coefficient",
    ylim=(-0.9, 0.9),
    xlim=(-23, 1),
    xlocator=2,
    xminorlocator=1,
    grid=False,
    xtickminor=True,
)


# %% [markdown]
# ## Regression Analysis
#

# %%
# ============================================
#   Climatological Regression Analysis
# ============================================

# Configuration
OPTIMAL_LAGS = {
    "ENSO": -4,  # ENSO leads CGR by 4 months
    "TMP": 0,  # TMP concurrent with CGR
    "TWS": 0,  # TWS concurrent with CGR
    "PRE": -5,  # PRE lags CGR by 5 months
}

CLIMATE_VARS = {"ENSO": "nina34", 
                "TMP": "tmp_cru", 
                "TWS": "tws_grace",
                "PRE": "pre_gpcc"}

# ============================================
# 1. Climate --> Carbon Regression Analysis

# Climate variables vs CGR
carbon_regression_results = []

for var_name, var_col in CLIMATE_VARS.items():
    lag = OPTIMAL_LAGS[var_name]
    X_data = df_normal[var_col].shift(-lag).values
    Y_data = df_normal["CGR"].values

    result = calculate_regression(X_data, Y_data, conf_level=0.95, min_samples=10)

    carbon_regression_results.append(
        {
            "Analysis": f"{var_name}_vs_CGR",
            "Y_var": "CGR",
            "X_var": var_name,
            "Lag": lag,
            "Coefficient": result["coef"],
            "P_value": result["pvalue"],
            "R_squared": result["r2"],
            "CI_lower": result["ci_lower"],
            "CI_upper": result["ci_upper"],
            "N_samples": result["n_samples"],
        }
    )

# ========================================
# 2. ENSO --> Climate Regression Analysis

# ENSO vs Climate variables
climate_regression_results = []

for var_name in ["TWS", "TMP", "PRE"]:

    var_lag = OPTIMAL_LAGS[var_name]
    enso_lag = OPTIMAL_LAGS["ENSO"]

    # Calculate relative lag to ENSO
    if enso_lag < var_lag:
        relative_lag = enso_lag - var_lag  
    else:
        relative_lag = 0

    X_data = df_normal["nina34"].shift(-relative_lag).values
    Y_data = df_normal[CLIMATE_VARS[var_name]].values

    result = calculate_regression(X_data, Y_data, conf_level=0.95, min_samples=10)

    climate_regression_results.append(
        {
            "Analysis": f"{var_name}_vs_ENSO",
            "Y_var": var_name,
            "X_var": "ENSO",
            "Relative_Lag": relative_lag,
            "Coefficient": result["coef"],
            "P_value": result["pvalue"],
            "R_squared": result["r2"],
            "CI_lower": result["ci_lower"],
            "CI_upper": result["ci_upper"],
            "N_samples": result["n_samples"],
        }
    )

# ==================================
# 3. Combination of Results

climatology_regression_results = pd.DataFrame(carbon_regression_results + climate_regression_results)

print(
    climatology_regression_results[
        ["Analysis", "Coefficient", "P_value", "R_squared", "N_samples"]
    ].round(3)
)


# %%

# %% [markdown]
# # Carbon Recovery Episodes
#

# %% [markdown]
# ## Thresholds for identification
#

# %%
def find_cgr_extrema(
    df, var, prominence=0.1, distance=12, width=None, height=None, threshold=None
):
    """
    Find local maxima and minima in the CGR time series

    Parameters:
        df: DataFrame containing the variable to analyze
        var: str
            Name of the variable column to analyze (e.g., 'CGR')
        prominence: float, optional
            The prominence of peaks - how much a peak stands out from its surrounding baseline (default: 0.1)
        distance: int, optional
            Minimum number of samples between adjacent peaks (default: 12 months)
        width: float or None, optional
            Required width of peaks in samples (default: None)
        height: float, optional
            Required height of peaks (default: 1)
        threshold: float or None, optional
            Required threshold of peak, the vertical distance to its neighboring samples (default: None)

    Returns:
        max_indices: Indices of local maxima
        min_indices: Indices of local minima
    """
    # load image processing
    from scipy.signal import find_peaks

    # Extract values from the specified variable column
    values = df[var].values

    # find the maximum values
    max_indices, _ = find_peaks(
        values,
        prominence=prominence,
        distance=distance,
        width=width,
        height=height,
        threshold=threshold,
    )

    # find the minimum values
    min_indices, _ = find_peaks(
        -values,
        prominence=prominence,
        distance=distance,
        width=width,
        height=height,
        threshold=threshold,
    )

    return max_indices, min_indices



# %%
def create_max_to_min_groups(df, max_indices, min_indices):
    """
    Create data groups from CGR maxima to minima

    Parameters:
        df: DataFrame containing 'time' and 'CGR' columns
        max_indices: Indices of CGR local maxima
        min_indices: Indices of CGR local minima

    Returns:
        max_to_min_groups: List of DataFrames containing data for each maxima to minima interval
        group_info: DataFrame containing information about each group
    """
    max_to_min_groups = []

    # For each maximum, find the next minimum
    for i, max_idx in enumerate(max_indices):

        # Find all minima after this maximum
        next_mins = min_indices[min_indices > max_idx]

        if len(next_mins) > 0:
            # Select the first minimum after the maximum
            min_idx = next_mins[0]

            # Extract data for this interval (including endpoints)
            group_data = df.iloc[max_idx : min_idx + 1].copy()

            # Append the group to the list
            max_to_min_groups.append(group_data)

    return max_to_min_groups



# %%
# Find the local maxima and minima
max_locations, min_locations = find_cgr_extrema(
    df_all, var="CGR", prominence=0.1, distance=12, width=2, height=0.5, threshold=None
)

print("Maximum's Locations:", len(max_locations), "in total", "\n", max_locations)
print("Minimum's Locations:", len(min_locations), "in total", "\n", min_locations)


# %%
# Find the max_to_min groups
max_to_min_groups = create_max_to_min_groups(
    df_all, max_indices=max_locations, min_indices=min_locations
)


# %%
# Select the groups that exclude eruption years and have more than 12 data points

# Set the minimum number of data points
sample_number = 12

# Filter out groups that contain eruption years and ensure normal groups have more than 12 data points
recovery_groups = []
eruption_groups = []

for i, group in enumerate(max_to_min_groups):
    if (group["eruption"] == 1).any():
        eruption_groups.append(group)
    elif (
        group.shape[0] > sample_number
    ):  # Only include groups with more than 12 data points
        recovery_groups.append(group)

# Add a group from 1984-01-01 to 1986-03-01
manual_group_1 = df_normal.loc["1984-01-01":"1986-03-01"].copy()
if manual_group_1.shape[0] > sample_number and (manual_group_1["eruption"] == 0).all():
    recovery_groups.append(manual_group_1)

# Sort the recovery groups by their start date
recovery_groups = sorted(recovery_groups, key=lambda g: g.index[0])


# %%
# Revise the groups manually
if True:
    # Remove specific groups according to the lag correlation analysis between CGR and ENSO
    recovery_groups.pop(-7)
    recovery_groups.pop(-4)
    recovery_groups.pop(0)


# %%
manual_group_2024 = df_normal.loc["2025-01-01":].copy()
if manual_group_2024.shape[0] > 0:
    recovery_groups.append(manual_group_2024)


# %%
recovery_groups[-1] 


# %%
print(f"Total groups: {len(max_to_min_groups)}")
print(f"Groups with eruptions: {len(eruption_groups)}")
print(
    f"Groups without eruptions and > {sample_number} data points: {len(recovery_groups)}"
)

# Example of accessing the filtered groups
print(10 * "=====")
if len(recovery_groups) > 0:
    print(f"First non-eruption group with > {sample_number} data points:")
    print(recovery_groups[-1].head())


# %%
recovery_groups[-1]


# %%

# %% [markdown]
# ## Data visualiztion
#

# %%
# ---- Plot the changes of CGR, water and heat variables ----

# ===============================
# CONFIGURATION SECTION
# ===============================

# Time periods and eruption events
ERUPTION_PERIODS = {
    "Mount Agung": ("1962-01-01", "1963-12-31"),
    "El Chichón": ("1982-01-01", "1982-12-31"),
    "Pinatubo": ("1992-01-01", "1993-12-31"),
}

ERUPTION_LABELS = [
    ("1962-12-31", "Mt. Agung"),
    ("1982-12-31", "El Chichón"),
    ("1992-12-31", "Pinatubo"),
]

# Color scheme - easy to adjust!
COLORS = {
    "cgr": "#CFCECE",  # Black color for CGR
    "sst": "#B22222",  # Red for SST
    "tws": "#4d9221",
    "temperature": "#DAA520",
    "eruption": "#000000",
    "eruption_text": "#000000",
}

# Plot styling
STYLE = {
    "linewidth": 2,
    "cgr_linewidth": 2,
    "markersize": 0,
    "font_size": 9,
    "small_font": 8,
    "eruption_alpha": 0.2,
    "enso_alpha": 0.75,
}

# Panel Y-axis configuration
Y_CONFIGS = {
    "tws": {"ylim": (-35, 35), "ylocator": 20, "ylabel": "TWS [mm]"},
    "cgr": {"ylim": (4.2, -4.2), "ylocator": 3, "ylabel": "CGR [Gt C]"},
    "temp": {"ylim": (-0.6, 0.6), "ylocator": 0.4, "ylabel": "TMP [°C]"},
}
"""
Notes: 
1 Gt = 10^9 ton
1 Pg = 10^15 gram = 10^9 ton
"""

# Base axis format
AXIS_FORMAT = {
    "xlim": (np.datetime64(df_all["time"][0]), np.datetime64(df_all["time"][-1])+24),
    "xlocator": ("year", 2),
    "xformatter": "%Y",
    "xminorlocator": ("year", 1),
    "ygrid": False,
    "xgrid": False,
}

# ENSO settings
ENSO_RANGE = 2.5

# ===============================
# CREATE FIGURE
# ===============================

fig, axs = pplt.subplots(
    nrows=3, ncols=1, refaspect=4, share=False, space=0.001, journal="nat2"
)

ax_top, ax_mid, ax_bottom = axs[0], axs[1], axs[2]

# ===============================
# TOP PANEL: TWS
# ===============================

# Setup left axis (hidden)
ax_top.format(ylim=(5.5, -5.5), ylocator=[], xlabel="", xticklabels=[], **AXIS_FORMAT)

# TWS on right axis
ax_tws = ax_top.dualy(funcscale="linear")
ax_tws.plot(
    df_all.index,
    df_all["tws_grace"].values,
    "-o",
    ms=STYLE["markersize"],
    lw=STYLE["linewidth"],
    color=COLORS["tws"],
)
ax_tws.format(
    ylim=Y_CONFIGS["tws"]["ylim"],
    ylocator=Y_CONFIGS["tws"]["ylocator"],
    ycolor=COLORS["tws"],
    **AXIS_FORMAT
)

# TWS label
ax_tws.text(
    1.07,
    0.5,
    Y_CONFIGS["tws"]["ylabel"],
    transform=ax_tws.transAxes,
    ha="left",
    va="center",
    color=COLORS["tws"],
    rotation=90,
    fontsize=STYLE["font_size"],
)

# ===============================
# MIDDLE PANEL: CGR
# ===============================

ax_mid.plot(
    df_all.index,
    df_all["CGR"].values,
    ms=STYLE["markersize"],
    lw=1,
    alpha=1,
)
ax_mid.format(
    ylim=Y_CONFIGS["cgr"]["ylim"],
    ylocator=Y_CONFIGS["cgr"]["ylocator"],
    xlabel="",
    xticklabels=[],
    **AXIS_FORMAT
)

# CGR directional labels
ax_mid.text(
    -0.07,
    0.15,
    "More net\nrelease\n←",
    transform=ax_mid.transAxes,
    ha="center",
    va="center",
    color="gray",
    rotation=90,
    fontsize=STYLE["small_font"],
)
ax_mid.text(
    -0.07,
    0.85,
    "More net\nuptake\n→",
    transform=ax_mid.transAxes,
    ha="center",
    va="center",
    color="gray",
    rotation=90,
    fontsize=STYLE["small_font"],
)
ax_mid.text(
    -0.13,
    0.5,
    Y_CONFIGS["cgr"]["ylabel"],
    transform=ax_mid.transAxes,
    ha="center",
    va="center",
    color="#000000",
    rotation=90,
    fontsize=STYLE["font_size"],
)

# ===============================
# BOTTOM PANEL: TEMPERATURE
# ===============================

# Setup left axis (hidden)
ax_bottom.format(ylim=(5.5, -5.5), ylocator=[], xlabel="", xrotation=90, **AXIS_FORMAT)

# Temperature on right axis
ax_temp = ax_bottom.dualy(funcscale="linear")
ax_temp.plot(
    df_all.index,
    df_all["tmp_cru"].values,
    "-o",
    ms=STYLE["markersize"],
    lw=STYLE["linewidth"],
    color=COLORS["temperature"],
)
ax_temp.format(
    ylim=Y_CONFIGS["temp"]["ylim"],
    ylocator=Y_CONFIGS["temp"]["ylocator"],
    ycolor=COLORS["temperature"],
    **AXIS_FORMAT
)

# Temperature label
ax_temp.text(
    1.07,
    0.5,
    Y_CONFIGS["temp"]["ylabel"],
    transform=ax_temp.transAxes,
    ha="left",
    va="center",
    color=COLORS["temperature"],
    rotation=90,
    fontsize=STYLE["font_size"],
)

# ===============================
# ADD RECOVERY EVENTS
# ===============================

# Create a list
circled_numbers = [str(i + 1) for i in range(len(recovery_groups))]  # More Pythonic way

for group, circled_number in zip(recovery_groups, circled_numbers):
    # Adjust font size and padding based on number of digits
    if len(circled_number) == 1:  # Single digit (1-9)
        fontsize = 7
        pad = 0.150
    else:  # Double digit (10-13)
        fontsize = 6.5
        pad = 0.150

    # Draw thick black line for each recovery period
    ax_mid.plot(
        group.index,
        group["CGR"].values,
        color="#000000",
        linewidth=STYLE["cgr_linewidth"],
        linestyle="-",
        marker="o",
        ms=0,
        zorder=5,
    )

    # Add numbered circle labels at center of each recovery period
    mid_point = group.index[len(group.index) // 2]
    ax_mid.text(
        mid_point,
        -4,
        str(circled_number),
        ha="left",
        va="center",
        color="k",
        fontsize=fontsize,
        zorder=100,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="circle"),
    )

# ===============================
# CLEAN UP PANEL SPINES
# ===============================

# === hide spines ===
if True:
    # top：only keep left and right borders
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    # mid：only keep left border
    ax_mid.spines["top"].set_visible(False)
    ax_mid.spines["bottom"].set_visible(False)
    ax_mid.spines["right"].set_visible(False)

    # bottom：only keep left and right borders
    ax_bottom.spines["top"].set_visible(False)

    # TWS and TMP panels
    ax_tws.spines["bottom"].set_visible(False)
    ax_temp.spines["top"].set_visible(False)

    # === hide left Y-axis for top and bottom panels ===
    for ax in [ax_top, ax_bottom]:
        ax.spines["left"].set_visible(False)  # hide left spine
        ax.tick_params(left=False, labelleft=False)  # hide Y-axis ticks and labels

# ===============================
# ADD BACKGROUND ELEMENTS
# ===============================

all_axes = [ax_top, ax_mid, ax_bottom]

# Eruption backgrounds
# for ax in all_axes:
#     for eruption, (start, end) in ERUPTION_PERIODS.items():
#         ax.axvspan(np.datetime64(start), np.datetime64(end),
#                   alpha=STYLE['eruption_alpha'], color=COLORS['eruption'],
#                   linewidth=0, zorder=0)

# Eruption labels (bottom panel only)
for date, name in ERUPTION_LABELS:
    ax_temp.text(
        np.datetime64(date),
        -0.55,
        name,
        ha="center",
        rotation=-270,
        fontsize=STYLE["small_font"],
        color=COLORS["eruption_text"],
        zorder=20,
    )

# ENSO background coloring (blue=La Niña, red=El Niño)              
normalized_abs = ENSO_RANGE  # 2.5                                
for i in range(len(df_all)):                                        
    t_month = df_all.index[i]    
    t_next  = t_month + pd.DateOffset(months=1)                     
    value   = df_all["nina34"].iloc[i]      
                                                                    
    if np.isnan(value):                
        continue  # 跳过NaN                                         
    elif value < 0:  # La Niña   
        color = pplt.Colormap('Blues')(abs(value) / normalized_abs) 
    else:  # El Niño                        
        color = pplt.Colormap('Reds')(abs(value) / normalized_abs)
                                
    for ax in all_axes:                                             
        ax.axvspan(t_month, t_next, color=color,
alpha=STYLE["enso_alpha"], linewidth=0, zorder=2) 

# ENSO colorbar
cmap_enso = pplt.Colormap("Blues_r", "Reds", alpha=STYLE["enso_alpha"])
scatter = ax_top.scatter(
    df_all.index,
    df_all["nina34"].values,
    c=df_all["nina34"].values,
    cmap=cmap_enso,
    vmin=-ENSO_RANGE,
    vmax=ENSO_RANGE,
    s=0,
    zorder=100,
    extend="both",
)

ax_mid.colorbar(
    scatter,
    loc="r",
    width=0.1,
    shrink=0.80,
    ticklabelsize=STYLE["small_font"],
    label="SSTA [°C]",
)

# ===============================
# SAVE FIGURE
# ===============================
# fig.savefig("03-res/figs/SP_carbon_and_climate_changes.png", dpi=600)


# %%
# ==============================
# Plot Carbon Recovery Periods (Only Middle Panel)
# ==============================

# Create figure
fig, ax = pplt.subplots(ncols=1, nrows=1, refaspect=3, journal="nat2")

# Plot ENSO background coloring (blue=La Niña, red=El Niño)
total_months = len(df_all)
normalized_abs = 2.5  # Color scale limit for ENSO values

for i in range(total_months):
    xmin, xmax = i / total_months, (i + 1) / total_months
    value = df_all["nina34"].iloc[i]

    # Determine background color based on ENSO state
    if abs(value) < 0 or np.isnan(value):
        color = "white"
    elif value < 0:  # La Niña (negative values)
        color = pplt.Colormap("Blues")(abs(value) / normalized_abs)
    else:  # El Niño (positive values)
        color = pplt.Colormap("Reds")(abs(value) / normalized_abs)

    ax.axhspan(
        -10,
        10,
        xmin=xmin,
        xmax=xmax,
        color=color,
        alpha=STYLE["enso_alpha"],
        linewidth=0,
        edgecolor=color,
        zorder=1,
    )

# Add ENSO colorbar for reference
cmap_enso = pplt.Colormap("Blues_r", "Reds", alpha=STYLE["enso_alpha"])
scatter = ax.scatter(
    df_all.index,
    df_all["nina34"],
    c=df_all["nina34"],
    cmap=cmap_enso,
    vmin=-normalized_abs,
    vmax=normalized_abs,
    s=0,
    zorder=100,
    extend="both",
)  # markers are not shown
ax.colorbar(scatter, loc="r", width=0.1, shrink=0.90, ticklabelsize=8, label="SSTA [°C]")

# Plot baseline CGR time series (gray line)
ax.plot(df_all.index, df_all["CGR"], "-", color=COLORS["cgr"], alpha=1, lw=1.5, zorder=5)

# Highlight carbon recovery periods (thick black lines with numbered labels)
for i, (group, circled_number) in enumerate(zip(recovery_groups, circled_numbers)):
    # Draw thick black line for each recovery period
    ax.plot(
        group.index,
        group["CGR"],
        color="k",
        lw=2,
        linestyle="-",
        marker="o",
        ms=0,
        zorder=5,
    )

    # Add numbered circle labels at center of each recovery period
    mid_point = group.index[len(group.index) // 2]
    # Adjust font size and padding based on number of digits
    if len(circled_number) == 1:  # Single digit (1-9)
        fontsize = 6.7
        pad = 0.150
    else:  # Double digit (10-13)
        fontsize = 6.5
        pad = 0.135
    ax.text(
        mid_point,
        -4.75,
        circled_number,
        ha="center",
        va="center",
        color="k",
        fontsize=fontsize,
        zorder=100,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="circle"),
    )

# Label major volcanic eruptions
eruptions = [
    ("1962-12-31", "Mt. Agung"),
    ("1982-12-31", "El Chichón"),
    ("1992-12-31", "Pinatubo"),
]
for date, name in eruptions:
    ax.text(
        np.datetime64(date),
        5.05,
        name,
        ha="center",
        rotation=-270,
        fontsize=8,
        color="grey",
        fontweight="bold",
    )

# Add descriptive labels for CGR direction
ax.text(
    -0.06,
    0.25,
    "More net\nrelease\n←",
    transform=ax.transAxes,
    ha="center",
    va="center",
    color="gray",
    rotation=90,
    fontsize=8,
)
ax.text(
    -0.06,
    0.75,
    "More net\nuptake\n→",
    transform=ax.transAxes,
    ha="center",
    va="center",
    color="gray",
    rotation=90,
    fontsize=8,
)
ax.text(
    -0.12,
    0.5,
    "CGR [Gt C]",
    transform=ax.transAxes,
    ha="center",
    va="center",
    color="k",
    rotation=90,
    fontsize=9,
)

# Format axes and appearance
ax.format(
    ylabel="",
    xlabel="",
    ylim=(5.5, -5.5),
    xlim=(np.datetime64(df_all["time"][0]), np.datetime64(df_all["time"][-1])),
    xlocator=("year", 5),
    xformatter="%Y",
    xminorlocator=("year", 1),
    xrotation=90,
    grid=False,
)

# Export figure
# fig.savefig("03-res/figs/carbon_recovery_periods.png", dpi=600)


# %%
g13 = recovery_groups[-1]  
new_start = g13.index[0] + pd.DateOffset(months=9)   
recovery_groups[-1] = g13.loc[new_start:]   

recovery_groups[-1]


# %%

# %% [markdown]
# # Recovery Stability to ENSO
#

# %% [markdown]
# ## Best lags during carbon recovery
#

# %%
# Configuration
lag_range_new = range(0, -13, -1)  # [0, -1, -2, ..., -12] months lag

climate_vars = {
    "ENSO": "nina34",
    "NINA3": "nina3",
    "NINA4": "nina4",
    "TMP": "tmp_cru",
    "PRE": "pre_gpcc",
    "TWS": "tws_grace",
}

# Store results using new method
recovery_correlation_results = []

# Analyze correlation for each recovery group using new method
for i, group in enumerate(recovery_groups):
    group_id = i + 1
    start_date, end_date = group.index[0], group.index[-1]

    # Get Y data (CGR) for this group
    Y_data = group["CGR"]

    # Test each climate variable at each lag
    for var_name, var_col in climate_vars.items():
        for lag in lag_range_new:

            # Calculate shifted time window for X variable
            X_start = start_date - pd.DateOffset(months=-lag)
            X_end = end_date - pd.DateOffset(months=-lag)

            # Check if shifted X data range is within available data bounds
            if X_start < df_normal.index[0] or X_end > df_normal.index[-1]:
                result = None
            else:
                # Extract X data for the shifted time period
                X_data = df_normal.loc[X_start:X_end, var_col]

                # Use the new improved correlation function
                result = calculate_correlation(
                    X_data, Y_data, conf_level=0.95, min_samples=10
                )

            # Store results
            if result is not None:
                recovery_correlation_results.append(
                    {
                        "group_id": group_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "duration": len(group),
                        "climate_var": var_name,
                        "lag": lag,
                        "correlation": result["r"],
                        "p_value": result["p-val"],
                        "CI_lower": result["CI95%"][0],
                        "CI_upper": result["CI95%"][1],
                        "n_samples": result["n_samples"],
                        "X_start": X_start,
                        "X_end": X_end,
                    }
                )
            else:
                # Store NaN for insufficient data
                recovery_correlation_results.append(
                    {
                        "group_id": group_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "duration": len(group),
                        "climate_var": var_name,
                        "lag": lag,
                        "correlation": float("nan"),
                        "p_value": float("nan"),
                        "CI_lower": float("nan"),
                        "CI_upper": float("nan"),
                        "n_samples": 0,
                        "X_start": X_start if "X_start" in locals() else None,
                        "X_end": X_end if "X_end" in locals() else None,
                    }
                )

# Convert to DataFrame
recovery_correlation_results = pd.DataFrame(recovery_correlation_results)

recovery_correlation_results


# %%
# Load packages for plotting
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# Plot correlation heatmaps for each climate variable
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
climate_vars = ["ENSO", "NINA3", "NINA4"]
cmap = "coolwarm"

for i, var in enumerate(climate_vars):
    # Prepare data for this climate variable
    var_data = recovery_correlation_results[
        recovery_correlation_results["climate_var"] == var
    ].copy()
    pivot_data = var_data.pivot(index="group_id", columns="lag", values="correlation")
    pivot_pval = var_data.pivot(index="group_id", columns="lag", values="p_value")

    # Create correlation heatmap
    sns.heatmap(
        pivot_data,
        annot=False,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.75,
        cbar=False,
        ax=axes[i],
    )

    # Mark significant correlations (p < 0.05)
    for row_idx, group_id in enumerate(pivot_data.index):
        for col_idx, lag in enumerate(pivot_data.columns):
            p_val = pivot_pval.loc[group_id, lag]
            if p_val < 0.05:
                axes[i].text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    "●",
                    ha="center",
                    va="center",
                    color="none",
                    fontsize=0,
                    zorder=5,
                )

    # Mark optimal lag for each group (strongest correlation within -12 to 0 months)
    for row_idx, group_id in enumerate(pivot_data.index):
        valid_lags = [lag for lag in pivot_data.columns if -12 <= lag <= 0]
        row_data = pivot_data.loc[group_id, valid_lags]

        # Find maximum correlation for ENSO/TMP, minimum for PRE/TWS
        if var in ["ENSO", "NINA3", "NINA4", "TMP"]:
            target_idx = row_data.idxmax()
        else:  # PRE, TWS (negative correlations expected)
            target_idx = row_data.idxmin()

        if pd.isna(target_idx):
            continue
        col_idx = list(pivot_data.columns).index(target_idx)
        axes[i].text(
            col_idx + 0.5,
            row_idx + 0.2,
            ".",
            fontsize=20,
            ha="center",
            va="center",
            color="k",
            fontweight="bold",
            zorder=10,
        )

    axes[i].set_xlabel("")
    axes[i].minorticks_off()

axes[0].set_ylabel("Carbon recovery episodes")
axes[1].set_ylabel("")
axes[2].set_ylabel("")
# Format subplot
axes[0].set_title(f"Niño 3.4 leads CGR by (months)\n(●, Optimal lag)", color="black")
axes[1].set_title(f"Niño 3 leads CGR by (months)\n(●, Optimal lag)", color="black")
axes[2].set_title(f"Niño 4 leads CGR by (months)\n(●, Optimal lag)", color="black")

# Add shared colorbar
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
smp = ScalarMappable(norm=norm, cmap=cmap)
smp.set_array([])

plt.subplots_adjust(bottom=0.10)
cbar = fig.colorbar(
    smp,
    ax=axes,
    orientation="horizontal",
    shrink=0.95,
    aspect=45,
    pad=0.2,
    extend="both",
    label="Correlation coefficient",
)

# Export figure
# fig.savefig("03-res/figs/SP_best_lags_with_ENSO_during_recovery.png", dpi=600)


# %%
# Plot correlation heatmaps for each climate variable
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
climate_vars = ["TMP", "TWS", "PRE"]
cmap = "coolwarm"

for i, var in enumerate(climate_vars):
    # Prepare data for this climate variable
    var_data = recovery_correlation_results[
        recovery_correlation_results["climate_var"] == var
    ].copy()
    pivot_data = var_data.pivot(index="group_id", columns="lag", values="correlation")
    pivot_pval = var_data.pivot(index="group_id", columns="lag", values="p_value")

    # Create correlation heatmap
    sns.heatmap(
        pivot_data,
        annot=False,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.75,
        cbar=False,
        ax=axes[i],
    )

    # Mark significant correlations (p < 0.05)
    for row_idx, group_id in enumerate(pivot_data.index):
        for col_idx, lag in enumerate(pivot_data.columns):
            p_val = pivot_pval.loc[group_id, lag]
            if p_val < 0.05:
                axes[i].text(
                    col_idx + 0.5,
                    row_idx + 0.2,
                    ".",
                    fontsize=20,
                    ha="center",
                    va="center",
                    color="none",
                    zorder=5,
                )

    # Mark optimal lag for each group (strongest correlation within -12 to 0 months)
    for row_idx, group_id in enumerate(pivot_data.index):
        valid_lags = [lag for lag in pivot_data.columns if -12 <= lag <= 0]
        row_data = pivot_data.loc[group_id, valid_lags]

        # Find maximum correlation for ENSO/TMP, minimum for PRE/TWS
        if var in ["ENSO", "NINA3", "NINA4", "TMP"]:
            target_idx = row_data.idxmax()
        else:  # PRE, TWS (negative correlations expected)
            target_idx = row_data.idxmin()

        if pd.isna(target_idx):
            continue
        col_idx = list(pivot_data.columns).index(target_idx)
        axes[i].text(
            col_idx + 0.50,
            row_idx + 0.50,
            ".",
            ha="center",
            va="center",
            color="k",
            fontweight="bold",
            zorder=2,
        )

    # Format subplot
    axes[i].set_title(f"{var} leads CGR by (months)\n(●, Optimal lag)", color="black")

    axes[i].set_xlabel("")
    
    axes[i].minorticks_off()

axes[0].set_ylabel("Carbon recovery episodes")
axes[1].set_ylabel("")
axes[2].set_ylabel("")
# Add shared colorbar
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
smp = ScalarMappable(norm=norm, cmap=cmap)
smp.set_array([])

plt.subplots_adjust(bottom=0.10)
cbar = fig.colorbar(
    smp,
    ax=axes,
    orientation="horizontal",
    shrink=0.95,
    aspect=45,
    pad=0.2,
    extend="both",
    label="Correlation coefficient",
)

# Export figure
# fig.savefig("03-res/figs/best_lags_with_climate_during_recovery.png", dpi=1000)


# %%
# Extract optimal lag and correlation for each climate variable
# Purpose: Find the lag with strongest correlation for each group-variable combination
# This will be used for subsequent regression analysis

optimal_correlation_results = []

for group_id in recovery_correlation_results["group_id"].unique():
    # Filter data for this group within valid lag range (-12 to 0 months)
    group_data = recovery_correlation_results[
        (recovery_correlation_results["group_id"] == group_id)
        & (recovery_correlation_results["lag"] >= -12)
        & (recovery_correlation_results["lag"] <= 0)
    ]

    row = {"group_id": group_id}

    # Find optimal lag for each climate variable
    for var in ["ENSO", "NINA3", "NINA4", "TMP", "PRE", "TWS"]:
        var_data = group_data[group_data["climate_var"] == var]

        if len(var_data) > 0 and not var_data["correlation"].isna().all():
            # For ENSO/TMP: find maximum positive correlation
            # For PRE/TWS: find maximum negative correlation (minimum value)
            if var in ["ENSO", "NINA3", "NINA4", "TMP"]:
                best_idx = var_data["correlation"].idxmax()
            else:  # PRE, TWS
                best_idx = var_data["correlation"].idxmin()

            if pd.isna(best_idx):
                continue
            best_result = var_data.loc[best_idx]

            # Store optimal parameters for this variable
            row[f"{var}_corr"] = best_result["correlation"]
            row[f"{var}_lag"] = best_result["lag"]
            row[f"{var}_sig"] = best_result["p_value"] #< 0.05

    optimal_correlation_results.append(row)

# Convert to DataFrame for easy access
recovery_optimal_correlation = pd.DataFrame(optimal_correlation_results).sort_values(
    "group_id"
)

recovery_optimal_correlation


# %%
print(recovery_optimal_correlation['ENSO_corr'].mean()) 

print(recovery_optimal_correlation['ENSO_corr'].std()) 


# %%
# =============================================
# Here we want to aggregate the recovery data
# into a single DataFrame with climate variables
# for each recovery group.
# This will allow us to analyze the relationship
# between carbon recovery and climate variables.
# =============================================

# Initialize an empty list to store aggregated data
aggregated_data = []

# Directly iterate over each recovery group
for i, group in enumerate(recovery_groups):
    group_id = i + 1
    start_date, end_date = group.index[0], group.index[-1]

    # Get the optimal lags for this group
    group_optimal = recovery_optimal_correlation[recovery_optimal_correlation["group_id"] == group_id]
    if len(group_optimal) == 0:
        continue

    def safe_lag(val, default=0):
        return int(val) if not pd.isna(val) else default

    enso_lag  = safe_lag(group_optimal["ENSO_lag"].iloc[0])
    nina3_lag = safe_lag(group_optimal["NINA3_lag"].iloc[0])
    nina4_lag = safe_lag(group_optimal["NINA4_lag"].iloc[0])
    tmp_lag   = safe_lag(group_optimal["TMP_lag"].iloc[0])
    tws_lag   = safe_lag(group_optimal["TWS_lag"].iloc[0])
    pre_lag   = safe_lag(group_optimal["PRE_lag"].iloc[0])

    # Calculate time windows for climate data
    enso_start = start_date - pd.DateOffset(months=-enso_lag)
    enso_end = end_date - pd.DateOffset(months=-enso_lag)

    nina3_start = start_date - pd.DateOffset(months=-nina3_lag)
    nina3_end = end_date - pd.DateOffset(months=-nina3_lag)

    nina4_start = start_date - pd.DateOffset(months=-nina4_lag)
    nina4_end = end_date - pd.DateOffset(months=-nina4_lag)

    tmp_start = start_date - pd.DateOffset(months=-tmp_lag)
    tmp_end = end_date - pd.DateOffset(months=-tmp_lag)

    tws_start = start_date - pd.DateOffset(months=-tws_lag)
    tws_end = end_date - pd.DateOffset(months=-tws_lag)

    pre_start = start_date - pd.DateOffset(months=-pre_lag)
    pre_end = end_date - pd.DateOffset(months=-pre_lag)

    # Extract CGR data for this group
    cgr_data = group["CGR"]

    # Extract climate data (check availability)
    if enso_start >= df_normal.index[0] and enso_end <= df_normal.index[-1]:
        enso_data = df_normal.loc[enso_start:enso_end, "nina34"]
    else:
        enso_data = pd.Series([np.nan] * len(cgr_data), index=cgr_data.index)

    if nina3_start >= df_normal.index[0] and nina3_end <= df_normal.index[-1]:
        nina3_data = df_normal.loc[nina3_start:nina3_end, "nina3"]
    else:
        nina3_data = pd.Series([np.nan] * len(cgr_data), index=cgr_data.index)

    if nina4_start >= df_normal.index[0] and nina4_end <= df_normal.index[-1]:
        nina4_data = df_normal.loc[nina4_start:nina4_end, "nina4"]
    else:
        nina4_data = pd.Series([np.nan] * len(cgr_data), index=cgr_data.index)

    if tmp_start >= df_normal.index[0] and tmp_end <= df_normal.index[-1]:
        tmp_data = df_normal.loc[tmp_start:tmp_end, "tmp_cru"]
    else:
        tmp_data = pd.Series([np.nan] * len(cgr_data), index=cgr_data.index)

    if tws_start >= df_normal.index[0] and tws_end <= df_normal.index[-1]:
        tws_data = df_normal.loc[tws_start:tws_end, "tws_grace"]
    else:
        tws_data = pd.Series([np.nan] * len(cgr_data), index=cgr_data.index)

    if pre_start >= df_normal.index[0] and pre_end <= df_normal.index[-1]:
        pre_data = df_normal.loc[pre_start:pre_end, "pre_gpcc"]
    else:
        pre_data = pd.Series([np.nan] * len(cgr_data), index=cgr_data.index)

    # Create a record for each time point
    for j, time_point in enumerate(cgr_data.index):
        record = {
            "time": time_point,
            "group_id": group_id,
            "CGR": cgr_data.iloc[j],
            "ENSO": enso_data.iloc[j],
            "NINA3": nina3_data.iloc[j],
            "NINA4": nina4_data.iloc[j],
            "TMP": tmp_data.iloc[j],
            "TWS": tws_data.iloc[j],
            "PRE": pre_data.iloc[j]
        }
        aggregated_data.append(record)

# Convert to DataFrame
recovery_data_df = pd.DataFrame(aggregated_data)

print(f"Aggregated data shape: {recovery_data_df.shape}")
print(f"Included recovery groups: {sorted(recovery_data_df['group_id'].unique())}")

recovery_data_df


# %%
# Display the time range for each recovery group
print("\n=== Recovery Groups Time Ranges ===")
for group_id in recovery_data_df["group_id"].unique():
    group = recovery_data_df[recovery_data_df["group_id"] == group_id]
    print(
        f"Group {group_id}: start = {group['time'].min()}, end = {group['time'].max()}"
    )


# %%
# Create arrays for start, middle, and end years
start_years = []
middle_years = []
end_years = []

for group_id in recovery_data_df["group_id"].unique():
    group = recovery_data_df[recovery_data_df["group_id"] == group_id]
    start_years.append(group['time'].min().year)
    middle_years.append(group['time'].median().year)
    end_years.append(group['time'].max().year)

start_years = np.array(start_years)
middle_years = np.array(middle_years)
end_years = np.array(end_years)

print(f"\nStart years: {start_years}")
print(f"Middle years: {middle_years}")
print(f"End years: {end_years}")


# %%
recovery_data_df


# %%

# %%
# --- 1. general lag ---
GENERAL_LAGS = {"ENSO": -4, "TMP": 0, "TWS": 0, "PRE": -5}
aggregated_data_genlag = []
for i, group in enumerate(recovery_groups):
    group_id = i + 1
    start_date, end_date = group.index[0], group.index[-1]
    cgr_data = group["CGR"]
    # ENSO
    enso_start = start_date - pd.DateOffset(months=-GENERAL_LAGS["ENSO"])
    enso_end = end_date - pd.DateOffset(months=-GENERAL_LAGS["ENSO"])
    enso_data = df_normal.loc[enso_start:enso_end, "nina34"].reindex(cgr_data.index)
    # TMP
    tmp_start = start_date - pd.DateOffset(months=-GENERAL_LAGS["TMP"])
    tmp_end = end_date - pd.DateOffset(months=-GENERAL_LAGS["TMP"])
    tmp_data = df_normal.loc[tmp_start:tmp_end, "tmp_cru"].reindex(cgr_data.index)
    # TWS
    tws_start = start_date - pd.DateOffset(months=-GENERAL_LAGS["TWS"])
    tws_end = end_date - pd.DateOffset(months=-GENERAL_LAGS["TWS"])
    tws_data = df_normal.loc[tws_start:tws_end, "tws_grace"].reindex(cgr_data.index)
    # PRE
    pre_start = start_date - pd.DateOffset(months=-GENERAL_LAGS["PRE"])
    pre_end = end_date - pd.DateOffset(months=-GENERAL_LAGS["PRE"])
    pre_data = df_normal.loc[pre_start:pre_end, "pre_gpcc"].reindex(cgr_data.index)
    # NINA3/NINA4 (same lag as ENSO)
    nina3_data = df_normal.loc[start_date:end_date, "nina3"].reindex(cgr_data.index)
    nina4_data = df_normal.loc[start_date:end_date, "nina4"].reindex(cgr_data.index)
    for j, time_point in enumerate(cgr_data.index):
        record = {
            "time": time_point,
            "group_id": group_id,
            "CGR": cgr_data.iloc[j],
            "ENSO": enso_data.iloc[j],
            "NINA3": nina3_data.iloc[j],
            "NINA4": nina4_data.iloc[j],
            "TMP": tmp_data.iloc[j],
            "TWS": tws_data.iloc[j],
            "PRE": pre_data.iloc[j]
        }
        aggregated_data_genlag.append(record)
test_recovery_df_genlag = pd.DataFrame(aggregated_data_genlag)


# %%

# %%
recovery_optimal_correlation[
        recovery_optimal_correlation['group_id'] == 1
    ]


# %%
recovery_data_df[recovery_data_df['group_id'] == group_id]


# %%
recovery_data_df['group_id'].unique()


# %%
# =======================================
# Scatter Plot: CGR vs ENSO for Each Recovery Event
# =======================================

# Create figure
n_groups = len(sorted(recovery_data_df['group_id'].unique()))
ncols = 4
nrows = (n_groups + ncols - 1) // ncols  # ceil division
fig, axs = pplt.subplots(
    nrows=nrows,
    ncols=ncols,
    journal='nat2',
    share=4  # Independent axes
)

fig.delaxes(axs[-1]); fig.delaxes(axs[-2]);  # Remove extra subplots if any

# Iterate over each recovery event
for idx, group_id in enumerate(sorted(recovery_data_df['group_id'].unique())):

    ax = axs[idx]
    
    # ============================================================
    # Get data for this recovery event
    # ============================================================
    group_data = recovery_data_df[recovery_data_df['group_id'] == group_id]
    
    # Get optimal lag and correlation from recovery_optimal_correlation
    corr_data = recovery_optimal_correlation[
        recovery_optimal_correlation['group_id'] == group_id
    ]
    
    if len(corr_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        continue
    
    optimal_lag = corr_data['ENSO_lag'].values[0]
    correlation = corr_data['ENSO_corr'].values[0]
    p_value = corr_data['ENSO_sig'].values[0] 

    # Get start and end years for event name
    start_year = group_data['time'].min().year
    end_year = group_data['time'].max().year
    event_name = f"{start_year}/{str(end_year)[-2:]}"  # e.g., "1966/67"
    
    # ============================================================
    # Prepare X (ENSO) and Y (CGR) data with optimal lag
    # ============================================================
    # ENSO is shifted by optimal_lag relative to CGR
    # optimal_lag < 0 means ENSO leads CGR
    
    # Get aligned data
    cgr_values = group_data['CGR'].values
    enso_values = group_data['ENSO'].values  # Assuming this is already shifted
    
    # Remove NaN pairs
    mask = ~(np.isnan(cgr_values) | np.isnan(enso_values))
    cgr_clean = cgr_values[mask]
    enso_clean = enso_values[mask]
    
    # ============================================================
    # Plot scatter
    # ============================================================
    ax.scatter(
        enso_clean,
        cgr_clean,
        s=50,
        c='#B22222',
        alpha=0.5,
        edgecolor='white',
        linewidth=0.3,
        zorder=3
    )
    
    # Add regression line
    sns.regplot(
        x=enso_clean,
        y=cgr_clean, 
        scatter=False,
        ax=ax,
        line_kws={'color': '#B22222', 'linewidth': 2, 'zorder': 4}
    )   
    
    # ============================================================
    # Add reference lines
    # ============================================================
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
    
    # ============================================================
    # Add annotations
    # ============================================================

    # Lag info with physical meaning
    if optimal_lag == 0:
        lag_text = "No lag"
    else:
        lead_var = "ENSO" if optimal_lag < 0 else "CGR"
        lag_text = f"Lag = {abs(optimal_lag)} mon"

    ax.text(
        0.05, 0.90,
        lag_text,
        transform=ax.transAxes,
    )
    
    # Correlation and p-value (bottom-right)
    p_text = rf"$\it{{P}}$ = {p_value:.3f}"
    corr_text = rf"N = {len(cgr_clean)}" + "\n" + rf"R = {correlation:.3f}" + "\n" + p_text

    ax.text(
        0.05, 0.05,
        corr_text,
        color='#B22222',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'),
        zorder=10
    )
    
    # ============================================================
    # Format axes
    # ============================================================
    ax.format(
        xlabel='SSTA [Niño 3.4; °C]',
        ylabel='CGR [Gt C]',
        grid=False,
        xlim=(-3, 3),
        ylim=(-5, 5),
        title=f"{event_name}", titleweight='bold'
    )

# fig.savefig('03-res/figs/SP_correlation_between_CGR_and_ENSO_with_best_lags.png', dpi=600)


# %%

# %% [markdown]
# ## Changes in carbon and climate variables
#

# %% [markdown]
# ### 1. Temporal trends
#

# %%
# ===============================
# Recovery Period Trend Analysis
# ===============================

print("=== Recovery Period Trend Analysis ===")

# Configuration - map to recovery_data_df columns
TREND_VARS = {"CGR": "CGR", 
              "ENSO": "ENSO", 
              "NINA3": "NINA3", 
              "NINA4": "NINA4",
              "TMP": "TMP", 
              "TWS": "TWS"}

# Store trend results
trend_results = []

# Calculate trends for each recovery group
for group_id in recovery_data_df["group_id"].unique():

    # Filter data for this group
    group_data = recovery_data_df[recovery_data_df["group_id"] == group_id]

    row_data = {"group_id": group_id, "duration_months": len(group_data)}

    # Calculate trend for each variable
    for var_name, var_col in TREND_VARS.items():

        # Get time series data
        Y_data = group_data[var_col].values
        X_data = np.arange(len(Y_data))  # Time index

        # Calculate linear trend using regression
        result = calculate_regression(X_data, Y_data, min_samples=10)

        # Store results
        row_data[f"{var_name}_trend"] = result["coef"]  # Change per month
        row_data[f"{var_name}_pvalue"] = result["pvalue"]
        row_data[f"{var_name}_r2"] = result["r2"]
        row_data[f"{var_name}_ci_lower"] = result["ci_lower"]
        row_data[f"{var_name}_ci_upper"] = result["ci_upper"]

        # Calculate total change over the period
        duration_months = len(group_data)
        row_data[f"{var_name}_total_change"] = result["coef"] * duration_months
        row_data[f"{var_name}_total_ci_lower"] = result["ci_lower"] * duration_months
        row_data[f"{var_name}_total_ci_upper"] = result["ci_upper"] * duration_months

    trend_results.append(row_data)

# Convert to DataFrame
trend_results = pd.DataFrame(trend_results)

trend_results 


# %%
# =======================================
# Plot Trends During Recovery Periods
# =======================================

fig, axes = pplt.subplots(
    ncols=4, nrows=1, refaspect=0.4, sharey=4, sharex=False, journal="nat2"
)

# Variable configuration
TREND_VARS = ["CGR", "ENSO", "TWS", "TMP"]

VAR_COLORS = {
    "CGR": "#000000",  # Dark Gray
    "ENSO": "#B22222",  # Fire Brick
    "TMP": "#DAA520",  # Goldenrod
    "TWS": "#4d9221",  # Steel Blue
}

# Panel X-axis configuration (adjust xlim/xlocator for per-month scale)
X_CONFIGS = {
    "CGR": {"xlim": (-10, 0), "xlocator": 2, "xlabel": "CGR change [Gt C]"},
    "ENSO": {"xlim": (-5, 0), "xlocator": 1, "xlabel": "SSTA change [°C]"},
    "TWS": {"xlim": (0, 50), "xlocator": 10, "xlabel": "TWS change [mm]"},
    "TMP": {"xlim": (-1.0, 0), "xlocator": 0.2, "xlabel": "TMP change [°C]"},
}

groups = np.arange(1, len(trend_results) + 1)
circled_numbers = [str(i) for i in groups]

for i, (ax, var) in enumerate(zip(axes, TREND_VARS)):
    # Extract per-month trend data
    trends = trend_results[f"{var}_total_change"].values

    # Get confidence intervals if available
    ci_lower_col = f"{var}_total_ci_lower"
    ci_upper_col = f"{var}_total_ci_upper"
    pvalue_col = f"{var}_pvalue"

    ci_lower = trend_results[ci_lower_col].values
    ci_upper = trend_results[ci_upper_col].values
    xerr_lower = trends - ci_lower
    xerr_upper = ci_upper - trends
    pval = trend_results[pvalue_col].values

    # Plot horizontal bars with significance-based styling
    for j, group in enumerate(groups):
        if pval[j] < 0.05:
            ax.barh(
                group,
                trends[j],
                color=VAR_COLORS[var],
                alpha=0.75,
                edgecolor=VAR_COLORS[var],
                linewidth=0,
                width=1.25,
            )
        else:
            ax.barh(
                group,
                trends[j],
                color="white",
                alpha=0.3,
                edgecolor=None,
                linewidth=1,
                width=1.25,
            )

    # plot the error bars for confidence intervals
    ax.errorbar(
        trends,
        groups,
        xerr=[xerr_lower, xerr_upper],
        fmt="o",
        color=VAR_COLORS[var],
        capsize=5,
        capthick=0,
        alpha=1,
        markersize=5,
        markerfacecolor="white",
        linestyle="none",
        linewidth=2,
        zorder=10,

    )

    # Add vertical reference line at zero
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0)

    # Format axes
    ax.format(
        ylim=(0.25, 12.75),
        ylocator=1,
        yticklabels=[] if i != 0 else None,
        ytickminor=False,
        xlim=X_CONFIGS[var]["xlim"],
        xlocator=X_CONFIGS[var]["xlocator"],
        grid=False,
        xlabel=X_CONFIGS[var]["xlabel"],
        xtickminor=False,
    )

    # Add circled numbers (only for first panel)
    if i == 0:
        for group_num, label in zip(groups, circled_numbers):
            fontsize = 8 if len(label) == 1 else 7.3
            pad = 0.15 if len(label) == 1 else 0.13
            ax.text(
                X_CONFIGS["CGR"]["xlim"][0] - 1,
                group_num,
                label,
                ha="center",
                va="center",
                color="k",
                fontsize=fontsize,
                zorder=100,
                bbox=dict(
                    facecolor="white",
                    edgecolor="k",
                    linewidth=0.5,
                    boxstyle=f"circle,pad={pad}",
                ),
            )

fig.format(leftlabels=["Carbon recovery episodes"])

# # Export figure
# fig.savefig("03-res/figs/temporal_trend_of_climate_during_recovery.png", dpi=600)


# %%

# %% [markdown]
# ### 2. Spatial trends
#

# %%
# Time range
start_date = pd.to_datetime("1959-03-01")
end_date   = pd.to_datetime("2024-12-31")

# ----- TMP
ds_TMP_cru = xr.open_dataset(os.path.join(proj_path, "01-data", "tmp_cru_detrend.nc"))
ds_TMP_cru = ds_TMP_cru["tmp"]
ds_TMP_cru = ds_TMP_cru.sel(time=slice(start_date, end_date))

# ----- SST
ds_ENSO_era = xr.open_dataset(os.path.join(proj_path, "01-data", "sst_era5_detrend.nc"))
ds_ENSO_era = ds_ENSO_era["sst"]
ds_ENSO_era = ds_ENSO_era.sel(time=slice(start_date, end_date))

# ----- TWS
ds_TWS_grace = xr.open_dataset(
    os.path.join(proj_path, "01-data", "tws_grace_detrend.nc")
)
ds_TWS_grace = ds_TWS_grace["rec_ensemble_mean"]
ds_TWS_grace = ds_TWS_grace.sel(time=slice(start_date, end_date))


# %%
variables = {"ENSO": ds_ENSO_era, "TWS": ds_TWS_grace, "TMP": ds_TMP_cru}

spatial_trend_results = {"ENSO": [], "TWS": [], "TMP": []}
spatial_change_results = {"ENSO": [], "TWS": [], "TMP": []} 

for i, group in enumerate(recovery_groups):
    group_id = i + 1
    start_date = group.index[0]
    end_date = group.index[-1]
    
    # 计算持续时间（月数）
    duration = len(group)  # 或者 (end_date - start_date).days / 30.44

    for var_name, ds_var in variables.items():
        group_data = recovery_optimal_correlation[
            recovery_optimal_correlation["group_id"] == group_id
        ]

        if len(group_data) > 0:
            lag_val = group_data[f"{var_name}_lag"].iloc[0]
            lag = int(lag_val) if not pd.isna(lag_val) else 0
            X_start = start_date - pd.DateOffset(months=-lag)
            X_end = end_date - pd.DateOffset(months=-lag)
            ds_group = ds_var.sel(time=slice(X_start, X_end))

            if len(ds_group.time) > 0:
                trend_result = linear_trend(ds_group, dim="time")
                change_result = trend_result * duration  # 总变化 = trend × 月数
            else:
                trend_result = xr.full_like(ds_var.isel(time=0), fill_value=np.nan)
                change_result = xr.full_like(ds_var.isel(time=0), fill_value=np.nan)

            spatial_trend_results[var_name].append(trend_result)
            spatial_change_results[var_name].append(change_result)  # 新增
        
        else:
            nan_result = xr.full_like(ds_var.isel(time=0), fill_value=np.nan)
            spatial_trend_results[var_name].append(nan_result)
            spatial_change_results[var_name].append(nan_result)  # 新增


# %%
# Get event names from recovery_data_df
event_names = []
for group_id in sorted(recovery_data_df['group_id'].unique()):
    group_data = recovery_data_df[recovery_data_df['group_id'] == group_id]
    start_year = group_data['time'].min().year
    end_year = group_data['time'].max().year
    event_name = f"{start_year}/{str(end_year)[-2:]}"  # e.g., "1966/67"
    event_names.append(event_name)

print(f"Event names: {event_names}")  # 确认顺序


# %%
spatial_change_results["ENSO"][0]


# %%
num_events = len(spatial_trend_results["TWS"])
num_vars = len(spatial_trend_results)

# Create the figure
fig, axs = pplt.subplots(
    ncols=num_vars, nrows=num_events, proj="cyl", share=3, refaspect=5, journal="nat2"
)

# Format all subplots
axs.format(
    # elements
    coast=True,
    coastcolor="k",
    coastlinewidth=1,
    coastzorder=10,
    # grid lines
    gridminor=False,
    grid=False,
    latlines=15,
    lonlines=20,
    labels=False,
    lonlim=(-180, 180),
    latlim=(-24, 24),
    reso="lo",
)

# Variable configuration - different colormaps
var_config = {
    "TWS": {
        "label": "TWS change",
        "cmap": "BrBG",
        "unit": "[mm]",
        "vmin": -100,
        "locator": 100,
    },
    "ENSO": {
        "label": "SSTA change",
        "cmap": "RdBu_r",
        "unit": "[°C]",
        "vmin": -1.0,
        "locator": 1.0,
    },
    "TMP": {
        "label": "TMP change",
        "cmap": "ColdHot",
        "unit": "[°C]",
        "vmin": -1,
        "locator": 1,
    },
}

var_names = ["ENSO", "TWS", "TMP"]

# Plot the data
for row in range(num_events):
    for col in range(num_vars):

        ax = axs[row, col]
        var_name = var_names[col]
        config = var_config[var_name]

        trend_data = spatial_change_results[var_name][row]

        if trend_data is not None and isinstance(trend_data, xr.Dataset) and "slope" in trend_data:
            # Create latitude and longitude grid
            nx = trend_data["lon"]
            ny = trend_data["lat"]
            nx, ny = np.meshgrid(nx, ny)

            # Plot using different colormaps
            im = ax.pcolormesh(
                nx,
                ny,
                trend_data["slope"],
                cmap=config["cmap"],
                vmin=config["vmin"],
                vmax=-config["vmin"],
                extend="both",
                discrete=False,
                zorder=1,
            )

            if row == num_events - 1:
                ax.colorbar(
                    im,
                    loc="b",
                    label=config["label"] + ' ' + config["unit"],
                    width=0.10,
                    shrink=0.75,
                    locator=config["locator"],
                )

        # Add row labels (event names) - only show in the first column
        if col == 0:
            label = event_names[row]  # Use event name instead of number
            ax.text(
                -0.03,  # 稍微左移一点，因为文字更长
                0.5,
                label,
                transform=ax.transAxes,
                rotation=0,  # 水平显示（不旋转）
                ha="right",  # 右对齐
                va="center",
                color="k",
                fontweight="bold",
                bbox=dict(facecolor="none", edgecolor="0", boxstyle="round,pad=0.3", alpha=0),
            )

# # Export figure
# fig.savefig("03-res/figs/SP_spatial_climate_changes_during_recovery.png", dpi=600)


# %%

# %% [markdown]
# ## Sensitivity during individual recovery groups
#

# %% [markdown]
# ### 1. Sensitivity changes
#

# %%
recovery_data_df


# %%
# ==================================
# Chain Decomposition
# ==================================

chain_regression_results = []

for group_id in recovery_data_df["group_id"].unique():
    group_data = recovery_data_df[recovery_data_df["group_id"] == group_id]
    row_data = {"group_id": group_id}

    regressions = {
        "CGR_ENSO": calculate_regression(group_data["ENSO"].values, group_data["CGR"].values),
        "CGR_NINA3": calculate_regression(group_data["NINA3"].values, group_data["CGR"].values),
        "CGR_NINA4": calculate_regression(group_data["NINA4"].values, group_data["CGR"].values),    
        "CGR_TWS": calculate_regression(group_data["TWS"].values, group_data["CGR"].values),
        "TWS_ENSO": calculate_regression(group_data["ENSO"].values, group_data["TWS"].values),
        "CGR_PRE": calculate_regression(group_data["PRE"].values, group_data["CGR"].values),
        "PRE_ENSO": calculate_regression(group_data["ENSO"].values, group_data["PRE"].values),
        "CGR_TMP": calculate_regression(group_data["TMP"].values, group_data["CGR"].values),
        "TMP_ENSO": calculate_regression(group_data["ENSO"].values, group_data["TMP"].values),
    }

    for reg_name, result in regressions.items():
        row_data[f"{reg_name}_coefficient"] = result["coef"]
        row_data[f"{reg_name}_pvalue"] = result["pvalue"]
        row_data[f"{reg_name}_ci_lower"] = result["ci_lower"]
        row_data[f"{reg_name}_ci_upper"] = result["ci_upper"]

    chain_regression_results.append(row_data)

chain_regression_df = pd.DataFrame(chain_regression_results)

# Estimated impacts
chain_regression_df["indirect_TWS"] = chain_regression_df["CGR_TWS_coefficient"] * chain_regression_df["TWS_ENSO_coefficient"]
chain_regression_df["indirect_TMP"] = chain_regression_df["CGR_TMP_coefficient"] * chain_regression_df["TMP_ENSO_coefficient"]
chain_regression_df["indirect_PRE"] = chain_regression_df["CGR_PRE_coefficient"] * chain_regression_df["PRE_ENSO_coefficient"]

chain_regression_df


# %%
# ================================
# Chain Rule Sensitivity Bar Plot
# ================================

VAR_CONFIG = {
    "CGR_ENSO": {"color": "#B22222", "ylim": (0, 7), "title": r"$\gamma_{SSTA}^{CGR}$", "unit": "Gt C per °C", "xlocator": 1, "show_labels": True},
}

fig, axes = pplt.subplots(
    ncols=1, nrows=1, refaspect=3, sharey=4, sharex=1, journal="nat2"
)

groups = np.arange(1, len(chain_regression_df) + 1)
circled_numbers = [str(i) for i in groups]

for ax, (var, config) in zip(axes, VAR_CONFIG.items()):
    coeffs = chain_regression_df[f"{var}_coefficient"].values
    ci_lower = chain_regression_df[f"{var}_ci_lower"].values
    ci_upper = chain_regression_df[f"{var}_ci_upper"].values
    pvalue = chain_regression_df[f"{var}_pvalue"].values
    significant = pvalue < 0.05

    yerr_lower = coeffs - ci_lower
    yerr_upper = ci_upper - coeffs

    for i, group in enumerate(groups):
        if significant[i]:
            ax.bar(
                group,
                coeffs[i],
                color=config["color"],
                alpha=0.75,
                edgecolor=None,
                linewidth=1,
                width=0.75,
            )
        else:
            ax.bar(
                group,
                coeffs[i],
                color="white",
                alpha=0.75,
                edgecolor=None,
                linewidth=1,
                width=0.75,
            )
   
    ax.errorbar(
        groups,
        coeffs,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color=config["color"],
        capsize=5,
        capthick=0,
        markersize=5,
        markerfacecolor="white",
        linestyle="none",
        linewidth=2,
        zorder=10,
    )

    valid_mask = ~np.isnan(np.array(coeffs))
    lng_res = pg.linear_regression(np.array(start_years)[valid_mask], np.array(coeffs)[valid_mask])
    r_value = lng_res["coef"].iloc[1]
    p_value = lng_res["pval"].iloc[1]
    ax.text(
        0.02,
        0.95,
        f"Slope = {r_value:.3f}, $\\mathit{{P}}$ = {p_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color=config["color"],
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

    ax.format(
        ylim=config["ylim"],
        xlim=(groups.min() - 0.75, groups.max() + 0.75),
        ylocator=1,
        xlocator=config["xlocator"],
        ylabel="",
        xlabel="",
        yticklabels=[] if not config["show_labels"] else None,
        xticklabels=[],
        xgrid=False,
        ygrid=False,
        xtickminor=False,
        ytickminor=False,
    )

    if config["show_labels"]:
        for group_num, label in zip(groups, circled_numbers):
            fontsize = 8 if len(label) == 1 else 7.5
            pad = 0.15 if len(label) == 1 else 0.1
            ax.text(
                group_num,
                config["ylim"][0] - 0.5,
                label,
                ha="center",
                va="center",
                color="k",
                fontsize=fontsize,
                zorder=100,
                bbox=dict(
                    facecolor="white",
                    edgecolor="k",
                    linewidth=0.5,
                    boxstyle=f"circle,pad={pad}",
                ),
            )
            # 新增：时间信息（在圆圈下方）
            start_year = start_years[group_num-1]
            end_year = end_years[group_num-1]
            
            ax.text(
                group_num,
                config["ylim"][0] - 1.5,  # 在圆圈下方
                f"{start_year}/{str(end_year)[-2:]}",
                ha="center", va="center", color="k",
                fontsize=9,
                rotation=90,
                zorder=99
            )

    ax.format(ylabel=config["title"] + f" [{config['unit']}]")

# # Export figure
# fig.savefig("03-res/figs/carbon_recovery_sensitivity_bar_plot_nino34.png", dpi=600)


# %%
### Bootstrap for trend significance of sensitivity changes

import numpy as np
from scipy.stats import linregress

# 数据 (drop NaN from event 14 missing data)
valid_mask = ~np.isnan(np.array(coeffs))
year = np.array(start_years)[valid_mask]
gamma = np.array(coeffs)[valid_mask]

# 原始回归
slope_obs, _, _, p_obs, _ = linregress(year, gamma)
print(f"Observed: slope={slope_obs:.4f}, P={p_obs:.3f}")

# Bootstrap
n_boot = 1000
slopes = []
p_values = []

for i in range(n_boot):
    # 有放回抽样
    indices = np.random.choice(len(year), size=len(year), replace=True)
    year_boot = year[indices]
    gamma_boot = gamma[indices]
    
    # 回归
    slope, _, _, p, _ = linregress(year_boot, gamma_boot)
    slopes.append(slope)
    p_values.append(p)

slopes = np.array(slopes)
p_values = np.array(p_values)

# 统计结果
ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])

# 正确的显著性检验
is_significant = not (ci_low <= 0 <= ci_high)
p_bootstrap = 2 * min(np.mean(slopes >= 0), np.mean(slopes <= 0))

print(f"Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Significant: {is_significant}")
print(f"Bootstrap P-value: {p_bootstrap:.3f}")


# %%
from scipy.stats import gaussian_kde

# Bootstrap for trend significance of sensitivity changes
fig, ax = pplt.subplots(refaspect=2, journal="nat2")

# Create histogram (normalized to density)
ax.hist(
    slopes, 
    bins=50, 
    alpha=0.5,  # 降低透明度，让曲线更明显
    edgecolor='none', 
    color='#B22222',
    density=True,  # ✅ 归一化为密度，与 KDE 曲线匹配
)

# ✅ Add smooth KDE curve
kde = gaussian_kde(slopes)
x_range = np.linspace(slopes.min(), slopes.max(), 500)
density = kde(x_range)

ax.plot(
    x_range, 
    density,
    color='#B22222',
    linewidth=2,
    zorder=10,
)

# Add vertical lines
#ax.axvline(slope_obs, color='#B22222', linestyle='--', linewidth=2, label='Observed')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.axvline(ci_low, color='gray', linestyle=':', label='95% CI')
ax.axvline(ci_high, color='gray', linestyle=':')

# Format the plot
ax.format(
    xlabel='Slope of $\gamma_{SSTA}^{CGR}$ ($Gt\\ C\\ °C^{-1}$)',
    ylabel='Density',  # ✅ 改为 Density
    xlim=(-0.10, 0.10),
    ylim=(0,50),
    grid=False
)

# Add legend
ax.legend(loc='ur', frameon=False, ncols=1)

# # Export figure
# fig.savefig("03-res/figs/SP_bootstrap_resample_trend.png", dpi=600)


# %%
from scipy.stats import mannwhitneyu

############################################################
# First 5 groups and last 5 groups
############################################################
group1 = chain_regression_df["CGR_ENSO_coefficient"].iloc[:5]
group2 = chain_regression_df["CGR_ENSO_coefficient"].iloc[-5:]

u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

print(f"--- 5 vs 5: U statistic: {u_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("There is a significant difference between the first 5 and last 5 groups.")
else:
    print("No significant difference between the first 5 and last 5 groups.")

############################################################
# First 3 groups and last 3 groups
############################################################
group1 = chain_regression_df["CGR_ENSO_coefficient"].iloc[:3]
group2 = chain_regression_df["CGR_ENSO_coefficient"].iloc[-3:]

u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

print(f"--- 3 vs 3: U statistic: {u_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("There is a significant difference between the first 3 and last 3 groups.")
else:
    print("No significant difference between the first 3 and last 3 groups.")



# %%
# 提取数据 - 前5 vs 后5
group1_5 = chain_regression_df["CGR_ENSO_coefficient"].iloc[:5]
group2_5 = chain_regression_df["CGR_ENSO_coefficient"].iloc[-5:]
u_stat_5, p_value_5 = mannwhitneyu(group1_5, group2_5, alternative="two-sided")

# 提取数据 - 前3 vs 后3
group1_3 = chain_regression_df["CGR_ENSO_coefficient"].iloc[:3]
group2_3 = chain_regression_df["CGR_ENSO_coefficient"].iloc[-3:]
u_stat_3, p_value_3 = mannwhitneyu(group1_3, group2_3, alternative="two-sided")

# 计算均值和标准误
means_5 = [group1_5.mean(), group2_5.mean()]
sems_5 = [group1_5.sem(), group2_5.sem()]
means_3 = [group1_3.mean(), group2_3.mean()]
sems_3 = [group1_3.sem(), group2_3.sem()]

fig, axs = pplt.subplots(journal="nat1", refaspect=1, ncols=2)

# 设置柱子位置
width = 0.5
pos = [0, 1]

# 画柱子 - 前5 vs 后5
axs[0].bar(
    pos,
    means_5,
    width=width,
    color=["#B22222", "#F1F2E5"],
    alpha=0.7,
    edgecolor="black",
    linewidth=0.8,
)
axs[0].errorbar(
    pos,
    means_5,
    yerr=sems_5,
    fmt="none",
    color="black",
    capsize=5,
    linewidth=1.5,
    capthick=0,
)

# 画柱子 - 前3 vs 后3
axs[1].bar(
    pos,
    means_3,
    width=width,
    color=["#B22222", "#F1F2E5"],
    alpha=0.7,
    edgecolor="black",
    linewidth=0.8,
)
axs[1].errorbar(
    pos,
    means_3,
    yerr=sems_3,
    fmt="none",
    color="black",
    capsize=5,
    linewidth=1.5,
    capthick=0,
)

# 零线
axs[0].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
axs[1].axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)

# 显著性标注
axs[0].text(1, 2.9, f"$\\mathit{{P}}$={p_value_5:.3f}", ha="center", fontsize=9)
axs[1].text(
    1,
    2.9,
    f"$\\mathit{{P}}$={p_value_3:.3f}",
    ha="center",
    fontsize=9,
)

# 格式化
axs.format(
    xlabel="",
    xlim=(-0.5, 1.5),
    ylabel=r"$\gamma_{SSTA}^{CGR}$ [Gt C per °C]",
    xticks=pos,
    grid=False,
)

axs[0].format(
    xticklabels=["First 5 \n episodes", "Last 5 \n episodes"],
)
axs[1].format(
    xticklabels=["First 3 \n episodes", "Last 3 \n episodes"],
)


# %%
############################################################
# First max 5 groups and min 5 groups 
############################################################
group1 = chain_regression_df["CGR_ENSO_coefficient"].nlargest(5)
group2 = chain_regression_df["CGR_ENSO_coefficient"].nsmallest(5)

u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

print(f"--- Max vs Min: U statistic: {u_stat:.3f}, p-value: {p_value:.3f}")
if p_value < 0.05:
    print("There is a significant difference between the max 5 and min 5 groups.")
else:
    print("No significant difference between the max 5 and min 5 groups.")


# %%
# ================================"
# Sensitivity Bar Plot （Nina 3 & Nina 4）
# ================================"

VAR_CONFIG = {
    "CGR_NINA3": {
        "color": "#B22222",
        "ylim": (0, 9),
        "title": r"$\gamma_{SSTA}^{CGR}$",
        "unit": "Niño 3; Gt C per °C",
        "xlocator": 1,
        "show_labels": True,
    },
    "CGR_NINA4": {
        "color": "#B22222",
        "ylim": (0, 9),
        "title": r"$\gamma_{SSTA}^{CGR}$",
        "unit": "Niño 4; Gt C per °C",
        "xlocator": 1,
        "show_labels": True,
    },
}

fig, axes = pplt.subplots(
    ncols=1, nrows=2, refaspect=3, sharey=0, sharex=1, journal="nat2"
)

groups = np.arange(1, len(chain_regression_df) + 1)
circled_numbers = [str(i) for i in groups]

for ax, (var, config) in zip(axes, VAR_CONFIG.items()):
    coeffs = chain_regression_df[f"{var}_coefficient"].values
    ci_lower = chain_regression_df[f"{var}_ci_lower"].values
    ci_upper = chain_regression_df[f"{var}_ci_upper"].values
    pvalue = chain_regression_df[f"{var}_pvalue"].values
    significant = pvalue < 0.05

    yerr_lower = coeffs - ci_lower
    yerr_upper = ci_upper - coeffs

    for i, group in enumerate(groups):
        if significant[i]:
            ax.bar(
                group,
                coeffs[i],
                color=config["color"],
                alpha=0.75,
                edgecolor=None,
                linewidth=1,
                width=0.75,
            )
        else:
            ax.bar(
                group,
                coeffs[i],
                color="white",
                alpha=0.75,
                edgecolor=None,
                linewidth=1,
                width=0.75,
            )

    ax.errorbar(
        groups,
        coeffs,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color=config["color"],
        capsize=5,
        capthick=0,
        markersize=5,
        markerfacecolor="white",
        linestyle="none",
        linewidth=2,
        zorder=10,
    )

    valid_mask = ~np.isnan(np.array(coeffs))
    lng_res = pg.linear_regression(np.array(start_years)[valid_mask], np.array(coeffs)[valid_mask])
    r_value = lng_res["coef"].iloc[1]
    p_value = lng_res["pval"].iloc[1]
    ax.text(
        0.02,
        0.95,
        f"Slope = {r_value:.3f}, $\\mathit{{P}}$ = {p_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color=config["color"],
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

    ax.format(
        ylim=config["ylim"],
        xlim=(groups.min() - 0.75, groups.max() + 0.75),
        ylocator=1,
        xlocator=config["xlocator"],
        ylabel=config["title"] + f" [{config['unit']}]",
        xlabel="",
        yticklabels=[] if not config["show_labels"] else None,
        xticklabels=[],
        xgrid=False,
        ygrid=False,
        xtickminor=False,
        ytickminor=False,
    )

    if config["show_labels"]:
        for group_num, label in zip(groups, circled_numbers):
            fontsize = 8 if len(label) == 1 else 7.5
            pad = 0.2 if len(label) == 1 else 0.1
            ax.text(
                group_num,
                config["ylim"][0] - 0.5,
                label,
                ha="center",
                va="center",
                color="k",
                fontsize=fontsize,
                zorder=100,
                bbox=dict(
                    facecolor="white",
                    edgecolor="k",
                    linewidth=0.5,
                    boxstyle=f"circle,pad={pad}",
                ),
            )
            # 新增：时间信息（在圆圈下方）
            start_year = start_years[group_num-1]  # 您已经有这个数组
            end_year = end_years[group_num-1]      # 您已经有这个数组
            
            ax.text(
                group_num,
                config["ylim"][0] - 1.7,  # 在圆圈下方
                f"{start_year}/{str(end_year)[-2:]}",
                ha="center", va="center", color="k", 
                fontsize=8,  # 更小的字体
                rotation=90,  # 可选：倾斜显示节省空间
                zorder=99
            )
# # Export figure
# fig.savefig("03-res/figs/SP_carbon_recovery_sensitivity_bar_plot_nino3_and_4.png", dpi=600)


# %% [markdown]
# #### Test for Genlag dataframe
#

# %%
# ==================================
# Chain Decomposition for General Lag Data
# ==================================
chain_regression_results_genlag = []

for group_id in test_recovery_df_genlag["group_id"].unique():
    group_data = test_recovery_df_genlag[test_recovery_df_genlag["group_id"] == group_id]
    row_data = {"group_id": group_id}

    regressions = {
        "CGR_ENSO": calculate_regression(group_data["ENSO"].values, group_data["CGR"].values),
        "CGR_NINA3": calculate_regression(group_data["NINA3"].values, group_data["CGR"].values),
        "CGR_NINA4": calculate_regression(group_data["NINA4"].values, group_data["CGR"].values),
        "CGR_TWS": calculate_regression(group_data["TWS"].values, group_data["CGR"].values),
        "TWS_ENSO": calculate_regression(group_data["ENSO"].values, group_data["TWS"].values),
        "CGR_PRE": calculate_regression(group_data["PRE"].values, group_data["CGR"].values),
        "PRE_ENSO": calculate_regression(group_data["ENSO"].values, group_data["PRE"].values),
        "CGR_TMP": calculate_regression(group_data["TMP"].values, group_data["CGR"].values),
        "TMP_ENSO": calculate_regression(group_data["ENSO"].values, group_data["TMP"].values),
    }

    for reg_name, result in regressions.items():
        row_data[f"{reg_name}_coefficient"] = result["coef"]
        row_data[f"{reg_name}_pvalue"] = result["pvalue"]
        row_data[f"{reg_name}_ci_lower"] = result["ci_lower"]
        row_data[f"{reg_name}_ci_upper"] = result["ci_upper"]

    chain_regression_results_genlag.append(row_data)

chain_regression_df_genlag = pd.DataFrame(chain_regression_results_genlag)

# Estimated impacts
chain_regression_df_genlag["indirect_TWS"] = chain_regression_df_genlag["CGR_TWS_coefficient"] * chain_regression_df_genlag["TWS_ENSO_coefficient"]
chain_regression_df_genlag["indirect_TMP"] = chain_regression_df_genlag["CGR_TMP_coefficient"] * chain_regression_df_genlag["TMP_ENSO_coefficient"]
chain_regression_df_genlag["indirect_PRE"] = chain_regression_df_genlag["CGR_PRE_coefficient"] * chain_regression_df_genlag["PRE_ENSO_coefficient"]

chain_regression_df_genlag


# %%
# ================================
# Chain Rule Sensitivity Bar Plot
# ================================

VAR_CONFIG = {
    "CGR_ENSO": {"color": "#B22222", "ylim": (0, 7), "title": r"$\gamma_{SSTA}^{CGR}$", "unit": "Gt C per °C", "xlocator": 1, "show_labels": True},
}

fig, axes = pplt.subplots(
    ncols=1, nrows=1, refaspect=3, sharey=4, sharex=1, journal="nat2"
)

groups = np.arange(1, len(chain_regression_df) + 1)
circled_numbers = [str(i) for i in groups]

for ax, (var, config) in zip(axes, VAR_CONFIG.items()):
    coeffs = chain_regression_df_genlag[f"{var}_coefficient"].values
    ci_lower = chain_regression_df_genlag[f"{var}_ci_lower"].values
    ci_upper = chain_regression_df_genlag[f"{var}_ci_upper"].values
    pvalue = chain_regression_df_genlag[f"{var}_pvalue"].values
    significant = pvalue < 0.05

    yerr_lower = coeffs - ci_lower
    yerr_upper = ci_upper - coeffs

    for i, group in enumerate(groups):
        if significant[i]:
            ax.bar(
                group,
                coeffs[i],
                color=config["color"],
                alpha=0.75,
                edgecolor=config["color"],
                linewidth=1,
                width=0.75,
            )
        else:
            ax.bar(
                group,
                coeffs[i],
                color="white",
                alpha=0.75,
                edgecolor=config["color"],
                linewidth=1,
                width=0.75,
            )
   
    ax.errorbar(
        groups,
        coeffs,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color=config["color"],
        capsize=5,
        capthick=0,
        markersize=5,
        markerfacecolor="white",
        linestyle="none",
        linewidth=2,
        zorder=10,
    )

    valid_mask = ~np.isnan(np.array(coeffs))
    lng_res = pg.linear_regression(np.array(groups)[valid_mask], np.array(coeffs)[valid_mask])
    r_value = lng_res["coef"].iloc[1]
    p_value = lng_res["pval"].iloc[1]
    ax.text(
        0.02,
        0.95,
        f"Slope = {r_value:.3f}, $\\mathit{{P}}$ = {p_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color=config["color"],
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

    ax.format(
        ylim=config["ylim"],
        xlim=(groups.min() - 0.75, groups.max() + 0.75),
        ylocator=1,
        xlocator=config["xlocator"],
        ylabel="",
        xlabel="",
        yticklabels=[] if not config["show_labels"] else None,
        xticklabels=[],
        xgrid=False,
        ygrid=False,
        xtickminor=False,
        ytickminor=False,
    )

    if config["show_labels"]:
        for group_num, label in zip(groups, circled_numbers):
            fontsize = 8 if len(label) == 1 else 7.5
            pad = 0.15 if len(label) == 1 else 0.1
            ax.text(
                group_num,
                config["ylim"][0] - 0.5,
                label,
                ha="center",
                va="center",
                color="k",
                fontsize=fontsize,
                zorder=100,
                bbox=dict(
                    facecolor="white",
                    edgecolor="k",
                    linewidth=0.5,
                    boxstyle=f"circle,pad={pad}",
                ),
            )

    ax.format(ylabel=config["title"] + f" [{config['unit']}]")


# %%

# %% [markdown]
# ### 2. Reference sensitivity
#

# %% [markdown]
# #### Type 1: Recovery groups
#

# %%
# =======================================
# Reference Sensitivities DataFrame
# =======================================

# Initialize storage
ref_regression_results = []

# Define relationships
relationships = [
    ("ENSO", "CGR"),
    ("TWS", "CGR"),
    ("TMP", "CGR"),
    ("ENSO", "TWS"),
    ("ENSO", "TMP"),
]

# Calculate each relationship
for x_var, y_var in relationships:
    X = recovery_data_df[x_var].values
    Y = recovery_data_df[y_var].values

    # Remove NaN
    valid = ~(np.isnan(X) | np.isnan(Y))
    X_clean, Y_clean = X[valid], Y[valid]

    if len(X_clean) >= 10:
        # OLS regression
        X_const = sm.add_constant(X_clean)
        model = sm.OLS(Y_clean, X_const).fit()

        if len(model.params) >= 2:
            conf_int = model.conf_int(alpha=0.05)
            ref_regression_results.append(
                {
                    "relationship": f"{y_var}_vs_{x_var}",
                    "coefficient": model.params[1],
                    "ci_lower": conf_int[1, 0],
                    "ci_upper": conf_int[1, 1],
                    "pvalue": model.pvalues[1],
                    "n_samples": len(X_clean),
                }
            )

# Create DataFrame
ref_regression_results = pd.DataFrame(ref_regression_results)

ref_regression_results


# %% [markdown]
# #### Type 2: All sampling points
#

# %%
# =======================================
# Sensitivity regression results for the full period (using optimal lags, consistent with recovery periods)
# =======================================

# Define optimal lags (consistent with recovery periods)
OPTIMAL_LAGS = {"ENSO": -4, "TMP": 0, "TWS": 0}  # ENSO leads CGR by 4 months

# Variable name to df_normal column mapping
VAR_COL_MAP = {"ENSO": "nina34", "TMP": "tmp_cru", "TWS": "tws_grace"}

relationships = [
    ("ENSO", "CGR"),
    ("TWS", "CGR"),
    ("TMP", "CGR"),
    ("ENSO", "TWS"),
    ("ENSO", "TMP"),
]

ref_regression_results = []

for x_var, y_var in relationships:
    # Lag setting
    lag = OPTIMAL_LAGS.get(x_var, 0)
    # Get X variable
    X_col = VAR_COL_MAP[x_var] if x_var in VAR_COL_MAP else x_var.lower()
    X = df_normal[X_col].shift(-lag).values
    # Get Y variable
    if y_var == "CGR":
        Y = df_normal["CGR"].values
    else:
        Y_col = VAR_COL_MAP[y_var] if y_var in VAR_COL_MAP else y_var.lower()
        Y = df_normal[Y_col].values

    valid = ~(np.isnan(X) | np.isnan(Y))
    X_clean, Y_clean = X[valid], Y[valid]

    if len(X_clean) >= 10:
        X_const = sm.add_constant(X_clean)
        model = sm.OLS(Y_clean, X_const).fit()
        if len(model.params) >= 2:
            conf_int = model.conf_int(alpha=0.05)
            ref_regression_results.append(
                {
                    "relationship": f"{y_var}_vs_{x_var}",
                    "coefficient": model.params[1],
                    "ci_lower": conf_int[1, 0],
                    "ci_upper": conf_int[1, 1],
                    "pvalue": model.pvalues[1],
                    "n_samples": len(X_clean),
                    "lag": lag,
                }
            )

ref_regression_results = pd.DataFrame(ref_regression_results)

ref_regression_results


# %% [markdown]
# # Physical Drivers of Stability
#

# %% [markdown]
# ## Chain decomposition validation
#

# %% [markdown]
# 1. ΔCGR/ΔSST = (ΔCGR/ΔTWS) × (ΔTWS/ΔSST)
# 2. ΔCGR/ΔSST = (ΔCGR/ΔTMP) × (ΔTMP/ΔSST)
#

# %% [markdown]
# ### 1. Validation
#

# %%
water_mask = chain_regression_df["CGR_TWS_pvalue"] < 0.10
heat_mask  = chain_regression_df["CGR_TMP_pvalue"] < 0.10 
enso_mask  = chain_regression_df['CGR_ENSO_pvalue'] < 0.10


# %%

xlabel_latex = r"$\gamma_{SSTA}^{CGR}$ [Gt C per °C]"
ylabel_water = r"$\gamma_{SSTA}^{TWS} \times \gamma_{TWS}^{CGR}$ [Gt C per °C]"
ylabel_heat  = r"$\gamma_{SSTA}^{TMP} \times \gamma_{TMP}^{CGR}$ [Gt C per °C]"
ylabel_pre   = r"$\gamma_{SSTA}^{PRE} \times \gamma_{PRE}^{CGR}$ [Gt C per °C]"

fig, axs = pplt.subplots(ncols=3, nrows=1, share=0, journal="nat2", refaspect=1)

# Plot data points
axs[0].scatter(
    chain_regression_df[water_mask]["CGR_ENSO_coefficient"],
    chain_regression_df[water_mask]["indirect_TWS"],
    markersize=100,
    c="w",
    markerfacecolor="white",
    edgecolor="#4682B4",
    edgewidth=1,
)
axs[1].scatter(
    chain_regression_df[heat_mask]["CGR_ENSO_coefficient"],
    chain_regression_df[heat_mask]["indirect_TMP"],
    markersize=100,
    c="w",
    markerfacecolor="white",
    edgecolor="#DAA520",
    edgewidth=1,
)
axs[2].scatter(
    chain_regression_df["CGR_ENSO_coefficient"],  # PRE不筛选mask
    chain_regression_df["indirect_PRE"],
    markersize=100,
    c="w",
    markerfacecolor="white",
    edgecolor="#2E8B57",  # 绿色
    edgewidth=1,
)

# Add number labels to each point
groups = np.arange(1, len(chain_regression_df) + 1)

# Water pathway (left panel)
water_indices = np.where(water_mask)[0]
for i, idx in enumerate(water_indices):
    group_num = groups[idx]
    x = chain_regression_df.loc[idx, "CGR_ENSO_coefficient"]
    y = chain_regression_df.loc[idx, "indirect_TWS"]
    axs[0].text(
        x, y, str(group_num),
        ha="center", va="center", color="#4682B4",
        fontweight="bold", fontsize=8, zorder=6,
    )

# Temperature pathway (middle panel)
temp_indices = np.where(heat_mask)[0]
for i, idx in enumerate(temp_indices):
    group_num = groups[idx]
    x = chain_regression_df.loc[idx, "CGR_ENSO_coefficient"]
    y = chain_regression_df.loc[idx, "indirect_TMP"]
    axs[1].text(
        x, y, str(group_num),
        ha="center", va="center", color="#DAA520",
        fontweight="bold", fontsize=8, zorder=6,
    )

# Precipitation pathway (right panel,所有组都标)
for idx in range(len(chain_regression_df)):
    group_num = groups[idx]
    x = chain_regression_df.loc[idx, "CGR_ENSO_coefficient"]
    y = chain_regression_df.loc[idx, "indirect_PRE"]
    axs[2].text(
        x, y, str(group_num),
        ha="center", va="center", color="#2E8B57",
        fontweight="bold", fontsize=8, zorder=6,
    )

# plot the correlation lines
res_water = pg.corr(
    chain_regression_df[water_mask]["CGR_ENSO_coefficient"], chain_regression_df[water_mask]["indirect_TWS"]
)
corr_water, pval_water = res_water["r"].values[0], res_water["p_val"].values[0]

res_temperature = pg.corr(
    chain_regression_df[heat_mask]["CGR_ENSO_coefficient"],
    chain_regression_df[heat_mask]["indirect_TMP"],
)
corr_temperature, pval_temperature = (
    res_temperature["r"].values[0],
    res_temperature["p_val"].values[0],
)

res_pre = pg.corr(
    chain_regression_df["CGR_ENSO_coefficient"], chain_regression_df["indirect_PRE"]
)
corr_pre, pval_pre = res_pre["r"].values[0], res_pre["p_val"].values[0]

axs[0].text(
    0.10, 4.75,
    f"R = {corr_water:.2f}, $\\mathit{{P}}$ = {pval_water:.3f}",
    ha="left", va="center", fontsize=10, color="#4682B4",
)
axs[1].text(
    0.10, 4.75,
    f"R = {corr_temperature:.2f}, $\\mathit{{P}}$ = {pval_temperature:.3f}",
    ha="left", va="center", fontsize=10, color="#DAA520",
)
axs[2].text(
    0.10, 4.75,
    f"R = {corr_pre:.2f}, $\\mathit{{P}}$ = {pval_pre:.3f}",
    ha="left", va="center", fontsize=10, color="#2E8B57",
)

# Add 1:1 reference line
for ax in axs:
    ax.plot([0, 5], [0, 5], color="k", linewidth=2, zorder=0)

for ax, xlabel, ylabel in zip(
    axs,
    [xlabel_latex] * 3,
    [ylabel_water, ylabel_heat, ylabel_pre]
):
    ax.format(
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(0, 5),
        ylim=(0, 5),
        grid=False,
        xlocator=1,
        ylocator=1,
    )

# Add panel titles
fig.format(toplabels=("Water", "Heat", "Water"))



# %% [markdown]
# ### 2. Results
#

# %%
ref_coefficients = {
    "CGR_ENSO": ref_regression_results.query("relationship == 'CGR_vs_ENSO'")["coefficient"].values[0],
    "CGR_TWS": ref_regression_results.query("relationship == 'CGR_vs_TWS'")["coefficient"].values[0],
    "TWS_ENSO": ref_regression_results.query("relationship == 'TWS_vs_ENSO'")["coefficient"].values[0],
    "CGR_TMP": ref_regression_results.query("relationship == 'CGR_vs_TMP'")["coefficient"].values[0],
    "TMP_ENSO": ref_regression_results.query("relationship == 'TMP_vs_ENSO'")["coefficient"].values[0],
}


# %%
# ==========================================
# Chain Rule Sensitivity Bar Plot (3x3 layout, LaTeX labels)
# ==========================================

# LaTeX formula labels for each panel
LATEX_LABELS = {
    "CGR_ENSO": r"$\gamma_{SSTA}^{CGR}$ [Gt C per °C]",
    "CGR_TWS": r"$\gamma_{TWS}^{CGR}$ [Gt C per °C]",
    "TWS_ENSO": r"$\gamma_{SSTA}^{TWS}$ [mm per °C]",
    "CGR_TMP": r"$\gamma_{TMP}^{CGR}$ [Gt C per °C]",
    "TMP_ENSO": r"$\gamma_{SSTA}^{TMP}$ [°C per °C]",
}

# Panel configuration: position, title, color, reference line, axis limits, tick locator
PANEL_CONFIG = {
    (1, 0): {
        "title": LATEX_LABELS["CGR_ENSO"],
        "color": "#B22222",
        "ref_line": ref_coefficients["CGR_ENSO"],
        "xlim": (0, 8),
        "xlocator": 2,
    },
    (0, 2): {
        "title": LATEX_LABELS["CGR_TWS"],
        "color": "#4682B4",
        "xlim": (0, -1),
        "xlocator": 0.2,
    },
    (0, 1): {
        "title": LATEX_LABELS["TWS_ENSO"],
        "color": "#87CEEB",
        "ref_line": ref_coefficients["TWS_ENSO"],
        "xlim": (0, -25),
        "xlocator": 5,
    },
    (2, 2): {
        "title": LATEX_LABELS["CGR_TMP"],
        "color": "#DAA520",
        "ref_line": ref_coefficients["CGR_TMP"],
        "xlim": (0, 27),
        "xlocator": 8,
    },
    (2, 1): {
        "title": LATEX_LABELS["TMP_ENSO"],
        "color": "#F0E68C",
        "ref_line": ref_coefficients["TMP_ENSO"],
        "xlim": (0, 0.4),
        "xlocator": 0.1,
    },
    # Other positions are left empty
    (1, 1): None,
    (1, 2): None,
    (0, 0): None,
    (2, 0): None,
}

# Mapping for data and mask for each panel
DATA_MAPPING = {
    (1, 0): "CGR_ENSO",
    (0, 2): "CGR_TWS",
    (0, 1): "TWS_ENSO",
    (2, 2): "CGR_TMP",
    (2, 1): "TMP_ENSO",
}
MASK_MAPPING = {
    (1, 0): None,
    (0, 2): water_mask,
    (0, 1): water_mask,
    (2, 2): heat_mask,
    (2, 1): heat_mask,
}

# Create 3x3 subplot grid
fig, axs = pplt.subplots(
    ncols=3, nrows=3, refaspect=0.75, sharey=4, sharex=0, journal="nat2", wspace=[8, 2]
)

groups = np.arange(1, len(chain_regression_df) + 1)
circled_numbers = [str(i) for i in groups]

for row in range(3):
    for col in range(3):
        ax = axs[row, col]
        config = PANEL_CONFIG.get((row, col))

        if config is None:
            ax.axis("off")
            continue

        data_key = DATA_MAPPING.get((row, col))
        mask = MASK_MAPPING.get((row, col))

        # Apply mask if available, otherwise use all data
        if mask is not None:
            coeffs = chain_regression_df.where(mask)[f"{data_key}_coefficient"].values
            ci_lower = chain_regression_df.where(mask)[f"{data_key}_ci_lower"].values
            ci_upper = chain_regression_df.where(mask)[f"{data_key}_ci_upper"].values
        else:
            coeffs = chain_regression_df[f"{data_key}_coefficient"].values
            ci_lower = chain_regression_df[f"{data_key}_ci_lower"].values
            ci_upper = chain_regression_df[f"{data_key}_ci_upper"].values

        xerr_lower = coeffs - ci_lower
        xerr_upper = ci_upper - coeffs

        # Draw horizontal bar plot
        ax.barh(
            groups,
            coeffs,
            color=config["color"],
            alpha=0.75,
            edgecolor=config["color"],
            linewidth=0,
            width=0.75,
        )

        # Add error bars
        ax.errorbar(
            coeffs,
            groups,
            xerr=[xerr_lower, xerr_upper],
            fmt="o",
            color=config["color"],
            capsize=5,
            capthick=0,
            markersize=5,
            markerfacecolor="white",
            linestyle="none",
            linewidth=2,
        )

        # Axis formatting
        ax.format(
            facecolor="white",
            xlim=config["xlim"],
            xlocator=config["xlocator"],
            ylim=(groups.min() - 0.65, groups.max() + 0.65),
            ylocator=1,
            ylabel="",
            yticklabels=(
                [] if col != 0 else None
            ),  # Only show y-axis labels on the leftmost column
            title=config["title"],
            titleweight="bold",
            grid=False,
            xtickminor=False,
            ytickminor=False,
        )

        # Draw circled group numbers on the left two columns only
        if col == 0 or col == 1:
            for group_num, label in zip(groups, circled_numbers):
                fontsize = 8 if len(label) == 1 else 7
                pad = 0.15 if len(label) == 1 else 0.1
                label_x = config["xlim"][0] - (config["xlim"][1] - config["xlim"][0]) * 0.1
                ax.text(
                    label_x,
                    group_num,
                    label,
                    ha="center",
                    va="center",
                    color="k",
                    fontweight="bold",
                    fontsize=fontsize,
                    zorder=100,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="k",
                        linewidth=1,
                        boxstyle=f"circle,pad={pad}",
                    ),
                )


# %%

# %% [markdown]
# ## ENSO legacy effects
#

# %% [markdown]
# ### Find the ENSO boundary
#

# %%
"""
Here we want to explore the impacts of ENSO strength before the recovery period on the recovery sensitivity.
"""

def find_enso_boundary(df_normal, recovery_start, max_lookback_months=24):
    """
    Look back to find the start of at least 3 consecutive months with SST < 0 (the indicator of La Niña).
    If not found, look back up to 12 months.
    """
    current_date = recovery_start
    consecutive_months = 0
    required_consecutive = 3

    for i in range(max_lookback_months):
        current_date = current_date - pd.DateOffset(months=1)
        if current_date < df_normal.index[0]:
            break
        try:
            sst_value = df_normal.loc[current_date, "nina34"]
            # if np.isnan(sst_value):
            #     # Stop period if np.nan encountered
            #     break
            if sst_value < 0:
                consecutive_months += 1

                if consecutive_months >= required_consecutive:
                    boundary_start = current_date + pd.DateOffset(months=required_consecutive)
                    return boundary_start

            else:
                consecutive_months = 0

        except (KeyError, IndexError):
            consecutive_months = 0
            continue

    # If not found, return 12 months before recovery start
    return recovery_start - pd.DateOffset(months=12)



# %%

# %%
enso_strength_results = []

for group_id in recovery_data_df["group_id"].unique():
    # Filter data for this group
    group_data = recovery_data_df[recovery_data_df["group_id"] == group_id]

    # Get the start and end dates of the recovery period
    recovery_start = group_data["time"].min()
    recovery_end = group_data["time"].max()

    # Find the ENSO event boundary
    extended_start = find_enso_boundary(
        df_normal, recovery_start, max_lookback_months=24
    )

    # Extract extended ENSO data from the main dataframe
    try:
        # Get ENSO data for the extended period
        extended_enso_data = df_normal.loc[extended_start:recovery_end, "nina34"]
        
        # Get data for the lookback period
        lookback_enso_data = df_normal.loc[extended_start:recovery_start, "nina34"]
        lookback_tws_data = df_normal.loc[extended_start:recovery_start, "tws_grace"]
        lookback_tmp_data = df_normal.loc[extended_start:recovery_start, "tmp_cru"]

        # Get ENSO values
        positive_enso = lookback_enso_data

        if len(positive_enso) > 0:
            avg_positive_enso = positive_enso.mean()
            count_positive = len(positive_enso)
        else:
            avg_positive_enso = np.nan
            count_positive = 0

        ###############################
        # -------------------- TWS
        ###############################
        if len(lookback_tws_data) >= 6:
            avg_lookback_tws = lookback_tws_data.mean()
            sum_lookback_tws = lookback_tws_data.sum()
            std_lookback_tws = lookback_tws_data.std()
            
            # 计算：最后3个月均值 - 最开始3个月均值
            if len(lookback_tws_data) >= 6:
                tws_diff_3m = lookback_tws_data.iloc[-3:].mean() - lookback_tws_data.iloc[:3].mean()
            else:
                tws_diff_3m = np.nan
            
            # 使用 calculate_regression 计算TWS趋势
            X_tws = np.arange(len(lookback_tws_data))
            Y_tws = lookback_tws_data.values
            reg_result_tws = calculate_regression(X_tws, Y_tws, conf_level=0.95, min_samples=6)
            
            if reg_result_tws is not None:
                slope_tws = reg_result_tws["coef"]
                pval_tws = reg_result_tws["pvalue"]
                # 总变化量 = 趋势 × 月数
                tws_total_change = slope_tws * len(lookback_tws_data)
            else:
                slope_tws = np.nan
                pval_tws = np.nan
                tws_total_change = np.nan
        else:
            avg_lookback_tws = np.nan
            sum_lookback_tws = np.nan
            std_lookback_tws = np.nan
            slope_tws = np.nan
            pval_tws = np.nan
            tws_total_change = np.nan
            tws_diff_3m = np.nan
        
        ###############################
        # -------------------- TMP
        ###############################
        if len(lookback_tmp_data) >= 6:
            avg_lookback_tmp = lookback_tmp_data.mean()
            sum_lookback_tmp = lookback_tmp_data.sum()
            std_lookback_tmp = lookback_tmp_data.std()
            
            # 计算：最后3个月均值 - 最开始3个月均值
            if len(lookback_tmp_data) >= 6:
                tmp_diff_3m = lookback_tmp_data.iloc[-3:].mean() - lookback_tmp_data.iloc[:3].mean()
            else:
                tmp_diff_3m = np.nan
            
            # 使用 calculate_regression 计算TMP趋势
            X_tmp = np.arange(len(lookback_tmp_data))
            Y_tmp = lookback_tmp_data.values
            reg_result_tmp = calculate_regression(X_tmp, Y_tmp, conf_level=0.95, min_samples=6)
            
            if reg_result_tmp is not None:
                slope_tmp = reg_result_tmp["coef"]
                pval_tmp = reg_result_tmp["pvalue"]
                # 总变化量 = 趋势 × 月数
                tmp_total_change = slope_tmp * len(lookback_tmp_data)
            else:
                slope_tmp = np.nan
                pval_tmp = np.nan
                tmp_total_change = np.nan
        else:
            avg_lookback_tmp = np.nan
            sum_lookback_tmp = np.nan
            std_lookback_tmp = np.nan
            slope_tmp = np.nan
            pval_tmp = np.nan
            tmp_total_change = np.nan
            tmp_diff_3m = np.nan

        # Total months in extended period
        total_extended_months = len(extended_enso_data)
        lookback_months = len(lookback_tws_data)

        # Calculate actual lookback months
        actual_lookback_months = (recovery_start.year - extended_start.year) * 12 + (
            recovery_start.month - extended_start.month
        )

    except (KeyError, IndexError):
        # Handle exceptions
        avg_positive_enso = np.nan
        count_positive = 0
        total_extended_months = len(group_data)
        actual_lookback_months = 0
        
        avg_lookback_tws = np.nan
        sum_lookback_tws = np.nan
        std_lookback_tws = np.nan
        slope_tws = np.nan
        pval_tws = np.nan
        tws_total_change = np.nan
        tws_diff_3m = np.nan
        
        avg_lookback_tmp = np.nan
        sum_lookback_tmp = np.nan
        std_lookback_tmp = np.nan
        slope_tmp = np.nan
        pval_tmp = np.nan
        tmp_total_change = np.nan
        tmp_diff_3m = np.nan
        
        lookback_months = 0

    enso_strength_results.append(
        {
            "group_id": group_id,
            "enso_avg": avg_positive_enso,
            "enso_count": count_positive,
            "enso_sum": avg_positive_enso * count_positive if not np.isnan(avg_positive_enso) else np.nan,
            
            # TWS变化相关变量
            "tws_avg": avg_lookback_tws,
            "tws_sum": sum_lookback_tws,
            "tws_std": std_lookback_tws,
            "tws_trend": slope_tws,
            "tws_trend_pval": pval_tws,
            "tws_total_change": tws_total_change,
            "tws_diff_3m": tws_diff_3m,  # 最后3月 - 最初3月
            
            # TMP变化相关变量
            "tmp_avg": avg_lookback_tmp,
            "tmp_sum": sum_lookback_tmp,
            "tmp_std": std_lookback_tmp,
            "tmp_trend": slope_tmp,
            "tmp_trend_pval": pval_tmp,
            "tmp_total_change": tmp_total_change,
            "tmp_diff_3m": tmp_diff_3m,  # 最后3月 - 最初3月
            
            "recovery_months": len(group_data),
            "extended_months": total_extended_months,
            "lookback_months": lookback_months,
            "actual_lookback_months": actual_lookback_months,
            "enso_fraction": (
                count_positive / total_extended_months
                if total_extended_months > 0
                else 0
            ),
            "extended_start": extended_start,
            "recovery_start": recovery_start,
            "recovery_end": recovery_end,
        }
    )

# Convert to DataFrame
group_enso_stats_df = pd.DataFrame(enso_strength_results)

group_enso_stats_df


# %%
# New DataFrame with ENSO strength and chain rule results
chain_df_extended = chain_regression_df.merge(group_enso_stats_df, on="group_id", how="left")


# %%
# we also need another column to store the most positive CGR value from extended_start to recovery_end, but for the robustness, we probably can select the 3 most positive CGR values
chain_df_extended["CGR_max"] = chain_df_extended.apply(
    lambda row: df_normal.loc[
        row["extended_start"]:row["recovery_end"], "CGR"
    ].nlargest(3).mean(),
    axis=1,
)


# %%

# %% [markdown]
# ### Statistical relationship
#

# %%
fig, ax = pplt.subplots(ncols=1, nrows=1, share=0, journal="nat2", refaspect=5)
groups = np.arange(1, len(chain_df_extended) + 1)

x_data = chain_df_extended["enso_avg"].values
y_data = chain_df_extended["CGR_ENSO_coefficient"].values
groups = chain_df_extended["group_id"].values

# plot scatters
sns.regplot(
    x=x_data,
    y=y_data,
    ax=ax,
    order=1,
    scatter_kws={
        "s": 200,
        "color": "white",
        "edgecolor": "#B22222",
        "linewidth": 1.5,
        "zorder": 3,
    },
    line_kws={"color": "#B22222", "linewidth": 2, "alpha": 0.8, "zorder": 2},
    ci=95,
    truncate=False,
)

# plot the group numbers
for x, y, group_num in zip(x_data, y_data, groups):
    if not (np.isnan(x) or np.isnan(y)):
        ax.text(
            x,
            y,
            str(group_num),
            ha="center",
            va="center",
            color="#B22222",
            fontsize=9,
            zorder=4,
        )

# calculate correlation
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
if valid_mask.sum() > 2:
    corr_result = pg.corr(x_data[valid_mask], y_data[valid_mask])
    r_val = corr_result["r"].values[0]
    p_val = corr_result["p_val"].values[0]
    ax.text(
        0.01,
        0.13,
        f"R = {r_val:.3f}, $\\mathit{{P}}$ = {p_val:.3f}",
        transform=ax.transAxes,
        color="#B22222",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

# format the axes
ax.format(
    xlabel="El Niño intensity [$\overline{SSTA}$; °C]",
    ylabel="$\gamma_{SSTA}^{CGR}$ [Gt C per °C]",
    grid=False,
    xlocator=0.1,
    xtickminor=False,
    ytickminor=False,
    #xlim=[0.15, 1.25],
    ylim=(0, 5),
)

# fig.savefig('03-res/figs/composite_scatter_ENSO_intensity_and_recovery_sensitivity.png', dpi=600)


# %%
# Exclude 5th groups (group_id == 5 and group_id == 10)
filtered_df = chain_df_extended[~chain_df_extended['group_id'].isin([5])]

print("Without 5th:", pg.corr(filtered_df['CGR_ENSO_coefficient'], filtered_df['enso_avg'], method='pearson'))

print("With 5th:", pg.corr(chain_df_extended['CGR_ENSO_coefficient'], chain_df_extended['enso_avg'], method='pearson'))


# %%
from scipy import stats

def bootstrap_corr(x, y, n_boot=10000, ci=95, method="pearson", seed=42):
    """Bootstrap correlation with confidence interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    if n < 3:
        return {"r": np.nan, "p_obs": np.nan, "p_boot": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}

    method = method.lower()
    if method == "pearson":
        r_obs, p_obs = stats.pearsonr(x, y)
    elif method == "spearman":
        r_obs, p_obs = stats.spearmanr(x, y)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    rng = np.random.default_rng(seed)
    boot_r = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        try:
            if method == "pearson":
                boot_r[i], _ = stats.pearsonr(xb, yb)
            else:
                boot_r[i], _ = stats.spearmanr(xb, yb)
        except Exception:
            boot_r[i] = np.nan

    boot_r = boot_r[~np.isnan(boot_r)]
    if len(boot_r) == 0:
        return {"r": r_obs, "p_obs": p_obs, "p_boot": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}

    lo, hi = np.percentile(boot_r, [(100 - ci) / 2, 100 - (100 - ci) / 2])

    # Bootstrap sign-based p-value
    p_boot = 2 * min((boot_r >= 0).mean(), (boot_r <= 0).mean())

    return {
        "r": r_obs,
        "p_obs": p_obs,
        "p_boot": p_boot,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n,
    }


for label, df in [("With group 5", chain_df_extended), ("Without group 5", filtered_df)]:
    res = bootstrap_corr(df["enso_avg"], df["CGR_ENSO_coefficient"], method="pearson")
    print(
        f"{label}: r={res['r']:.3f}, 95%CI=[{res['ci_lo']:.3f},{res['ci_hi']:.3f}], "
        f"p_obs={res['p_obs']:.3f}, p_boot={res['p_boot']:.3f}, n={res['n']}")


# %%
# === CORE ANALYSIS: 前期ENSO强度 vs CGR对ENSO敏感性 ===
print("\n" + "="*70)
print("CORE: 前期ENSO强度 vs CGR对ENSO敏感性")
print("="*70)

# %%
# === ADD EVENT 14 DIRECT ENSO-CGR COEFFICIENT ===
# Event 14 lacks TWS data, so indirect path fails
# Calculate direct ENSO-CGR correlation instead

print("\n--- Calculating Event 14 Direct ENSO-CGR Sensitivity ---")
event_14_data = recovery_data_df[recovery_data_df['group_id'] == 14]
print(f"Event 14总数据点数: {len(event_14_data)}")
print(f"日期范围: {event_14_data['time'].min()} 到 {event_14_data['time'].max()}")
print(f"应该有14个月 (2025-01到2026-02)")

# Check raw recovery_groups[-1] data
raw_event14 = recovery_groups[-1] if len(recovery_groups) > 0 else None
if raw_event14 is not None:
    print(f"\nRaw recovery_groups[-1]数据:")
    print(f"  日期范围: {raw_event14.index[0]} 到 {raw_event14.index[-1]}")
    print(f"  总行数: {len(raw_event14)}")
    print(f"  eruption标记统计: {(raw_event14['eruption'] == 1).sum()}个")
    print(f"  各列NaN统计:")
    for col in ['CGR', 'ENSO', 'TMP', 'TWS', 'PRE']:
        nan_count = raw_event14[col].isna().sum()
        valid = (~raw_event14[col].isna()).sum()
        print(f"    {col}: {valid}个有效, {nan_count}个NaN")

# Check for NaN in recovery_data_df
print(f"\n恢复后recovery_data_df中Event 14各列有效数据点数:")
for col in ['CGR', 'ENSO', 'TMP', 'TWS', 'PRE']:
    valid = (~event_14_data[col].isna()).sum()
    print(f"  {col}: {valid}")

if len(event_14_data) > 0:
    enso_vals = event_14_data['ENSO'].values
    cgr_vals = event_14_data['CGR'].values

    if len(enso_vals) > 2 and not np.all(np.isnan(enso_vals)) and not np.all(np.isnan(cgr_vals)):
        event14_result = calculate_regression(enso_vals, cgr_vals, conf_level=0.95, min_samples=3)

        print(f"Event 14 (n={len(enso_vals)}):")
        print(f"  CGR_ENSO_coefficient = {event14_result['coef']:.4f}")
        print(f"  p-value = {event14_result['pvalue']:.4f}")
        print(f"  95% CI = [{event14_result['ci_lower']:.4f}, {event14_result['ci_upper']:.4f}]")

        # Update chain_regression_df row for event 14
        chain_regression_df.loc[chain_regression_df['group_id'] == 14, 'CGR_ENSO_coefficient'] = event14_result['coef']
        chain_regression_df.loc[chain_regression_df['group_id'] == 14, 'CGR_ENSO_pvalue'] = event14_result['pvalue']
        chain_regression_df.loc[chain_regression_df['group_id'] == 14, 'CGR_ENSO_ci_lower'] = event14_result['ci_lower']
        chain_regression_df.loc[chain_regression_df['group_id'] == 14, 'CGR_ENSO_ci_upper'] = event14_result['ci_upper']

        # Update chain_df_extended
        chain_df_extended.loc[chain_df_extended['group_id'] == 14, 'CGR_ENSO_coefficient'] = event14_result['coef']

        print("✓ Event 14 CGR_ENSO_coefficient updated in datasets")

# %%
# Now recalculate with event 14 included
print("\n" + "="*70)
print("CORE: 前期ENSO强度 vs CGR对ENSO敏感性 (含Event 14)")
print("="*70)

core_data = chain_df_extended[['group_id', 'enso_avg', 'CGR_ENSO_coefficient']].dropna()
print(f"\nData availability:")
print(f"Total events: {len(chain_df_extended)}")
print(f"Events with both enso_avg & CGR_ENSO_coefficient: {len(core_data)}")

if len(core_data) > 0:
    # Compute correlation
    x = core_data['enso_avg'].values
    y = core_data['CGR_ENSO_coefficient'].values

    # Simple Pearson
    r_obs, p_obs = stats.pearsonr(x, y)

    # Bootstrap
    n = len(x)
    rng = np.random.default_rng(42)
    boot_r = []
    for _ in range(10000):
        idx = rng.integers(0, n, size=n)
        boot_r.append(stats.pearsonr(x[idx], y[idx])[0])
    boot_r = np.array(boot_r)

    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    p_boot = 2 * min((boot_r >= 0).mean(), (boot_r <= 0).mean())

    print(f"\nResults (n={n}):")
    print(f"  Pearson r = {r_obs:.4f}")
    print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  p-value (parametric) = {p_obs:.4f}")
    print(f"  p-value (bootstrap) = {p_boot:.4f}")

    # Significance interpretation
    if p_boot < 0.05:
        sig_text = "✓ SIGNIFICANT"
    elif p_boot < 0.10:
        sig_text = "~ MARGINALLY SIGNIFICANT"
    else:
        sig_text = "✗ NOT SIGNIFICANT"
    print(f"  Interpretation: {sig_text}")

    # Show which events included/excluded
    print(f"\nIncluded events (group_id): {sorted(core_data['group_id'].values)}")
    excluded = set(chain_df_extended['group_id'].values) - set(core_data['group_id'].values)
    if excluded:
        print(f"Excluded events (group_id): {sorted(excluded)}")


# %%

# %%

# %%
# =====================================================
# CGR_max vs ENSO strength (enso_avg) Analysis
# =====================================================

fig, ax = pplt.subplots(ncols=1, nrows=1, share=0, journal="nat2", refaspect=5)

x_data = chain_df_extended["enso_avg"].values
y_data = chain_df_extended["CGR_max"].values
groups = chain_df_extended["group_id"].values

# plot scatters
sns.regplot(
    x=x_data,
    y=y_data,
    ax=ax,
    order=1,
    scatter_kws={
        "s": 200,
        "color": "white",
        "edgecolor": "#B22222",
        "linewidth": 1.5,
        "zorder": 3,
    },
    line_kws={"color": "#B22222", "linewidth": 2, "alpha": 0.8, "zorder": 2},
    ci=95,
    truncate=False,
)

# plot the group numbers
for x, y, group_num in zip(x_data, y_data, groups):
    if not (np.isnan(x) or np.isnan(y)):
        ax.text(
            x,
            y,
            str(group_num),
            ha="center",
            va="center",
            color="#B22222",
            fontsize=9,
            zorder=4,
        )

# calculate correlation
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
if valid_mask.sum() > 2:
    corr_result = pg.corr(x_data[valid_mask], y_data[valid_mask])
    r_val = corr_result["r"].values[0]
    p_val = corr_result["p_val"].values[0]
    ax.text(
        0.01,
        0.95,
        f"R = {r_val:.3f}, $\\mathit{{P}}$ = {p_val:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color="#B22222",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

# format the axes
ax.format(
    xlabel="El Niño intensity [$\overline{SSTA}$; °C]",
    ylabel="Maximum CGR [Gt C]",
    grid=False,
    xlocator=0.1,
    xtickminor=False,
    ytickminor=False,
    #xlim=[0.15, 1.25],
    ylim=[0, 5],
)

# fig.savefig('03-res/figs/composite_scatter_ENSO_intensity_and_maximum_carbon_loss.png', dpi=600)


# %%
# Exclude 5th groups (group_id == 5 and group_id == 10)
filtered_df = chain_df_extended[~chain_df_extended['group_id'].isin([5])]

print("Without 5th:", pg.corr(filtered_df['enso_avg'], filtered_df['CGR_max'], method='pearson'))
print("With 5th:", pg.corr(chain_df_extended['enso_avg'], chain_df_extended['CGR_max'], method='pearson'))


# %% [markdown]
# #### Water
#

# %%
#------------------- Recovery sensitivity vs drought strength
#-------------------#-------------------#-------------------#

# Exclude 5th groups (group_id == 5 and group_id == 10)
filtered_df = chain_df_extended[~chain_df_extended['group_id'].isin([5])]

print("Without 5th:", pg.corr(filtered_df['enso_avg'], filtered_df['tws_avg'], method='pearson'))

print("With 5th:", pg.corr(chain_df_extended['enso_avg'], chain_df_extended['tws_avg'], method='pearson'))


# %%
# Water sensitivity vs ENSO strength
fig, ax = pplt.subplots(ncols=1, nrows=1, share=0, journal="nat2", refaspect=5)
groups = np.arange(1, len(chain_df_extended) + 1)

x_data = chain_df_extended["enso_avg"].values
y_data = chain_df_extended["tws_avg"].values
groups = chain_df_extended["group_id"].values

# plot scatters
sns.regplot(
    x=x_data,
    y=y_data,
    ax=ax,
    order=1,
    scatter_kws={
        "s": 200,
        "color": "white",
        "edgecolor": "#4d9221",  # 替换为蓝色（水）
        "linewidth": 1.5,
        "zorder": 3,
    },
    line_kws={"color": "#4d9221", "linewidth": 2, "alpha": 0.8, "zorder": 2},  # 替换为蓝色
    ci=95,
    truncate=False,
)

# plot the group numbers
for x, y, group_num in zip(x_data, y_data, groups):
    if not (np.isnan(x) or np.isnan(y)):
        ax.text(
            x,
            y,
            str(group_num),
            ha="center",
            va="center",
            color="#4d9221",  # 替换为蓝色
            fontsize=9,
            zorder=4,
        )

# calculate correlation
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
if valid_mask.sum() > 2:
    corr_result = pg.corr(x_data[valid_mask], y_data[valid_mask])
    r_val = corr_result["r"].values[0]
    p_val = corr_result["p_val"].values[0]
    ax.text(
        0.01,
        0.13,
        f"R = {r_val:.3f}, $\\mathit{{P}}$ = {p_val:.3f}",
        transform=ax.transAxes,
        color="#4d9221",  # 替换为蓝色
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

# format the axes
ax.format(
    xlabel="El Niño intensity [$\overline{SSTA}$, °C]",
    ylabel="TWS [mm]",  # 替换ylabel
    grid=False,
    xlocator=0.1,
    xtickminor=False,
    ytickminor=False,
    #xlim=[0.2, 1.3],
    #ylim=(0, 5), 
)

# fig.savefig('03-res/figs/composite_scatter_ENSO_intensity_and_TWS.png', dpi=600)


# %%
#------------------- Recovery sensitivity vs drought strength
#-------------------#-------------------#-------------------#

# Exclude 5th groups (group_id == 5 and group_id == 10)
filtered_df = chain_df_extended[~chain_df_extended['group_id'].isin([5])]

print("Without 5th:", pg.corr(filtered_df['tws_avg'], filtered_df['CGR_ENSO_coefficient'], method='pearson'))

print("With 5th:", pg.corr(chain_df_extended['tws_avg'], chain_df_extended['CGR_ENSO_coefficient'], method='pearson'))


# %%
# Water sensitivity vs ENSO strength
fig, ax = pplt.subplots(ncols=1, nrows=1, share=0, journal="nat2", refaspect=5)
groups = np.arange(1, len(chain_df_extended) + 1)

x_data = chain_df_extended["tws_avg"].values
y_data = chain_df_extended["CGR_ENSO_coefficient"].values
groups = chain_df_extended["group_id"].values

# plot scatters
sns.regplot(
    x=x_data,
    y=y_data,
    ax=ax,
    order=1,
    scatter_kws={
        "s": 200,
        "color": "white",
        "edgecolor": "#4d9221",
        "linewidth": 1.5,
        "zorder": 3,
    },
    line_kws={"color": "#4d9221", "linewidth": 2, "alpha": 0.8, "zorder": 2}, 
    ci=95,
    truncate=False,
)

# plot the group numbers
for x, y, group_num in zip(x_data, y_data, groups):
    if not (np.isnan(x) or np.isnan(y)):
        ax.text(
            x,
            y,
            str(group_num),
            ha="center",
            va="center",
            color="#4d9221",  # 替换为蓝色
            fontsize=9,
            zorder=4,
        )

# calculate correlation
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
if valid_mask.sum() > 2:
    corr_result = pg.corr(x_data[valid_mask], y_data[valid_mask])
    r_val = corr_result["r"].values[0]
    p_val = corr_result["p_val"].values[0]
    ax.text(
        0.01,
        0.13,
        f"R = {r_val:.3f}, $\\mathit{{P}}$ = {p_val:.3f}",
        transform=ax.transAxes,
        color="#4d9221",  # 替换为蓝色
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

# format the axes
ax.format(
    xlabel="TWS [mm]",
    ylabel="$\gamma_{CGR}^{SSTA}$ [Gt C per °C]",  # 替换ylabel
    grid=False,
    xlocator=2,
    xtickminor=False,
    ytickminor=False,
    #xlim=[-8, 8],
    ylim=(0, 5), 
)

# fig.savefig('03-res/figs/composite_scatter_TWS_and_CGR_sensitivity.png', dpi=600)


# %%

# %% [markdown]
# #### Heat
#

# %%
#------------------- Recovery sensitivity vs heat strength
#-------------------#-------------------#-------------------#

# Exclude 5th groups (group_id == 5 and group_id == 10)
filtered_df = chain_df_extended[~chain_df_extended['group_id'].isin([5])]

print("Without 5th:", pg.corr(filtered_df['enso_avg'], filtered_df['tmp_avg'], method='pearson'))

print("With 5th:", pg.corr(chain_df_extended['enso_avg'], chain_df_extended['tmp_avg'], method='pearson'))

# Water sensitivity vs ENSO strength
fig, ax = pplt.subplots(ncols=1, nrows=1, share=0, journal="nat2", refaspect=5)
groups = np.arange(1, len(chain_df_extended) + 1)

x_data = chain_df_extended["enso_avg"].values
y_data = chain_df_extended["tmp_avg"].values
groups = chain_df_extended["group_id"].values

# plot scatters
sns.regplot(
    x=x_data,
    y=y_data,
    ax=ax,
    order=1,
    scatter_kws={
        "s": 200,
        "color": "white",
        "edgecolor": "#f16913", 
        "linewidth": 1.5,
        "zorder": 3,
    },
    line_kws={"color": "#f16913", "linewidth": 2, "alpha": 0.8, "zorder": 2}, 
    ci=95,
    truncate=False,
)

# plot the group numbers
for x, y, group_num in zip(x_data, y_data, groups):
    if not (np.isnan(x) or np.isnan(y)):
        ax.text(
            x,
            y,
            str(group_num),
            ha="center",
            va="center",
            color="#f16913", 
            fontsize=9,
            zorder=4,
        )

# calculate correlation
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
if valid_mask.sum() > 2:
    corr_result = pg.corr(x_data[valid_mask], y_data[valid_mask])
    r_val = corr_result["r"].values[0]
    p_val = corr_result["p_val"].values[0]
    ax.text(
        0.01,
        0.13,
        f"R = {r_val:.3f}, $\\mathit{{P}}$ = {p_val:.3f}",
        transform=ax.transAxes,
        color="#f16913", 
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

# format the axes
ax.format(
    xlabel="El Niño intensity [$\overline{SSTA}$, °C]",
    ylabel="TMP [°C]", 
    grid=False,
    xlocator=0.1,
    xtickminor=False,
    ytickminor=False,
    #xlim=[0.15, 1.25],
    ylim=[-0.25, 0.25],
)

fig.savefig('03-res/figs/composite_scatter_ENSO_intensity_and_TMP.png', dpi=600)


# %%
#------------------- Recovery sensitivity vs heat strength
#-------------------#-------------------#-------------------#

# Exclude 5th groups (group_id == 5 and group_id == 10)
filtered_df = chain_df_extended[~chain_df_extended['group_id'].isin([5])]

print("Without 5th:", pg.corr(filtered_df['tmp_avg'], filtered_df['CGR_ENSO_coefficient'], method='pearson'))

print("With 5th:", pg.corr(chain_df_extended['tmp_avg'], chain_df_extended['CGR_ENSO_coefficient'], method='pearson'))

# Water sensitivity vs ENSO strength
fig, ax = pplt.subplots(ncols=1, nrows=1, share=0, journal="nat2", refaspect=5)
groups = np.arange(1, len(chain_df_extended) + 1)

x_data = chain_df_extended["tmp_avg"].values
y_data = chain_df_extended["CGR_ENSO_coefficient"].values
groups = chain_df_extended["group_id"].values

# plot scatters
sns.regplot(
    x=x_data,
    y=y_data,
    ax=ax,
    order=1,
    scatter_kws={
        "s": 200,
        "color": "white",
        "edgecolor": "#f16913",  # 替换为蓝色（水）
        "linewidth": 1.5,
        "zorder": 3,
    },
    line_kws={"color": "#f16913", "linewidth": 2, "alpha": 0.8, "zorder": 2},  # 替换为蓝色
    ci=95,
    truncate=False,
)

# plot the group numbers
for x, y, group_num in zip(x_data, y_data, groups):
    if not (np.isnan(x) or np.isnan(y)):
        ax.text(
            x,
            y,
            str(group_num),
            ha="center",
            va="center",
            color="#f16913",  # 替换为蓝色
            fontsize=9,
            zorder=4,
        )

# calculate correlation
valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
if valid_mask.sum() > 2:
    corr_result = pg.corr(x_data[valid_mask], y_data[valid_mask])
    r_val = corr_result["r"].values[0]
    p_val = corr_result["p_val"].values[0]
    ax.text(
        0.01,
        0.13,
        f"R = {r_val:.3f}, $\\mathit{{P}}$ = {p_val:.3f}",
        transform=ax.transAxes,
        color="#f16913",  
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0),
    )

# format the axes
ax.format(
    xlabel="TMP [°C]",
    ylabel="$\gamma_{CGR}^{SSTA}$ [Gt C per °C]",  # 替换ylabel
    grid=False,
    #xlocator=50,
    xtickminor=False,
    ytickminor=False,
    #xlim=[-0.25, 0.25],
    ylim=(0, 5), 
)

fig.savefig('03-res/figs/composite_scatter_TMP_and_CGR_sensitivity.png', dpi=600)


# %%

# %% [markdown]
# ### Spatial water and heat maps
#

# %%
####################################
# -------------- TWS changes
####################################

event_changes_TWS = []

for idx, row in chain_df_extended.iterrows():
    # Skip if TWS data unavailable for this period
    tws_times = ds_TWS_grace.time.values
    n_times = int(((tws_times >= np.datetime64(row['extended_start'])) &
                   (tws_times <= np.datetime64(row['recovery_start']))).sum())
    if n_times < 3:
        event_changes_TWS.append(None)
        continue

    # First phase
    test = ds_TWS_grace.where(
        (ds_TWS_grace.time >= row['extended_start']) &
        (ds_TWS_grace.time <= row['recovery_start']),
        drop=True
    )

    # Calculate the changes (trend × time)
    trend_result = linear_trend(test, dim='time')
    spatial_change = trend_result['slope'] * len(test['time'])
    spatial_sig    = trend_result['p_value']
    
    # 新增：计算平均和累计
    spatial_mean = test.mean(dim='time')  # 平均TWS
    spatial_sum = test.sum(dim='time')    # 累计TWS

    event_changes_TWS.append({
        'group_id': row['group_id'],
        'spatial_change': spatial_change,
        'spatial_sig': spatial_sig,
        'spatial_mean': spatial_mean,   # 新增
        'spatial_sum': spatial_sum       # 新增
    })


####################################
# -------------- TMP changes
####################################

event_changes_TMP = []

for idx, row in chain_df_extended.iterrows():
    tmp_times = ds_TMP_cru.time.values
    n_times = int(((tmp_times >= np.datetime64(row['extended_start'])) &
                   (tmp_times <= np.datetime64(row['recovery_start']))).sum())
    if n_times < 3:
        event_changes_TMP.append(None)
        continue

    # First phase
    test = ds_TMP_cru.where(
        (ds_TMP_cru.time >= row['extended_start']) &
        (ds_TMP_cru.time <= row['recovery_start']),
        drop=True
    )

    # Calculate the trends (trend × time)
    trend_result = linear_trend(test, dim='time')
    spatial_change = trend_result['slope'] * len(test['time'])
    spatial_sig    = trend_result['p_value']
    
    # 新增：计算平均和累计
    spatial_mean = test.mean(dim='time')  # 平均TMP
    spatial_sum = test.sum(dim='time')    # 累计TMP

    # Store the result
    event_changes_TMP.append({
        'group_id': row['group_id'],
        'spatial_change': spatial_change,
        'spatial_sig': spatial_sig,
        'spatial_mean': spatial_mean,   # 新增
        'spatial_sum': spatial_sum       # 新增
    })


# %%
####################################
# -------------- Plot maps
####################################

num_events = len(event_changes_TMP)
num_vars = 2  # TWS & TMP

# Configure colormap
var_config = {
    "TWS": {
        "label": "TWS change [mm]",
        "cmap": "BrBG",
        "unit": "mm",
        "vmin": -150,
        "vmax": 150,
        "locator": 45,
    },
    "TMP": {
        "label": "TMP change [°C]",
        "cmap": "RdBu_r",
        "unit": "°C",
        "vmin": -1,
        "vmax": 1,
        "locator": 0.3,
    },
}
var_names = ["TWS", "TMP"]

# Create the maps
fig, axs = pplt.subplots(
    ncols=num_vars, nrows=num_events, proj="cyl", share=3, refaspect=5, journal="nat2"
)

# Format all subplots
axs.format(
    # elements
    coast=True,
    coastcolor="k",
    coastlinewidth=1,
    coastzorder=10,
    # grid lines
    gridminor=False,
    grid=False,
    latlines=15,
    lonlines=20,
    labels=False,
    lonlim=(-120, 180),
    latlim=(-24, 24),
    reso="lo",
)

# Plot the data
for row in range(num_events):
    for col in range(num_vars):
        ax = axs[row, col]
        var_name = var_names[col]
        config = var_config[var_name]

        # Acquire the data
        if var_name == "TWS":
            entry = event_changes_TWS[row]
        else:
            entry = event_changes_TMP[row]

        if entry is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            continue

        trend_data = entry['spatial_change']

        # Plot the spatial maps
        im = ax.pcolormesh(
            trend_data.lon,
            trend_data.lat,
            trend_data.where(trend_data != 0, np.nan),
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"],
            extend="both",
            discrete=False,
            zorder=1,
        )

        # Add the colorbar in the last
        if row == num_events - 1:
            ax.colorbar(
                im,
                loc="b",
                label=f"{config['label']} [{config['unit']}]",
                width=0.10,
                shrink=0.75,
                locator=config["locator"],
            )

        if col == 0:
            ax.text(
                -0.05,
                0.50,
                str(row + 1),
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                color="k",
                fontsize=8,
                bbox=dict(facecolor="gray1", edgecolor="k", boxstyle="circle", alpha=1),
            )


# %%
####################################
# -------------- Plot maps (Mean & Sum)
####################################

num_events = len(event_changes_TMP)
num_vars = 2  # TWS & TMP

# Configure colormap for Mean
var_config_mean = {
    "TWS": {
        "label": "TWS average",
        "cmap": "BrBG",
        "unit": "mm",
        "vmin": -30,
        "vmax": 30,
        "locator": 10,
    },
    "TMP": {
        "label": "TMP average",
        "cmap": "RdBu_r",
        "unit": "°C",
        "vmin": -0.5,
        "vmax": 0.5,
        "locator": 0.2,
    },
}

var_names = ["TWS", "TMP"]

# ============== Plot Mean ==============
fig1, axs1 = pplt.subplots(
    ncols=num_vars, nrows=num_events, proj="cyl", share=3, refaspect=5, journal="nat2"
)

axs1.format(
    coast=True,
    coastcolor="k",
    coastlinewidth=1,
    coastzorder=10,
    gridminor=False,
    grid=False,
    latlines=15,
    lonlines=20,
    labels=False,
    lonlim=(-120, 180),
    latlim=(-24, 24),
    reso="lo",
)

# Plot Mean data
for row in range(num_events):
    for col in range(num_vars):
        ax = axs1[row, col]
        var_name = var_names[col]
        config = var_config_mean[var_name]

        # Acquire the data
        entry = event_changes_TWS[row] if var_name == "TWS" else event_changes_TMP[row]
        if entry is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            continue
        mean_data = entry['spatial_mean']

        # Plot
        im = ax.pcolormesh(
            mean_data.lon,
            mean_data.lat,
            mean_data.where(mean_data != 0, np.nan),
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"],
            extend="both",
            discrete=False,
            zorder=1,
        )

        # Colorbar
        if row == num_events - 1:
            ax.colorbar(
                im,
                loc="b",
                label=f"{config['label']} [{config['unit']}]",
                width=0.10,
                shrink=0.75,
                locator=config["locator"],
            )

        # Event number
        if col == 0:
            ax.text(
                -0.05, 0.50, str(row + 1),
                transform=ax.transAxes,
                rotation=90, ha="center", va="center",
                color="k", fontsize=8,
                bbox=dict(facecolor="gray1", edgecolor="k", boxstyle="circle", alpha=1),
            )

# fig1.savefig('03-res/figs/maps_comparison_strong_and_weak_ENSO.png', dpi=600)


# %%

# %% [markdown]
# ### Area statistics analysis
#

# %%
def calculate_pixel_area(lat, lon, resolution_deg=0.5):
    """
    Calculate pixel area (km²) for each grid cell, return as xarray.DataArray.
    """
    R = 6371.0  # Earth radius in km
    dlat = np.deg2rad(resolution_deg)
    dlon = np.deg2rad(resolution_deg)
    # Broadcast lat/lon to 2D meshgrid
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    area = (R**2) * dlat * dlon * np.cos(np.deg2rad(lat2d))
    # Return as DataArray with correct dims and coords
    return xr.DataArray(area, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="pixel_area")

lat = ds_TMP_cru['lat'].values
lon = ds_TMP_cru['lon'].values
resolution_deg = abs(lat[1] - lat[0])

pixel_area = calculate_pixel_area(lat, lon, resolution_deg)


# %%
######################## TWS
tws_statics = []
for idx in range(len(event_changes_TWS)):
    if event_changes_TWS[idx] is None:
        tws_statics.append(None)
        continue
    da = event_changes_TWS[idx]['spatial_mean']
    sig = event_changes_TWS[idx]['spatial_sig']
    # Only count land (valid data region)
    valid_mask = da.notnull()
    # Negative TWS and significant (p < 0.05)
    neg_sig_mask = (da < 0) & (sig < 0.05) & valid_mask

    # Number of significant negative TWS pixels
    area_neg_tws_count = np.sum(neg_sig_mask.values)

    # Total area of significant negative TWS pixels
    area_neg_tws_km2 = np.nansum(pixel_area.where(neg_sig_mask))

    # Area-weighted mean TWS change over land
    weighted_sum_tws = np.nansum(da.where(valid_mask).values * pixel_area.where(valid_mask).values) / np.nansum(pixel_area.where(valid_mask).values)

    tws_statics.append({
        'group_id': event_changes_TWS[idx]['group_id'],
        'area_neg_TWS_count': area_neg_tws_count,
        'area_neg_TWS_km2': area_neg_tws_km2,
        'weighted_sum_TWS': weighted_sum_tws
    })

######################## TMP
tmp_statics = []
for idx in range(len(event_changes_TMP)):
    if event_changes_TMP[idx] is None:
        tmp_statics.append(None)
        continue
    da = event_changes_TMP[idx]['spatial_mean']
    sig = event_changes_TMP[idx]['spatial_sig']
    # Only count land (valid data region)
    valid_mask = da.notnull()
    # Positive TMP and significant (p < 0.05)
    pos_sig_mask = (da > 0) & (sig < 0.05) & valid_mask

    # Number of significant positive TMP pixels
    area_pos_tmp_count = np.sum(pos_sig_mask.values)

    # Total area of significant positive TMP pixels
    area_pos_tmp_km2 = np.nansum(pixel_area.where(pos_sig_mask))

    # Area-weighted mean TMP change over land
    weighted_sum_tmp = np.nansum(da.where(valid_mask).values * pixel_area.where(valid_mask).values) / np.nansum(pixel_area.where(valid_mask).values)

    tmp_statics.append({
        'group_id': event_changes_TMP[idx]['group_id'],
        'area_pos_TMP_count': area_pos_tmp_count,
        'area_pos_TMP_km2': area_pos_tmp_km2,
        'weighted_sum_TMP': weighted_sum_tmp
    })

tws_area_df = pd.DataFrame([s for s in tws_statics if s is not None])
tmp_area_df = pd.DataFrame([s for s in tmp_statics if s is not None])
area_weighted_df = tws_area_df.merge(tmp_area_df, on='group_id')


# %%
# Merge area stats back onto chain_df_extended (inner join drops events without data)
area_merged_df = chain_df_extended.merge(area_weighted_df, on='group_id', how='inner')

# ---- TWS
print("=== enso_avg ~ area_neg_TWS_km2 ===")
corr_tws_area = pg.corr(
    area_merged_df['enso_avg'],
    area_merged_df['area_neg_TWS_km2'],
    method='pearson'
)
print(corr_tws_area)

print("=== enso_avg ~ weighted_sum_TWS ===")
corr_tws_weighted = pg.corr(
    area_merged_df['enso_avg'],
    area_merged_df['weighted_sum_TWS'],
    method='pearson'
)
print(corr_tws_weighted)


# %%
# ---- TMP
print("=== enso_avg ~ area_pos_TMP_km2 ===")
corr_tmp_area = pg.corr(
    area_merged_df['enso_avg'],
    area_merged_df['area_pos_TMP_km2'],
    method='pearson'
)
print(corr_tmp_area)

print("\n=== enso_avg ~ weighted_sum_TMP ===")
corr_tmp_weighted = pg.corr(
    area_merged_df['enso_avg'],
    area_merged_df['weighted_sum_TMP'],
    method='pearson'
)
print(corr_tmp_weighted)


# %%

# %% [markdown]
# ### Composite analysis
#

# %%
# Sort by ENSO strength and get indices for top/bottom 3
sorted_indices = chain_df_extended['enso_avg'].sort_values().index
strong_enso_indices = sorted_indices[-5:] # 2, 6, 11, 8, 5 [+1]
weak_enso_indices   = sorted_indices[:5]  # 4, 3, 9, 1, 7. [+1]

pval_threshold = 1


# %%
# Get spatial changes for TWS and TMP (skip events with no data)
# 1. Strong
strong_enso_tws     = [event_changes_TWS[i]['spatial_mean'] for i in strong_enso_indices if event_changes_TWS[i] is not None]
strong_enso_tws_sig = [event_changes_TWS[i]['spatial_sig']  for i in strong_enso_indices if event_changes_TWS[i] is not None]
strong_enso_tmp     = [event_changes_TMP[i]['spatial_mean'] for i in strong_enso_indices if event_changes_TMP[i] is not None]
strong_enso_tmp_sig = [event_changes_TMP[i]['spatial_sig']  for i in strong_enso_indices if event_changes_TMP[i] is not None]
# 2. Weak
weak_enso_tws     = [event_changes_TWS[i]['spatial_mean'] for i in weak_enso_indices if event_changes_TWS[i] is not None]
weak_enso_tws_sig = [event_changes_TWS[i]['spatial_sig']  for i in weak_enso_indices if event_changes_TWS[i] is not None]
weak_enso_tmp     = [event_changes_TMP[i]['spatial_mean'] for i in weak_enso_indices if event_changes_TMP[i] is not None]
weak_enso_tmp_sig = [event_changes_TMP[i]['spatial_sig']  for i in weak_enso_indices if event_changes_TMP[i] is not None]

# Calculate mean spatial anomalies using xarray
# 1. Strong
mean_strong_tws = xr.concat(strong_enso_tws, dim='event').mean(dim='event')
mean_strong_tws_sig = xr.concat(strong_enso_tws_sig, dim='event').mean(dim='event')
mean_strong_tws = mean_strong_tws.where(mean_strong_tws_sig < pval_threshold)

mean_strong_tmp = xr.concat(strong_enso_tmp, dim='event').mean(dim='event')
mean_strong_tmp_sig = xr.concat(strong_enso_tmp_sig, dim='event').mean(dim='event')
mean_strong_tmp = mean_strong_tmp.where(mean_strong_tmp_sig < pval_threshold)

# 2. Weak
mean_weak_tws = xr.concat(weak_enso_tws, dim='event').mean(dim='event')
mean_weak_tws_sig = xr.concat(weak_enso_tws_sig, dim='event').mean(dim='event')
mean_weak_tws = mean_weak_tws.where(mean_weak_tws_sig < pval_threshold)

mean_weak_tmp = xr.concat(weak_enso_tmp, dim='event').mean(dim='event')
mean_weak_tmp_sig = xr.concat(weak_enso_tmp_sig, dim='event').mean(dim='event')
mean_weak_tmp = mean_weak_tmp.where(mean_weak_tmp_sig < pval_threshold)


# %%
var_config = {
    "TWS": {
        "label": "TWS Changes",
        "cmap": "BrBG",
        "unit": "mm",
        "vmin": -30,
        "vmax": 30,
        "locator": 30,
    },
    "TMP": {
        "label": "TMP Changes",
        "cmap": "RdBu_r",
        "unit": "°C",
        "vmin": -0.5,
        "vmax": 0.5,
        "locator": 0.50,
    },
}
composite_data = {
    "TWS": {"Weak El Niño": mean_weak_tws, "Strong El Niño": mean_strong_tws},
    "TMP": {"Weak El Niño": mean_weak_tmp, "Strong El Niño": mean_strong_tmp},
}

row_labels = ["TWS", "TMP"]
left_labels = ["TWS [mm]", "TMP [°C]"]
col_labels = ["Weak El Niño", "Strong El Niño"]

fig, axs = pplt.subplots(
    ncols=2, nrows=2, journal="nat2", proj="cyl", refaspect=5
)
axs.format(
    coast=True,
    coastcolor="k",
    coastlinewidth=1,
    coastzorder=10,
    gridminor=False,
    grid=False,
    latlines=20,
    lonlines=45,
    labels=False,
    lonlim=(-120, 180),
    latlim=(-30, 30),
    reso="lo",
    toplabels=col_labels,
    leftlabels=left_labels,
    toplabelweight="normal",
    leftlabelweight="normal",
    toplabelpad=1,
    leftlabelpad=3, 
)

for row, var_name in enumerate(row_labels):
    for col, enso_strength in enumerate(col_labels):
        ax = axs[row, col]
        config = var_config[var_name]
        data = composite_data[var_name][enso_strength]
        im = ax.pcolormesh(
            data.lon,
            data.lat,
            data.where(data != 0, np.nan),
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"],
            discrete=False,
            zorder=1, extend="both",
        )
       
        if col == len(col_labels) - 1:
            ax.colorbar(im, loc="r", label="", shrink=0.7, locator=config["locator"], width=0.10, extendsize=0.5)

for ax in axs:
    ax.spines['geo'].set_linewidth(0)

# fig.savefig('03-res/figs/composite_maps_comparison_strong_and_weak_ENSO.png', dpi=600)


# %%
# 2. Area of negative TWS and positive TMP

# Get group_ids for strong/weak indices
strong_group_ids = chain_df_extended.loc[strong_enso_indices, 'group_id'].values
weak_group_ids   = chain_df_extended.loc[weak_enso_indices, 'group_id'].values

# TWS (lookup by group_id, skip missing)
strong_tws_area = [tws_area_df.loc[tws_area_df['group_id'] == gid, 'area_neg_TWS_km2'].values[0]
                   for gid in strong_group_ids if gid in tws_area_df['group_id'].values]
weak_tws_area   = [tws_area_df.loc[tws_area_df['group_id'] == gid, 'area_neg_TWS_km2'].values[0]
                   for gid in weak_group_ids if gid in tws_area_df['group_id'].values]
strong_tws_area = np.array(strong_tws_area) if strong_tws_area else np.array([np.nan])
weak_tws_area   = np.array(weak_tws_area) if weak_tws_area else np.array([np.nan])

# TMP (lookup by group_id, skip missing)
strong_tmp_area = [tmp_area_df.loc[tmp_area_df['group_id'] == gid, 'area_pos_TMP_km2'].values[0]
                   for gid in strong_group_ids if gid in tmp_area_df['group_id'].values]
weak_tmp_area   = [tmp_area_df.loc[tmp_area_df['group_id'] == gid, 'area_pos_TMP_km2'].values[0]
                   for gid in weak_group_ids if gid in tmp_area_df['group_id'].values]
strong_tmp_area = np.array(strong_tmp_area) if strong_tmp_area else np.array([np.nan])
weak_tmp_area   = np.array(weak_tmp_area) if weak_tmp_area else np.array([np.nan])

# Mean and std
mean_tws_area = [np.mean(strong_tws_area), np.mean(weak_tws_area)]
err_tws_area  = [np.std(strong_tws_area), np.std(weak_tws_area)]

mean_tmp_area = [np.mean(strong_tmp_area), np.mean(weak_tmp_area)]
err_tmp_area  = [np.std(strong_tmp_area), np.std(weak_tmp_area)]

# 显著性检验
u_stat_tws, p_val_tws = mannwhitneyu(strong_tws_area, weak_tws_area, alternative='two-sided')
u_stat_tmp, p_val_tmp = mannwhitneyu(strong_tmp_area, weak_tmp_area, alternative='two-sided')

print(f"TWS Area: U={u_stat_tws:.2f}, p={p_val_tws:.3f}")
print(f"TMP Area: U={u_stat_tmp:.2f}, p={p_val_tmp:.3f}")


# %%
print("Negative TWS Area", mean_tws_area,)

print("Positive TMP Area", mean_tmp_area)


# %%
print("TWS Anomalies", [np.mean(mean_strong_tws).values*5, np.mean(mean_weak_tws).values]*1) 

print("TMP Anomalies", [np.mean(mean_strong_tmp).values*1, np.mean(mean_weak_tmp).values]*1)


# %%
fig, axs = pplt.subplots(ncols=4, nrows=1, share=0, journal="nat2", refaspect=2.25,)

# Variable configuration
bar_data = [
    # 1. TWS
    # TWS Changes
    {
        "mean": [np.mean(mean_strong_tws)*5, np.mean(mean_weak_tws)],
        "err": [np.std(mean_strong_tws), np.std(mean_weak_tws)],
        "color": ["#00441b", "#a6dba0"],
        "title": "Average [mm]",
        "ylim": [-30,30],
        "ylocator": 30
    },
    # Negative TWS Area
    {
        "mean": mean_tws_area,
        "err": err_tws_area,
        "color": ["#00441b", "#a6dba0"],
        "title": "Area [km$^2$]",
        "ylim": [0,6e7],
        "ylocator": 3e7
    },
    # 2. TMP
    # TMP Changes
    {
        "mean": [np.mean(mean_strong_tmp), np.mean(mean_weak_tmp)],
        "err": [np.std(mean_strong_tmp), np.std(mean_weak_tmp)],
        "color": ["#d94801", "#fdd0a2"],
        "title": "Average [°C]",
        "ylim": [-0.3,0.3],
        "ylocator": 0.3
    },
    # Negative TMP Area
    {
        "mean": mean_tmp_area,
        "err": err_tmp_area,
        "color": ["#d94801", "#fdd0a2"],
        "title": "Area [km$^2$]",
        "ylim": [0,6e7],
        "ylocator": 3e7
    },
]

# Plot the data
for i, data in enumerate(bar_data):

    axs[0, i].bar([0, 1], data["mean"], color=data["color"], alpha=0.75, width=0.5)

    axs[0, i].errorbar(
        [0, 1],
        data["mean"],
        yerr=data["err"],
        fmt="o",
        elinewidth=2,
        capsize=0,
        capthick=2,
        alpha=1,
        markersize=7,
        markerfacecolor="white",
        color=data["color"][0],
        ecolor=data["color"][0],
        markeredgecolor=data["color"][0],
        linestyle="none",
        linewidth=2,
        zorder=10,
    )
    axs[0, i].format(xgrid=True, ygrid=False, 
        xlocator=[0, 1], xticklabels=["Strong\nEl Niño", "Weak\nEl Niño"], title=data["title"], xlim=(-0.7, 1.7), ylim=data["ylim"], ylocator=data["ylocator"], ylabel="", xtickminor=False,
    )


# revise the spines
if True:
    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

# add the p values
p_val_list = [0.000,  0.056, 0.000, 0.095]
for ax, p_val in zip(axs, p_val_list):
    ax.text(
        0.05, 0.95, f"$\\mathit{{P}}$ = {p_val:.3f}",            
        transform=ax.transAxes,
        ha="left", va="top",
        color="black",
        zorder=10
    )

axs.format(grid=True)

fig.text(
    0.26, 0.10, "TWS", ha="center", va="top", color="#00441b", fontweight="bold"
);
fig.text(
    0.76, 0.10, "TMP", ha="center", va="top", color="#d94801", fontweight="bold"
);

fig.savefig('03-res/figs/composite_bars_comparison_strong_and_weak_ENSO.png', dpi=600)


# %%
# water storage values for statistical test

tws_weak = mean_weak_tws.values.flatten()
tws_weak = tws_weak[~np.isnan(tws_weak)]

tws_strong = mean_strong_tws.values.flatten()
tws_strong = tws_strong[~np.isnan(tws_strong)]

# Mann-Whitney U test for difference between two distributions
u_stat, p_value = mannwhitneyu(tws_strong, tws_weak, alternative='two-sided')

print(f"Mann-Whitney U: {u_stat:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("The two groups have a significant difference in distribution.")
else:
    print("No significant difference between the two group distributions.")


# %%
# water storage values for statistical test

tmp_weak = mean_weak_tmp.values.flatten()
tmp_weak = tmp_weak[~np.isnan(tmp_weak)]

tmp_strong = mean_strong_tmp.values.flatten()
tmp_strong = tmp_strong[~np.isnan(tmp_strong)]

# Mann-Whitney U test for difference between two distributions
u_stat, p_value = mannwhitneyu(tmp_strong, tmp_weak, alternative='two-sided')

print(f"Mann-Whitney U: {u_stat:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("The two groups have a significant difference in distribution.")
else:
    print("No significant difference between the two group distributions.")


# %%

# %% [markdown]
# # Decompostion Methods
#

# %% [markdown]
# ## Variance Analysis for the Sensitivity
#

# %%
def variance_decomposition_linear_with_residual_general(chain_df, pathway="water"):
    """
    Use linear variance decomposition method to analyze the contributions of components along a specified pathway (including residuals).

    # Note: Var(X1 x X2) = (mean_X2)² x Var(X1) + (mean_X1)² x Var(X2) + 2 x mean_X1 x mean_X2 x Cov(X1,X2)

    Parameters:
    -----------
    chain_df : DataFrame
    pathway : str
        'water' or 'heat'

    Returns:
    --------
    dict : results
    """
    total_var = "CGR_SST_coefficient"  # Total effect (ΔCGR/ΔSST)
    if pathway == "water":
        eco_var = "CGR_TWS_coefficient"  # Ecological sensitivity (ΔCGR/ΔTWS)
        climate_var = "TWS_SST_coefficient"  # Climate sensitivity (ΔTWS/ΔSST)
        pathway_name = "Water Pathway"
        eco_name = "ΔCGR/ΔTWS"
        climate_name = "ΔTWS/ΔSST"
    elif pathway == "heat":
        eco_var = "CGR_TMP_coefficient"  # Ecological sensitivity (ΔCGR/ΔTMP)
        climate_var = "TMP_SST_coefficient"  # Climate sensitivity (ΔTMP/ΔSST)
        pathway_name = "Heat Pathway"
        eco_name = "ΔCGR/ΔTMP"
        climate_name = "ΔTMP/ΔSST"
    else:
        raise ValueError("pathway must be 'water' or 'heat'")

    valid_mask = ~(
        np.isnan(chain_df[total_var])
        | np.isnan(chain_df[eco_var])
        | np.isnan(chain_df[climate_var])
    )

    valid_data = chain_df[valid_mask].copy()

    # Total (Y)
    Y = valid_data[total_var].values

    # Ecological sensitivity component (X1)
    X1 = valid_data[eco_var].values

    # Climate sensitivity component (X2)
    X2 = valid_data[climate_var].values

    # Variance calculations
    var_Y = np.var(Y, ddof=1)
    var_X1 = np.var(X1, ddof=1)
    var_X2 = np.var(X2, ddof=1)

    # Covariance calculations
    cov_X1_X2 = np.cov(X1, X2, ddof=1)[0, 1]

    # Mean calculations
    mean_X1 = np.mean(X1)
    mean_X2 = np.mean(X2)

    # Linear variance decomposition
    var_contrib_X1 = (mean_X2**2) * var_X1
    var_contrib_X2 = (mean_X1**2) * var_X2
    var_contrib_interaction = 2 * mean_X1 * mean_X2 * cov_X1_X2

    # Total predicted variance
    var_predicted = var_contrib_X1 + var_contrib_X2 + var_contrib_interaction

    # ★ Calculate residual variance ★
    var_residual = var_Y - var_predicted

    # Calculate percentage contributions
    total_contrib = (
        abs(var_contrib_X1)
        + abs(var_contrib_X2)
        + abs(var_contrib_interaction)
        + abs(var_residual)
    )

    if total_contrib > 0:
        pct_X1 = (abs(var_contrib_X1) / total_contrib) * 100
        pct_X2 = (abs(var_contrib_X2) / total_contrib) * 100
        pct_interaction = (abs(var_contrib_interaction) / total_contrib) * 100
        pct_residual = (abs(var_residual) / total_contrib) * 100
    else:
        pct_X1 = pct_X2 = pct_interaction = pct_residual = 0

    # Explained variance ratio
    explained_variance_ratio = var_predicted / var_Y if var_Y > 0 else 0

    results = {
        "method": f"linear_variance_decomposition_with_residual_{pathway}",
        "pathway": pathway,
        "pathway_name": pathway_name,
        "n_groups": len(valid_data),
        "observed_variance": var_Y,
        "predicted_variance": var_predicted,
        "residual_variance": var_residual,
        "explained_variance_ratio": explained_variance_ratio,
        "ecological_variance_contrib": var_contrib_X1,
        "climate_variance_contrib": var_contrib_X2,
        "interaction_variance_contrib": var_contrib_interaction,
        "residual_variance_contrib": var_residual,
        "ecological_percentage": pct_X1,
        "climate_percentage": pct_X2,
        "interaction_percentage": pct_interaction,
        "residual_percentage": pct_residual,
        "ecological_name": eco_name,
        "climate_name": climate_name,
        "mean_ecological": mean_X1,
        "mean_climate": mean_X2,
        "var_ecological": var_X1,
        "var_climate": var_X2,
        "cov_ecological_climate": cov_X1_X2,
    }

    return results



# %%
print("=" * 60)
print("Variance Decomposition: Water and Heat Pathways")
print("=" * 60)

# Analyze Water Pathway
water_variance_results = variance_decomposition_linear_with_residual_general(
    chain_regression_df[water_mask], pathway="water"
)

# Analyze Heat Pathway
temperature_variance_results = variance_decomposition_linear_with_residual_general(
    chain_regression_df[heat_mask], pathway="heat"
)

# Show results
for pathway_results, pathway_name in [
    (water_variance_results, "Water Pathway"),
    (temperature_variance_results, "Heat Pathway"),
]:
    if pathway_results:
        print(f"\n=== {pathway_name} Variance Decomposition, with Residual ===")
        print(f"Valid groups: {pathway_results['n_groups']}")
        print(f"Observed variance: {pathway_results['observed_variance']:.6f}")
        print(f"Predicted variance: {pathway_results['predicted_variance']:.6f}")
        print(f"Residual variance: {pathway_results['residual_variance']:.6f}")
        print(
            f"Explained variance ratio: {pathway_results['explained_variance_ratio']:.3f}"
        )
        print(f"\nVariance contribution breakdown:")
        print(
            f"  Ecological sensitivity ({pathway_results['ecological_name']}): {pathway_results['ecological_variance_contrib']}, {pathway_results['ecological_percentage']:.1f}%"
        )
        print(
            f"  Climate sensitivity ({pathway_results['climate_name']}): {pathway_results['climate_variance_contrib']}, {pathway_results['climate_percentage']:.1f}%"
        )
        print(
            f"  Interaction: {pathway_results['interaction_variance_contrib']}, {pathway_results['interaction_percentage']:.1f}%"
        )
        print(
            f"  Residual: {pathway_results['residual_variance_contrib']}, {pathway_results['residual_percentage']:.1f}%"
        )
    else:
        print(f"\n=== {pathway_name} Variance Decomposition Failed ===")
        print("Insufficient data or other issues")


# %%
# ===============================================
# Ecological vs Climate Sensitivity Scatter Plot
# ===============================================

# Create figure with two panels
fig, axes = pplt.subplots(
    ncols=2, nrows=1, refaspect=1.0, sharey=False, sharex=False, journal="nat2"
)

# Color configuration
PATHWAY_COLORS = {
    "water": ["#4682B4", "#B0C4DE"],  # Steel Blue for water pathway
    "temp": ["#DAA520", "#D5D0A7"],  # Goldenrod for temperature pathway
}

# Panel configurations
PANEL_CONFIG = {
    0: {
        "colors": PATHWAY_COLORS["water"],
        "eco_var": "CGR_TWS",  # Ecological sensitivity: ΔCGR/ΔTWS
        "climate_var": "TWS_SST",  # Climate sensitivity: ΔTWS/ΔSST
        "xlabel": "ΔTWS/ΔSST [mm per °C]",
        "ylabel": "ΔCGR/ΔTWS [Gt C per mm]",
        "mask": water_mask,
        "xlim": (0, -250),
        "xlocator": 50,
        "ylim": (0, -0.06),  # (0.01, -0.09),
        "ylocator": 0.012,
    },
    1: {
        "colors": PATHWAY_COLORS["temp"],
        "eco_var": "CGR_TMP",  # Ecological sensitivity: ΔCGR/ΔTMP
        "climate_var": "TMP_SST",  # Climate sensitivity: ΔTMP/ΔSST
        "xlabel": "ΔTMP/ΔSST [°C per °C]",
        "ylabel": "ΔCGR/ΔTMP [Gt C per °C]",
        "mask": temperature_mask,
        "xlim": (0, 0.5),
        "xlocator": 0.1,
        "ylim": (0, 35),
        "ylocator": 7,
    },
}

# Get group numbers for labeling
groups = np.arange(1, len(chain_df) + 1)
circled_numbers = [str(i) for i in groups]

for i, (ax, config) in enumerate(zip(axes, PANEL_CONFIG.values())):

    # Extract data with mask applied
    mask = config["mask"]

    # Apply mask to extract only valid data points
    eco_sens = chain_df.loc[mask, f'{config["eco_var"]}_coefficient'].values  # Y-axis
    climate_sens = chain_df.loc[
        mask, f'{config["climate_var"]}_coefficient'
    ].values  # X-axis

    total_sens = chain_df.loc[mask, "CGR_SST_coefficient"].values  # Total sensitivity

    # Extract confidence intervals for error bars (with mask)
    eco_ci_lower = chain_df.loc[mask, f'{config["eco_var"]}_ci_lower'].values
    eco_ci_upper = chain_df.loc[mask, f'{config["eco_var"]}_ci_upper'].values
    climate_ci_lower = chain_df.loc[mask, f'{config["climate_var"]}_ci_lower'].values
    climate_ci_upper = chain_df.loc[mask, f'{config["climate_var"]}_ci_upper'].values

    # Get group numbers for masked data
    masked_groups = groups[mask]
    masked_circled_numbers = [str(j + 1) for j in range(len(groups)) if mask[j]]

    # Calculate error bars
    eco_yerr_lower = eco_sens - eco_ci_lower
    eco_yerr_upper = eco_ci_upper - eco_sens
    climate_xerr_lower = climate_sens - climate_ci_lower
    climate_xerr_upper = climate_ci_upper - climate_sens

    # Plot the scatter plot
    scatter = ax.scatter(
        climate_sens,
        eco_sens,
        s=150,
        alpha=0.70,
        c=total_sens,
        cmap="Boreal",
        vmin=1,
        vmax=3,
        extend="both",
        edgecolor="white",
        linewidth=0,
        discrete=False,
        zorder=5,
    )

    # Create scatter plot with error bars
    for j, (x, y, group_num) in enumerate(
        zip(climate_sens, eco_sens, masked_circled_numbers)
    ):

        # Plot point with mixed color (average of X and Y colors)
        # ax.scatter(x, y, s=120, facecolor=config['colors'][0], alpha=0.8,
        #             edgecolor='white', linewidth=0, zorder=5)

        # Add error bars with X and Y specific colors
        # X-axis error bars (horizontal) - use X color
        ax.errorbar(
            x,
            y,
            xerr=[[climate_xerr_lower[j]], [climate_xerr_upper[j]]],
            fmt="none",
            color=config["colors"][0],
            alpha=0.8,
            capsize=0,
            capthick=1.5,
            linewidth=2,
            zorder=3,
        )
        # Y-axis error bars (vertical) - use Y color
        ax.errorbar(
            x,
            y,
            yerr=[[eco_yerr_lower[j]], [eco_yerr_upper[j]]],
            fmt="none",
            color=config["colors"][1],
            alpha=0.8,
            capsize=0,
            capthick=1.5,
            linewidth=2,
            zorder=3,
        )

        # Add group number labels
        ax.text(
            x,
            y,
            group_num,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8,
            zorder=6,
        )

    # # Add regression line
    # sns.regplot(x=climate_sens, y=eco_sens,
    #         order=1, ax=ax,
    #         color='k',
    #         line_kws={'color': 'k', 'linewidth': 2, 'alpha': 0.75, 'zorder': 1},
    #         ci=95, truncate=False)

    # Add reference lines with red color for climatology
    if i == 0:  # Water pathway
        ref_eco = ref_sensitivity_df[ref_sensitivity_df["relationship"] == "CGR_vs_TWS"]
        ref_climate = ref_sensitivity_df[
            ref_sensitivity_df["relationship"] == "TWS_vs_ENSO"
        ]

        # Reference lines in red for climatology
        ax.axhline(
            y=ref_eco["coefficient"].values,
            color="k",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            zorder=2,
        )
        ax.axvline(
            x=ref_climate["coefficient"].values,
            color="k",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            zorder=2,
        )

        # Confidence intervals as gray shaded areas
        ax.axvspan(
            ref_climate["ci_lower"].values[0],
            ref_climate["ci_upper"].values[0],
            alpha=0.2,
            color="k",
            linewidth=0,
            zorder=1,
        )

        ax.axhspan(
            ref_eco["ci_lower"].values[0],
            ref_eco["ci_upper"].values[0],
            alpha=0.2,
            color="k",
            linewidth=0,
            zorder=1,
        )

    else:  # Temperature pathway
        ref_eco = ref_sensitivity_df[ref_sensitivity_df["relationship"] == "CGR_vs_TMP"]
        ref_climate = ref_sensitivity_df[
            ref_sensitivity_df["relationship"] == "TMP_vs_ENSO"
        ]

        # Reference lines in red for climatology
        ax.axhline(
            y=ref_eco["coefficient"].values[0],
            color="k",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            zorder=2,
        )
        ax.axvline(
            x=ref_climate["coefficient"].values[0],
            color="k",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
            zorder=2,
        )

        # Confidence intervals as gray shaded areas
        ax.axvspan(
            ref_climate["ci_lower"].values[0],
            ref_climate["ci_upper"].values[0],
            alpha=0.2,
            color="k",
            linewidth=0,
            zorder=1,
        )

        ax.axhspan(
            ref_eco["ci_lower"].values[0],
            ref_eco["ci_upper"].values[0],
            alpha=0.2,
            color="k",
            linewidth=0,
            zorder=1,
        )

    # Format axes with X and Y specific colors for labels AND tick labels
    ax.format(
        facecolor="gray3",
        xlabel=config["xlabel"],
        xlabelweight="bold",
        ylabel=config["ylabel"],
        ylabelweight="bold",
        xlabelcolor=config["colors"][0],
        ylabelcolor=config["colors"][1],
        xticklabelcolor=config["colors"][0],
        yticklabelcolor=config["colors"][1],
        xticklabelweight="bold",
        yticklabelweight="bold",
        xtickcolor=config["colors"][0],
        ytickcolor=config["colors"][1],
        grid=False,
        gridcolor="w",
        gridalpha=0.3,
        xtickminor=False,
        ytickminor=False,
        xlim=config["xlim"],
        xlocator=config["xlocator"],
        ylim=config["ylim"],
        ylocator=config["ylocator"],
    )

colorbar = fig.colorbar(
    scatter,
    loc="r",
    width=0.1,
    shrink=0.9,
    label="ΔCGR/ΔSST [Gt C per °C]",
    labelweight="bold",
    locator=0.5,
    ticklabelweight="bold",
)


# %%

# %% [markdown]
# ## Drivers of Sensitivity Anomaly
#

# %%
# ===============================
# Component Contribution Analysis
# ===============================
print("=== Component Contribution Analysis ===")

""" 
ΔA.     = ΔBxC₁      + B₁xΔC      + ΔBxΔC           + Residual
(A2-A1) = (B2-B1)xC₁ + B1x(C2-C1) + (B2-B1)x(C2-C1) + Residual

where A is ΔCGR/ΔSST, B is ΔCGR/ΔTWS (or ΔCGR/ΔTMP), and C is ΔTWS/ΔSST (or ΔTMP/ΔSST).
"""

# Set the reference coefficients
ref_cgr_sst = ref_coefficients["CGR_SST"]
ref_cgr_tws = ref_coefficients["CGR_TWS"]
ref_tws_sst = ref_coefficients["TWS_SST"]
ref_cgr_tmp = ref_coefficients["CGR_TMP"]
ref_tmp_sst = ref_coefficients["TMP_SST"]

# Define masks for pathway significance
water_mask = chain_df["CGR_TWS_pvalue"] < 0.05
heat_mask = chain_df["CGR_TMP_pvalue"] < 0.05

# Store results
decomposition_results = []

for i, row in chain_df.iterrows():
    group_id = row["group_id"]
    print(f"\n--- Group {group_id} ---")

    row_data = {"group_id": group_id}

    # Get recovery period coefficients
    A2_cgr_sst = row["CGR_SST_coefficient"]  # Total effect: CGR→SST
    B2_cgr_tws = row["CGR_TWS_coefficient"]  # Water pathway: CGR→TWS
    C2_tws_sst = row["TWS_SST_coefficient"]  # Water pathway: TWS→SST
    B2_cgr_tmp = row["CGR_TMP_coefficient"]  # Heat pathway: CGR→TMP
    C2_tmp_sst = row["TMP_SST_coefficient"]  # Heat pathway: TMP→SST

    # ==========================
    #   Water Pathway Analysis
    # ==========================
    # Check if this group passes the water pathway significance test
    if water_mask.iloc[i] and not (
        np.isnan(A2_cgr_sst) or np.isnan(B2_cgr_tws) or np.isnan(C2_tws_sst)
    ):

        # Calculate changes from reference group
        Delta_A = A2_cgr_sst - ref_cgr_sst
        Delta_B = B2_cgr_tws - ref_cgr_tws
        Delta_C = C2_tws_sst - ref_tws_sst

        # Decompose contributions using reference as baseline
        contrib_B = Delta_B * ref_tws_sst  # ΔB × C1
        contrib_C = ref_cgr_tws * Delta_C  # B1 × ΔC
        contrib_interaction = Delta_B * Delta_C  # ΔB × ΔC
        residual = Delta_A - (contrib_B + contrib_C + contrib_interaction)

        # Calculate absolute percentages (including residual)
        abs_sum = (
            abs(contrib_B) + abs(contrib_C) + abs(contrib_interaction) + abs(residual)
        )
        if abs_sum > 0:
            pct_B = abs(contrib_B) / abs_sum * 100
            pct_C = abs(contrib_C) / abs_sum * 100
            pct_int = abs(contrib_interaction) / abs_sum * 100
            pct_res = abs(residual) / abs_sum * 100
        else:
            pct_B = pct_C = pct_int = pct_res = 0

        # Store water pathway results
        row_data.update(
            {
                "water_total_change": Delta_A,
                "water_contrib_cgr_tws": contrib_B,
                "water_contrib_tws_sst": contrib_C,
                "water_contrib_interaction": contrib_interaction,
                "water_residual": residual,
                "water_abs_pct_cgr_tws": pct_B,
                "water_abs_pct_tws_sst": pct_C,
                "water_abs_pct_interaction": pct_int,
                "water_abs_pct_residual": pct_res,
            }
        )

        print(f"Water pathway:")
        print(f"  Total change: {Delta_A:.4f}")
        print(f"  CGR→TWS: {contrib_B:.4f} ({pct_B:.1f}%)")
        print(f"  TWS→SST: {contrib_C:.4f} ({pct_C:.1f}%)")
        print(f"  Interaction: {contrib_interaction:.4f} ({pct_int:.1f}%)")
        print(f"  Residual: {residual:.4f} ({pct_res:.1f}%)")
    else:
        # Fill with NaN for groups that don't pass water pathway significance test
        for key in [
            "water_total_change",
            "water_contrib_cgr_tws",
            "water_contrib_tws_sst",
            "water_contrib_interaction",
            "water_residual",
            "water_abs_pct_cgr_tws",
            "water_abs_pct_tws_sst",
            "water_abs_pct_interaction",
            "water_abs_pct_residual",
        ]:
            row_data[key] = np.nan

        if not water_mask.iloc[i]:
            print("Water pathway: Not significant (p >= 0.05)")
        else:
            print("Water pathway: Insufficient data")

    # ===============================
    # Temperature Pathway Analysis
    # ===============================
    # Check if this group passes the temperature pathway significance test
    if temperature_mask.iloc[i] and not (
        np.isnan(A2_cgr_sst) or np.isnan(B2_cgr_tmp) or np.isnan(C2_tmp_sst)
    ):
        # Calculate changes from reference group
        Delta_A = A2_cgr_sst - ref_cgr_sst
        Delta_B = B2_cgr_tmp - ref_cgr_tmp
        Delta_C = C2_tmp_sst - ref_tmp_sst

        # Decompose contributions using reference as baseline
        contrib_B = Delta_B * ref_tmp_sst  # ΔB × C1
        contrib_C = ref_cgr_tmp * Delta_C  # B1 × ΔC
        contrib_interaction = Delta_B * Delta_C  # ΔB × ΔC
        residual = Delta_A - (contrib_B + contrib_C + contrib_interaction)

        # Calculate absolute percentages (including residual)
        abs_sum = (
            abs(contrib_B) + abs(contrib_C) + abs(contrib_interaction) + abs(residual)
        )
        if abs_sum > 0:
            pct_B = abs(contrib_B) / abs_sum * 100
            pct_C = abs(contrib_C) / abs_sum * 100
            pct_int = abs(contrib_interaction) / abs_sum * 100
            pct_res = abs(residual) / abs_sum * 100
        else:
            pct_B = pct_C = pct_int = pct_res = 0

        # Store temperature pathway results
        row_data.update(
            {
                "temp_total_change": Delta_A,
                "temp_contrib_cgr_tmp": contrib_B,
                "temp_contrib_tmp_sst": contrib_C,
                "temp_contrib_interaction": contrib_interaction,
                "temp_residual": residual,
                "temp_abs_pct_cgr_tmp": pct_B,
                "temp_abs_pct_tmp_sst": pct_C,
                "temp_abs_pct_interaction": pct_int,
                "temp_abs_pct_residual": pct_res,
            }
        )

        print(f"Temperature pathway:")
        print(f"  Total change: {Delta_A:.4f}")
        print(f"  CGR→TMP: {contrib_B:.4f} ({pct_B:.1f}%)")
        print(f"  TMP→SST: {contrib_C:.4f} ({pct_C:.1f}%)")
        print(f"  Interaction: {contrib_interaction:.4f} ({pct_int:.1f}%)")
        print(f"  Residual: {residual:.4f} ({pct_res:.1f}%)")
    else:
        # Fill with NaN for groups that don't pass temperature pathway significance test
        for key in [
            "temp_total_change",
            "temp_contrib_cgr_tmp",
            "temp_contrib_tmp_sst",
            "temp_contrib_interaction",
            "temp_residual",
            "temp_abs_pct_cgr_tmp",
            "temp_abs_pct_tmp_sst",
            "temp_abs_pct_interaction",
            "temp_abs_pct_residual",
        ]:
            row_data[key] = np.nan

        if not temperature_mask.iloc[i]:
            print("Temperature pathway: Not significant (p >= 0.05)")
        else:
            print("Temperature pathway: Insufficient data")

    decomposition_results.append(row_data)

# Create DataFrame
decomposition_df = pd.DataFrame(decomposition_results)

decomposition_df


# %%

# %%
# Summary statistics
print(f"\n=== Summary Statistics ===")
for pathway in ["water", "temp"]:
    pathway_name = "Water" if pathway == "water" else "Temperature"
    var_name = "tws" if pathway == "water" else "tmp"

    cgr_pct = decomposition_df[f"{pathway}_abs_pct_cgr_{var_name}"].dropna()
    sst_pct = decomposition_df[f"{pathway}_abs_pct_{var_name}_sst"].dropna()
    int_pct = decomposition_df[f"{pathway}_abs_pct_interaction"].dropna()
    res_pct = decomposition_df[f"{pathway}_abs_pct_residual"].dropna()

    if len(cgr_pct) > 0:
        print(f"\n{pathway_name} pathway ({len(cgr_pct)} valid groups):")
        print(f"  CGR component: {cgr_pct.mean():.1f}% ± {cgr_pct.std():.1f}%")
        print(f"  SST component: {sst_pct.mean():.1f}% ± {sst_pct.std():.1f}%")
        print(f"  Interaction: {int_pct.mean():.1f}% ± {int_pct.std():.1f}%")
        print(f"  Residual: {res_pct.mean():.1f}% ± {res_pct.std():.1f}%")


# %%
# ----------------------------------------
# Validation on the decomposition results
# ----------------------------------------

fig, axes = pplt.subplots(
    ncols=2, nrows=1, refaspect=1.0, sharey=False, sharex=False, journal="nat2"
)

# calculate observed and predicted values
water_observed = decomposition_df["water_total_change"]
water_predicted = (
    decomposition_df["water_contrib_cgr_tws"]
    + decomposition_df["water_contrib_tws_sst"]
    + decomposition_df["water_contrib_interaction"]
)

temp_observed = decomposition_df["temp_total_change"]
temp_predicted = (
    decomposition_df["temp_contrib_cgr_tmp"]
    + decomposition_df["temp_contrib_tmp_sst"]
    + decomposition_df["temp_contrib_interaction"]
)

# plot observed vs predicted values
axes[0].scatter(
    water_observed[water_mask],
    water_predicted[water_mask],
    s=120,
    facecolor="white",
    edgecolor="#4682B4",
    linewidth=1,
    alpha=0.8,
    zorder=5,
)

axes[1].scatter(
    temp_observed[temperature_mask],
    temp_predicted[temperature_mask],
    s=120,
    facecolor="white",
    edgecolor="#DAA520",
    linewidth=1,
    alpha=0.8,
    zorder=5,
)

# add labels for each group
groups = np.arange(1, len(decomposition_df) + 1)
for i, group_num in enumerate(groups):
    # Water pathway labels
    if not (np.isnan(water_observed.iloc[i]) or np.isnan(water_predicted.iloc[i])):
        axes[0].text(
            water_observed.iloc[i],
            water_predicted.iloc[i],
            str(group_num),
            ha="center",
            va="center",
            color="#4682B4",
            fontsize=8,
            zorder=6,
        )

    # Temperature pathway labels
    if not (np.isnan(temp_observed.iloc[i]) or np.isnan(temp_predicted.iloc[i])):
        axes[1].text(
            temp_observed.iloc[i],
            temp_predicted.iloc[i],
            str(group_num),
            ha="center",
            va="center",
            color="#DAA520",
            fontsize=8,
            zorder=6,
        )

# Draw 1:1 reference line
axes.plot([-0.75, 4], [-0.75, 4], color="k", linewidth=2, zorder=3)

# Calculate and display correlation coefficients
water_corr = pg.corr(water_observed.dropna(), water_predicted.dropna())["r"].values[0]
temp_corr = pg.corr(temp_observed.dropna(), temp_predicted.dropna())["r"].values[0]

water_pval = pg.corr(water_observed.dropna(), water_predicted.dropna())["p-val"].values[
    0
]
temp_pval = pg.corr(temp_observed.dropna(), temp_predicted.dropna())["p-val"].values[0]

axes[0].text(
    0.05,
    0.95,
    f"R = {water_corr:.3f}, $\\mathit{{P}}$ = {water_pval:.3f}",
    transform=axes[0].transAxes,
    fontsize=10,
    color="#4682B4",
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0),
)

axes[1].text(
    0.05,
    0.95,
    f"R = {temp_corr:.3f}, $\\mathit{{P}}$ = {temp_pval:.3f}",
    transform=axes[1].transAxes,
    fontsize=10,
    color="#DAA520",
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0),
)

axes.format(
    grid=False,
    gridcolor="gray",
    gridalpha=0.3,
    xlim=(-0.75, 4),
    ylim=(-0.75, 4),
    xlocator=1,
    ylocator=1,
    xlabel="Prediction",
    ylabel="Observation",
)

fig.format(toplabels=["Water", "Heat"])


# %%
# ===========================================================
# Sensitivity Contribution Stack Plot (Decomposition Results)
# ===========================================================

# Function to cumulate positive and negative values separately for stacked bar plotting
def get_cumulated_array(data, **kwargs):
    """
    Take negative and positive data apart and cumulate
    """
    # Take negative and positive data apart and cumulate
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

# Create the figure with two columns for water and heat pathways
fig, axes = pplt.subplots(ncols=2, nrows=1, refaspect=0.7, sharey=4, journal="nat2")

# Color configuration for each pathway and contribution type
COLORS = {
    "water": {"eco": "#2c5aa0", "climate": "#5b9bd5", "interaction": "#cce4f7"},
    "temp": {"eco": "#a0522d", "climate": "#cd853f", "interaction": "#f5e6d3"},
}

LABELS = {
    "water": {"eco": "ΔCGR/ΔTWS", "climate": "ΔTWS/ΔSST", "interaction": "Interaction"},
    "temp":  {"eco": "ΔCGR/ΔTMP", "climate": "ΔTMP/ΔSST", "interaction": "Interaction"}
}

# Group/event indices and circled number labels for y-axis
groups = np.arange(1, len(decomposition_df) + 1)
circled_numbers = [str(i) for i in groups]

# Pathway configuration for each subplot
pathway_configs = {
    0: {"pathway": "water", "show_labels": True},
    1: {"pathway": "temp", "show_labels": False},
}

# Loop over each pathway (water and heat) and plot the stacked contributions
for i, (ax, config) in enumerate(zip(axes, pathway_configs.values())):
    pathway = config["pathway"]

    # Extract decomposition contribution values for the selected pathway
    if pathway == "water":
        eco_contrib = decomposition_df["water_contrib_cgr_tws"].values
        climate_contrib = decomposition_df["water_contrib_tws_sst"].values
        interaction_contrib = decomposition_df["water_contrib_interaction"].values
        total_change = decomposition_df["water_total_change"].values
    else:
        eco_contrib = decomposition_df["temp_contrib_cgr_tmp"].values
        climate_contrib = decomposition_df["temp_contrib_tmp_sst"].values
        interaction_contrib = decomposition_df["temp_contrib_interaction"].values
        total_change = decomposition_df["temp_total_change"].values

    # Replace NaN with 0 for plotting, but keep original signs
    eco_contrib = np.nan_to_num(eco_contrib, nan=0)
    climate_contrib = np.nan_to_num(climate_contrib, nan=0)
    interaction_contrib = np.nan_to_num(interaction_contrib, nan=0)
    total_change = np.nan_to_num(total_change, nan=0)

    # Prepare data for stacked bar plotting (rows: eco, climate, interaction)
    data = np.zeros([3, len(groups)])
    data[0, :] = eco_contrib
    data[1, :] = climate_contrib
    data[2, :] = interaction_contrib

    # Calculate cumulated values for stacking (positive and negative separately)
    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)

    # Merge negative and positive cumulated data for correct stacking
    row_mask = data < 0
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data

    # Get the colors for each contribution type
    colors = [
        COLORS[pathway]["eco"],
        COLORS[pathway]["climate"],
        COLORS[pathway]["interaction"],
    ]

    labels = [
        LABELS[pathway]["eco"],
        LABELS[pathway]["climate"],
        LABELS[pathway]["interaction"],
    ]

    # Plot stacked horizontal bars for each contribution type
    data_shape = np.shape(data)
    for j in np.arange(0, data_shape[0]):
        ax.barh(
            groups,
            data[j],
            left=data_stack[j],
            color=colors[j],
            alpha=1,
            edgecolor="white",
            linewidth=0,
            label=labels[j]
        )

    # Add markers for total change verification (vertical lines)
    valid_mask = ~np.isnan(total_change) & (total_change != 0)

    if np.any(valid_mask):
        # Add scatter points showing where the total change should be
        ax.scatter(
            total_change[valid_mask],
            groups[valid_mask],
            marker="o",
            s=50,
            color="white",
            edgecolor="#B22222",
            linewidth=1,
            alpha=1,
            zorder=5,
            label="ΔCGR/ΔSST"
        )

    # Add event number labels (circled numbers) for the first panel only
    if config["show_labels"]:
        for group_num, label in zip(groups, circled_numbers):
            fontsize = 8 if len(label) == 1 else 7
            pad = 0.15 if len(label) == 1 else 0.1

            ax.text(
                -4,
                group_num,
                label,
                ha="center",
                va="center",
                color="k",
                fontweight="bold",
                fontsize=fontsize,
                zorder=100,
                bbox=dict(
                    facecolor="white",
                    edgecolor="k",
                    linewidth=1,
                    boxstyle=f"circle,pad={pad}",
                ),
            )

    # Format axes appearance and ticks
    ax.format(
        facecolor="white",
        xlim=(-3.5, 5.5),
        xlocator=1,
        ylim=(groups.min() - 0.75, groups.max() + 0.75),
        ylocator=1,
        xlabel="",
        ylabel="",
        yticklabels=[] if not config["show_labels"] else None,
        ygrid=False,
        xgrid=False,
        gridcolor="gray",
        gridalpha=0.35,
        xtickminor=True,
        ytickminor=False,
    )

    # Add vertical reference line at zero
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.7)

    # Add legend for the contributions
    ax.legend(loc='b', ncols=2, frameon=False)

# Add left and top labels for the figure
fig.format(
    leftlabels=["Carbon Recovery Events"],
    toplabels=["Water Pathway", "Heat Pathway"],
)


# %%
