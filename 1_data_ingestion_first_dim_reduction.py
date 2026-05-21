"""
First-step ingestion and feature reduction pipeline.

Main goals:
- Load raw AVQ data.
- Keep valid rows for target variables.
- Reduce dimensionality without throwing away too many features.
- Handle NaN values with imputation instead of dropping rows.
- Save cleaned datasets for downstream scripts.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

# -----------------------------
# Configuration
# -----------------------------
MISSING_COL_THRESHOLD = 0.60
LOW_VARIANCE_THRESHOLD = 0.01
HIGH_CORR_THRESHOLD = 0.90
TARGET_CORR_THRESHOLD = 0.02
NUMERIC_PARSE_RATIO = 0.90

# Read the data
start_load = time.time()
print("\nLoading data...")
raw_df = pd.read_csv("data/AVQ_Microdati_2021.csv")
end_load = time.time()
print("Time to load data: {:.2f} minutes".format((end_load - start_load) / 60))

n_obs = raw_df.shape[0]

#%% Target variables
# The target variables are:
# - "SPRENER" (fare attenzione a non sprecare energia elettrica)
# - "SPRACQUA" (fare attenzione a non sprecare acqua)
print("\nTarget variables...")

target_vars = ["SPRENER", "SPRACQUA"]
raw_df[target_vars].describe()

# Check for missing values
print("Missing values in targets:")
print(raw_df[target_vars].isnull().sum())

# remove missing values from dataset
df = raw_df.dropna(subset=target_vars).copy()

# Target variables are categorical, let's plot their distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i, var in enumerate(target_vars):
    df[var].value_counts().plot(kind="bar", ax=ax[i], title=var)
plt.tight_layout()
plt.savefig("exploratory_analysis/target_vars_distribution_original.png")

# encode the target categorical variables with strings
#map_encode = {1: "Usually",
#              2: "Sometimes",
#              3: "Rarely",
#              4: "Never"}
#df[target_vars] = df[target_vars].replace(map_encode)

# remove target vars from dataset
sprener = df.pop("SPRENER")
spracqua = df.pop("SPRACQUA")

#%% Reduce number of independent variables
print("\nReducing number of independent variables...")

initial_n_cols = df.shape[1]

# Standardize empty strings to NaN first
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

# Convert object columns to numeric when possible.
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
forced_numeric_cols = []
still_non_numeric_cols = []
for col in object_cols:
    converted = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index)
    non_null = df[col].notna().sum()
    parse_ratio = converted.notna().sum() / non_null if non_null > 0 else 1.0
    if parse_ratio >= NUMERIC_PARSE_RATIO:
        df[col] = converted
        forced_numeric_cols.append(col)
    else:
        still_non_numeric_cols.append(col)

if forced_numeric_cols:
    print(f"Converted {len(forced_numeric_cols)} object columns to numeric.")

# Drop genuinely non-numeric columns to keep pipeline simple for ML scripts.
if still_non_numeric_cols:
    print(f"Dropping {len(still_non_numeric_cols)} non-numeric columns: {still_non_numeric_cols}")
    df.drop(columns=still_non_numeric_cols, inplace=True)

### Remove variables with very high missingness
missing_ratio = df.isnull().sum() / len(df)
high_missing_cols = missing_ratio[missing_ratio > MISSING_COL_THRESHOLD].index.tolist()
if high_missing_cols:
    print(f"Dropping {len(high_missing_cols)} columns with > {MISSING_COL_THRESHOLD:.0%} missing values.")
    df.drop(columns=high_missing_cols, inplace=True)

# check again for missing values after initial cleaning
missing_ratio_after = df.isnull().sum() / len(df)
print("Max missing ratio after filtering: {:.2%}".format(missing_ratio_after.max()))

### Identify constant variables
constant_vars = df.columns[df.nunique(dropna=True) <= 1]
# Variables with constant values across all samples won't contribute to the model. Remove them.
if len(constant_vars) > 0:
    print(f"Dropping {len(constant_vars)} constant columns.")
    df.drop(constant_vars, axis=1, inplace=True)

### Remove known non-informative financial coefficient if present.
if "COEFIN" in df.columns:
    df.drop("COEFIN", axis=1, inplace=True)

# Variables with very low variance carry little signal.
temp_for_stats = df.fillna(df.median(numeric_only=True))
variance = temp_for_stats.var()
low_var_cols = variance[variance < LOW_VARIANCE_THRESHOLD].index.tolist()
if low_var_cols:
    print(f"Dropping {len(low_var_cols)} low-variance columns (< {LOW_VARIANCE_THRESHOLD}).")
    df.drop(columns=low_var_cols, inplace=True)

### Remove variables with high correlation
corr_input = df.fillna(df.median(numeric_only=True))
corr_matrix = corr_input.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than threshold
to_drop = [column for column in upper.columns if any(upper[column] > HIGH_CORR_THRESHOLD)]
# remove variables with high correlation
if to_drop:
    print(f"Dropping {len(to_drop)} highly correlated columns (>{HIGH_CORR_THRESHOLD}).")
    df.drop(columns=to_drop, inplace=True)

### Remove variables with low correlation with target variables
# compute correlation between target variables and independent variables
corr_input = df.fillna(df.median(numeric_only=True))
corr_sprener = pd.Series(corr_input.corrwith(sprener))
corr_spracqua = pd.Series(corr_input.corrwith(spracqua))
low_corr_sprener = set()
for col, value in corr_sprener.items():
    try:
        if abs(float(value)) < TARGET_CORR_THRESHOLD:
            low_corr_sprener.add(col)
    except (TypeError, ValueError):
        continue

low_corr_spracqua = set()
for col, value in corr_spracqua.items():
    try:
        if abs(float(value)) < TARGET_CORR_THRESHOLD:
            low_corr_spracqua.add(col)
    except (TypeError, ValueError):
        continue

# Drop only columns weak for BOTH targets (less aggressive than union).
to_drop_low_target_corr = list(low_corr_sprener & low_corr_spracqua)
if to_drop_low_target_corr:
    print(
        f"Dropping {len(to_drop_low_target_corr)} columns with weak correlation "
        f"(< {TARGET_CORR_THRESHOLD}) for both targets."
    )
    df.drop(columns=to_drop_low_target_corr, inplace=True)

### remove columns sf9 sf11 sf13 sf14 sf15 which are correlated with variable MH
df.drop(["SF9", "SF11", "SF13", "SF14", "SF15"], axis=1, inplace=True, errors="ignore")

### NANs: columns that still contain missing values
nan_cols = df.columns[df.isna().any()]
print("Columns with NaN before imputation: {}".format(list(nan_cols)))

# Inspect correlation with targets for columns that have NaN values.
if len(nan_cols) > 0:
    corr_sprener_nan = df[nan_cols].corrwith(sprener)
    corr_spracqua_nan = df[nan_cols].corrwith(spracqua)

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    corr_sprener_nan.plot(kind="bar", ax=ax[0], title="SPRENER")
    corr_spracqua_nan.plot(kind="bar", ax=ax[1], title="SPRACQUA")
    plt.ylabel("Correlation")
    plt.xlabel("Variables with NaN values")
    plt.tight_layout()
    plt.savefig("exploratory_analysis/to_impute_correlation_with_target_vars.png")


#%% SAVE PREPROCESSED DATA
print("\nHere we have {} observations, {} independent variables, which {} of them presents nan values."
    .format(df.shape[0], df.shape[1], len(nan_cols)))

df_nan = df.copy()
df_nan.to_csv("data/preprocessed_data_with_nan.csv", index=False)

# Impute remaining NaN values column-wise with the median to preserve features and rows.
medians = df.median(numeric_only=True)
df = df.fillna(medians)

# Final safety check: downstream scripts expect no NaN in features.
if df.isna().sum().sum() > 0:
    raise ValueError("NaN values still present after imputation. Please inspect preprocessing steps.")

df.to_csv("data/preprocessed_data.csv", index=False)
print("After imputation dataset has {} observations and {} features.".format(df.shape[0], df.shape[1]))
print("Feature reduction: {} -> {} columns.".format(initial_n_cols, df.shape[1]))

# save target variables
sprener[df.index].to_csv("data/tar_sprener.csv", index=False)
spracqua[df.index].to_csv("data/tar_spracqua.csv", index=False)

print("\nPreprocessed datasets saved.")

