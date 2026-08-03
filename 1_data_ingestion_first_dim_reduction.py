"""
================================================================================
Script 1 - Data ingestion and first-step dimension reduction
================================================================================

WHAT THIS SCRIPT DOES
---------------------
1. Loads the raw ISTAT AVQ 2021 micro-data file (data/AVQ_Microdati_2021.csv).
   The AVQ ("Aspetti della vita quotidiana") survey is the Italian yearly
   survey on daily-life aspects; this study uses the 2021 edition. The unit of
   analysis is the individual/respondent.

2. Keeps only the rows for which BOTH target variables are valid:
     - SPRENER : self-reported attention to not wasting electricity
                 (ordinal 1-4; 1 = high attention, 4 = low attention)
     - SPRACQUA: self-reported attention to not wasting water
                 (ordinal 1-4; same scale)
   The targets are left ORDINAL (they are not re-encoded in this script).

3. Recovers the checklist (multi-response) questions. In AVQ these are stored
   with one column per possible answer, whose value is the answer item-code if
   selected and NaN otherwise (e.g. CAMCLI is 3 = "climate change" when the
   respondent chose it). Every non-NaN cell in such a column is identical, so
   naive pipelines mistake these columns for "constant" or "mostly missing"
   and delete them. Here the batteries relevant to the study goals are
   binarised (selected -> 1, not selected -> 0) BEFORE the generic filters.

4. Reduces the number of predictor columns with generic statistical heuristics
   (missingness, variance, collinearity, correlation with the targets), but the
   four semantically meaningful variable blocks aligned with the study's three
   research goals are PROTECTED: they are exempt from the variance / target-
   correlation / collinearity cuts so that whole scientific categories are not
   erased by statistical shortcuts.

5. Imputes the remaining NaN cells with the column median and saves the tables
   used by every subsequent script.

RESEARCH GOALS AND VARIABLE BLOCKS
----------------------------------
  Goal 1: how human characteristics relate to environmental attitudes and
          eco behaviours.
  Goal 2: how socioeconomic, motivational and preference factors explain
          variation in electricity and water consumption.
  Goal 3: which spatial / living-area factors drive these behaviours.

  The four protected blocks are:

  BLOCK_SOCIO         - individual and family socioeconomic characteristics
                        (sex, education, employment, income class, family type,
                        marital status, household size).
  BLOCK_ATTITUDES     - the 15-item "which environmental problems worry you"
                        checklist (climate change, air pollution, waste, ...).
  BLOCK_CONSUMPTION   - energy/water behaviours: waste type brought to eco
                        stations, dissatisfaction with door-to-door waste
                        collection, water supply source, tap-water drinking,
                        commuting mode and distance, the 8-item sustainable
                        behaviour battery, and dwelling energy services
                        (gas supply, heating, electricity service satisfaction).
  BLOCK_SPATIAL       - living-area conditions: region, geographic zone,
                        perceived neighbourhood problems (traffic, noise,
                        crime, pollution, ...) and local waste services.

PIPELINE STEPS (in order)
-------------------------
  A. Load data, drop target-missing rows, plot target distributions.
  B. Replace blank strings with NaN.
  C. Binarise the relevant checklist batteries (see BINARISE_BATTERIES below).
  D. Coerce object columns to numeric where >= 90% of cells parse as numbers.
  E. Drop columns with > 60% missing values (applies to all columns).
  F. Drop constant columns (applies to all columns).
  G. Drop COEFIN (reconstructed financial coefficient, not informative here).
  H. Drop low-variance columns (variance < 0.01) - NON-block columns only.
  I. Drop highly collinear columns (|Pearson| > 0.90) - NON-block only.
  J. Drop columns weakly correlated with both targets (< 0.02) - NON-block only.
  K. Drop SF9, SF11, SF13, SF14, SF15 (the 5 SF-36 items that are combined
     into the synthetic index MH; keeping them duplicates MH's content).
  L. Impute the remaining NaN cells with the per-column median.
  M. Save the datasets.

KNOWN ISSUES / CAVEATS (important for external users)
-----------------------------------------------------
1. Only the batteries listed in BINARISE_BATTERIES are recovered. Many OTHER
   checklist batteries exist in the raw file (politics, internet use, health
   access, online purchases, ...). They are deliberately NOT binarised here
   because they are not central to the three research goals; they are left in
   their raw encoding and therefore removed by the constant/missingness
   filters. Add batteries to BINARISE_BATTERIES if they become relevant.

2. Median imputation is not ideal for ordinal/categorical codes: it can
   produce fractional values (e.g. 2.5 for a 1-4 Likert item) that correspond
   to no real category. Binarised 0/1 columns are unaffected (median of a
   0/1 flag with majority 0 is 0).

3. Target-correlation filtering (step J) is statistically fragile: the targets
   are heavily imbalanced (about 68% in class 1), so |Pearson correlation|
   with them is structurally tiny for almost every feature. This is exactly why
   the block columns are exempted from this step.

4. Low-variance filtering (step H) removes rare binary events (< ~0.5% of
   rows) among NON-block columns. Rare-but-meaningful indicators inside the
   protected blocks (e.g. METRO at 1.1%) are kept.

5. Deprecation warnings under newer pandas: df.replace(r"^\s*$", np.nan, ...)
   raises a FutureWarning on pandas >= 2.x because of the silent down-casting
   change. The behaviour is still correct on the pinned environment
   (pandas 1.5.2) used for this project.

OUTPUTS (relative to the repository root)
-----------------------------------------
  - data/preprocessed_data.csv            : imputed, complete predictor table
  - data/preprocessed_data_with_nan.csv   : same table BEFORE imputation
  - data/tar_sprener.csv                  : SPRENER target column
  - data/tar_spracqua.csv                 : SPRACQUA target column
  - exploratory_analysis/target_vars_distribution_original.png
  - exploratory_analysis/to_impute_correlation_with_target_vars.png

HOW TO RUN
----------
  python 1_data_ingestion_first_dim_reduction.py   (from the repo root)
  Run with the "forecasting" conda environment (Python 3.10 / pandas 1.5.2).
================================================================================
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# "sns.set()" applies seaborn's default style to matplotlib figures so that the
# plots produced in this script have a consistent, publication-friendly look.
sns.set()

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Every threshold used by the reduction steps is defined here so that an
# external reader can tune the pipeline in one place.
MISSING_COL_THRESHOLD = 0.60        # step E: drop columns with >60% NaN
LOW_VARIANCE_THRESHOLD = 0.01       # step H: drop columns with variance <0.01
HIGH_CORR_THRESHOLD = 0.90          # step I: drop columns with |r| >0.90
TARGET_CORR_THRESHOLD = 0.02        # step J: drop columns with |r|<0.02 to both targets
NUMERIC_PARSE_RATIO = 0.90          # step D: coerce to numeric if >=90% of
                                    #         non-null cells parse as numbers

# ------------------------------------------------------------------------------
# Checklist batteries to binarise
# ------------------------------------------------------------------------------
# AVQ encodes multi-response ("checklist") questions as one column per answer,
# storing the answer item-code if the respondent selected it and NaN otherwise.
# We convert the batteries that are relevant to the three research goals into
# 0/1 indicators (selected -> 1, not selected -> 0). Each entry is (battery
# name, list of column names); only columns actually present are touched.
BINARISE_BATTERIES = {
    # Environmental-concern checklist (Goal 1 - attitudes).
    # "Problemi ambientali che la preoccupano maggiormente".
    "concern_attitudes": [
        "SERRA", "ESTINZ", "CAMCLI", "SMARIF", "AMRUM", "IARIA", "INQSU",
        "INQFIU", "DISDR", "CATASTR", "FORES2", "INQELET2", "PAESAG2",
        "ESRISO2", "ALTAMB2",
    ],
    # Types of waste brought to ecological stations (Goal 2 - recycling).
    # "Tipo di rifiuti portato nelle stazioni ecologiche".
    "waste_to_station": [
        "ECOCAR", "ECOVET", "ECOPLA", "ECOPNE", "ECOMET", "ECOLEG", "ECOFAR",
        "ECOBAT", "ECOING", "ECOINE", "ECOELE", "ECOOLI", "ECOINF", "ECOPOT",
        "ECOTESS", "ECOALT",
    ],
    # Reasons for dissatisfaction with door-to-door waste collection
    # (Goal 2/3 - waste service quality).
    "waste_service_dissat": [
        "NOSPO1", "NOSPO2", "NOSPO3", "NOSPO4", "NOSPO5", "NOSPO6", "NOSPO7",
        "NOSPO8",
    ],
    # Water supply source of the dwelling (Goal 2 - water infrastructure).
    "water_supply": ["REACQ1", "REACQ2", "REACQ3", "REACQ4", "REACQ5"],
    # Whether the family drinks tap water, and if not why (Goal 2 - water).
    "tap_water": ["SIACQ", "ACQBRU", "NOACQ"],
    # Commuting transport modes used (Goal 2 - transport behaviour).
    "transport_modes": [
        "TRENO", "TRAM", "METRO", "BUS", "COR", "PAZSC", "AUTOC", "AUTOP",
        "MOTO", "BICI", "ALMEZ",
    ],
    # Commuting distance (Goal 3 - spatial dimension of commuting).
    "commute_distance": ["STCOM", "STPROV", "STREG", "ALTREG", "ESTERO"],
}

# ------------------------------------------------------------------------------
# Protected variable blocks (aligned with the three research goals)
# ------------------------------------------------------------------------------
# Columns belonging to any block are EXEMPT from the variance / collinearity /
# target-correlation filters (steps H, I, J). They still pass through the
# missingness (E) and constant (F) filters, which only remove truly useless
# columns. Blocks may overlap; a column protected by any block is kept.
BLOCK_SOCIO = [        # Goal 1 - human characteristics
    "SESSO", "ISTRMi", "CONDMi", "REDPRMi", "TIPFA2Mi", "STCIVMi", "NCOMP",
]

BLOCK_ATTITUDES = BINARISE_BATTERIES["concern_attitudes"]  # Goal 1 - attitudes

BLOCK_CONSUMPTION = (  # Goal 2 - electricity/water consumption & behaviours
    BINARISE_BATTERIES["waste_to_station"]
    + BINARISE_BATTERIES["waste_service_dissat"]
    + BINARISE_BATTERIES["water_supply"]
    + BINARISE_BATTERIES["tap_water"]
    + BINARISE_BATTERIES["transport_modes"]
    + BINARISE_BATTERIES["commute_distance"]
    # The 8 sustainable-behaviour items (the two SPR* targets are handled
    # separately, see below).
    + ["ETICHET", "BIOLOG", "ALOCAL", "GCARTE", "DOPFIL", "ARUMOR", "TRASPO",
       "USAGETT"]
    # Dwelling energy services.
    + ["GAS5", "TRISC", "RISCAL", "SENELE"]
)

BLOCK_SPATIAL = [      # Goal 3 - living area and local services
    "REGMf", "RIPMf", "SPORCO", "TRAF", "INQAR", "RUMORE", "CRIM", "PARCH",
    "COLMP", "ODSGR", "ILLSTR", "CONPAV", "POAPO", "ECOSTAZ", "GODAB",
]

# ------------------------------------------------------------------------------
# A. Load the raw survey data
# ------------------------------------------------------------------------------
start_load = time.time()
print("\nLoading data...")
# The raw file is ~81 MB and contains ~19,827 household rows and 735 columns.
raw_df = pd.read_csv("data/AVQ_Microdati_2021.csv")
end_load = time.time()
print("Time to load data: {:.2f} minutes".format((end_load - start_load) / 60))

# ------------------------------------------------------------------------------
# Basic Info about the dataset
# ------------------------------------------------------------------------------
n_obs = raw_df.shape[0]
print(f"Number of observations: {n_obs}")
n_cols = raw_df.shape[1]
print(f"Number of columns: {n_cols}")

# Quick snapshot for terminal readability
print(f"Shape (rows, cols): {raw_df.shape}")

# Memory footprint
mem_mb = raw_df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Approx. memory usage: {mem_mb:.2f} MB")

# Missingness overview
total_missing = int(raw_df.isna().sum().sum())
missing_pct = (total_missing / (n_obs * n_cols)) * 100 if n_obs and n_cols else 0
print(f"Total missing values: {total_missing} ({missing_pct:.2f}% of all cells)")

# Dtype distribution
dtype_counts = raw_df.dtypes.value_counts()
print("Column types:")
for dtype_name, count in dtype_counts.items():
    print(f"  - {dtype_name}: {count}")

# Duplicate rows
dup_rows = int(raw_df.duplicated().sum())
print(f"Duplicate rows: {dup_rows}")

# First columns preview (helps users orient quickly)
preview_cols = raw_df.columns[:15].tolist()
print(f"First {len(preview_cols)} columns: {preview_cols}")

# Top missing columns (if any)
missing_by_col = raw_df.isna().sum().sort_values(ascending=False)
top_missing = missing_by_col[missing_by_col > 0].head(10)
if not top_missing.empty:
    print("Top columns by missing values:")
    for col, miss in top_missing.items():
        print(f"  - {col}: {int(miss)}")
else:
    print("No missing values detected in any column.")

    

# ------------------------------------------------------------------------------
# Target variables
# ------------------------------------------------------------------------------
# The two behavioural targets of the study, both measured on an ordinal scale
# 1..4 (1 = "always/often" careful, 4 = "never" careful):
#   - "SPRENER"  (fare attenzione a non sprecare energia elettrica)
#   - "SPRACQUA" (fare attenzione a non sprecare acqua)
print("\nTarget variables...")

target_vars = ["SPRENER", "SPRACQUA"]
# NOTE: the call below only computes summary statistics; the result is not
# printed or stored, so it has no effect. Kept to preserve the original code.
raw_df[target_vars].describe()

# Check for missing values (they correspond to survey non-response).
print("Missing values in targets:")
print(raw_df[target_vars].isnull().sum())

# Keep only the observations with a valid value on BOTH targets.
df = raw_df.dropna(subset=target_vars).copy()

# The targets are ordinal categories: plot their (very skewed) distributions.
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i, var in enumerate(target_vars):
    df[var].value_counts().plot(kind="bar", ax=ax[i], title=var)
plt.tight_layout()
plt.savefig("exploratory_analysis/target_vars_distribution_original.png")

# Remove the targets from the feature table and keep them aside as separate
# Series. "pop" extracts the column and deletes it from df in one step.
# The targets stay ordinal (codes 1-4) - they are not re-encoded here.
sprener = df.pop("SPRENER")
spracqua = df.pop("SPRACQUA")

# ------------------------------------------------------------------------------
# Reduce number of independent variables
# ------------------------------------------------------------------------------
print("\nReducing number of independent variables...")

# Number of predictor columns we start with (used later for the summary log).
initial_n_cols = df.shape[1]

# Many AVQ columns store "blank" cells (spaces) to mean "not answered".
# Normalise those blanks to proper NaN so that every later step treats them
# as missing values instead of valid data.
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

# ------------------------------------------------------------------------------
# C. Binarise the relevant checklist batteries
# ------------------------------------------------------------------------------
# Turn the "item-code if mentioned, else NaN" encoding into proper 0/1
# indicators. This must happen BEFORE the missingness/constant filters,
# otherwise these columns would be dropped as "constant" or ">60% missing".
n_binarised = 0
for battery_name, columns in BINARISE_BATTERIES.items():
    present = [c for c in columns if c in df.columns]
    for col in present:
        df[col] = df[col].notna().astype(int)
    if present:
        print(f"Binarised battery '{battery_name}': {len(present)} columns "
              f"({', '.join(present)}).")
        n_binarised += len(present)
print(f"\nTotal checklist columns binarised: {n_binarised}.")

# ------------------------------------------------------------------------------
# D. Coerce object columns to numeric when possible
# ------------------------------------------------------------------------------
# Some columns are read as strings (dtype "object") although they only contain
# numeric survey codes. Convert them if the overwhelming majority of the
# non-null cells can be parsed as numbers; otherwise the column is genuinely
# textual and cannot be used by the ML pipeline.
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
forced_numeric_cols = []      # columns that were successfully converted
still_non_numeric_cols = []   # columns that stay textual
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
    print(f"\nConverted {len(forced_numeric_cols)} object columns to numeric.")

# Drop genuinely non-numeric columns to keep the pipeline simple for the ML
# scripts (no one-hot encoding of text is performed in this project).
if still_non_numeric_cols:
    print(f"\nDropping {len(still_non_numeric_cols)} non-numeric columns: {still_non_numeric_cols}")
    df.drop(columns=still_non_numeric_cols, inplace=True)

# ------------------------------------------------------------------------------
# E. Remove variables with very high missingness
# ------------------------------------------------------------------------------
# A column with more than 60% of its cells missing carries too little signal to
# impute reliably and would mostly propagate the imputation value, so drop it.
# This applies to ALL columns, including the protected blocks: binarised
# batteries have 0% missing here, so the blocks are unaffected in practice.
missing_ratio = df.isnull().sum() / len(df)
high_missing_cols = missing_ratio[missing_ratio > MISSING_COL_THRESHOLD].index.tolist()
if high_missing_cols:
    print(f"\nDropping {len(high_missing_cols)} columns with > {MISSING_COL_THRESHOLD:.0%} missing values.")
    df.drop(columns=high_missing_cols, inplace=True)

# Sanity log: the worst remaining column should now have a low missingness.
missing_ratio_after = df.isnull().sum() / len(df)
print("Max missing ratio after filtering: {:.2%}".format(missing_ratio_after.max()))

# ------------------------------------------------------------------------------
# F. Identify and drop constant variables
# ------------------------------------------------------------------------------
# A variable that takes the same value for every row cannot help separate the
# target classes, so it is useless for prediction. Applied to ALL columns:
# after binarisation, the checklist batteries have both 0s and 1s, so they are
# not constant (unless an answer was never selected - then it is dropped).
constant_vars = df.columns[df.nunique(dropna=True) <= 1]
if len(constant_vars) > 0:
    print(f"\nDropping {len(constant_vars)} constant columns.")
    df.drop(constant_vars, axis=1, inplace=True)

# ------------------------------------------------------------------------------
# G. Drop the reconstructed financial coefficient, if present
# ------------------------------------------------------------------------------
# COEFIN is a synthetic coefficient attached to the survey sample design
# (household size x income proxy); it is not a substantive explanatory variable
# for this study, so it is removed whenever it survives the previous steps.
if "COEFIN" in df.columns:
    df.drop("COEFIN", axis=1, inplace=True)

# ------------------------------------------------------------------------------
# Compute the protected-column set (block membership).
# ------------------------------------------------------------------------------
protected_cols = list({
    col
    for block in (BLOCK_SOCIO, BLOCK_ATTITUDES, BLOCK_CONSUMPTION, BLOCK_SPATIAL)
    for col in block
    if col in df.columns
})
n_protected = len(protected_cols)
print(f"Protected block columns still present: {n_protected} "
      f"(SOCIO={len([c for c in BLOCK_SOCIO if c in df.columns])}, "
      f"ATTITUDES={len([c for c in BLOCK_ATTITUDES if c in df.columns])}, "
      f"CONSUMPTION={len([c for c in BLOCK_CONSUMPTION if c in df.columns])}, "
      f"SPATIAL={len([c for c in BLOCK_SPATIAL if c in df.columns])}).")

# ------------------------------------------------------------------------------
# H. Drop variables with very low variance (NON-block columns only)
# ------------------------------------------------------------------------------
# Variance is a rough proxy for "how much information a column carries".
# To compute it we must have no NaN, so we first fill with the column median
# (a purely provisional fill - it is NOT what is finally saved).
# Block columns are exempt so that rare-but-meaningful indicators inside the
# research-relevant batteries (e.g. METRO at ~1% of rows) are not erased.
temp_for_stats = df.fillna(df.median(numeric_only=True))
variance = temp_for_stats.var()
low_var_candidates = variance[variance < LOW_VARIANCE_THRESHOLD].index.tolist()
low_var_cols = [c for c in low_var_candidates if c not in protected_cols]
if low_var_cols:
    print(f"\nDropping {len(low_var_cols)} low-variance columns (< {LOW_VARIANCE_THRESHOLD}).")
    df.drop(columns=low_var_cols, inplace=True)

# ------------------------------------------------------------------------------
# I. Remove variables that are almost perfectly collinear (NON-block only)
# ------------------------------------------------------------------------------
# Highly redundant columns (|correlation| > 0.90) add little new information and
# can destabilise regression / tree-ensemble models. Compute the absolute
# Pearson correlation matrix (median-filled again, purely for the check) and
# inspect only the upper triangle (above the diagonal) to avoid double counting.
corr_input = df.fillna(df.median(numeric_only=True))
corr_matrix = corr_input.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# A column is dropped if ANY entry in its upper-triangle row exceeds 0.90.
# Block columns are exempt so that correlated-but-meaningful pairs inside a
# block (e.g. AUTOC/AUTOP car commuting) are both preserved.
high_corr_candidates = [column for column in upper.columns
                        if any(upper[column] > HIGH_CORR_THRESHOLD)]
high_corr_cols = [c for c in high_corr_candidates if c not in protected_cols]
if high_corr_cols:
    print(f"\nDropping {len(high_corr_cols)} highly correlated columns (>{HIGH_CORR_THRESHOLD}).")
    df.drop(columns=high_corr_cols, inplace=True)

# ------------------------------------------------------------------------------
# J. Remove variables with low correlation with the target variables (NON-block)
# ------------------------------------------------------------------------------
# Keep only columns that show at least a weak linear association with one of
# the two targets. Correlation is computed on the median-filled table.
# Block columns are exempt: they are selected for their scientific relevance to
# the research goals, not for their marginal Pearson correlation with an
# imbalanced ordinal target.
corr_input = df.fillna(df.median(numeric_only=True))
corr_sprener = pd.Series(corr_input.corrwith(sprener))
corr_spracqua = pd.Series(corr_input.corrwith(spracqua))

# Collect the set of columns whose |correlation| with SPRENER is < threshold.
low_corr_sprener = set()
for col, value in corr_sprener.items():
    try:
        if abs(float(value)) < TARGET_CORR_THRESHOLD:
            low_corr_sprener.add(col)
    except (TypeError, ValueError):
        # Non-finite or non-convertible correlation values are ignored; the
        # column is not flagged as "low correlation".
        continue

# Same procedure for SPRACQUA.
low_corr_spracqua = set()
for col, value in corr_spracqua.items():
    try:
        if abs(float(value)) < TARGET_CORR_THRESHOLD:
            low_corr_spracqua.add(col)
    except (TypeError, ValueError):
        continue

# Drop only the columns that are weak for BOTH targets (intersection),
# excluding the protected blocks.
to_drop_low_target_corr = [
    c for c in (low_corr_sprener & low_corr_spracqua) if c not in protected_cols
]
if to_drop_low_target_corr:
    print(
        f"Dropping {len(to_drop_low_target_corr)} columns with weak correlation "
        f"(< {TARGET_CORR_THRESHOLD}) for both targets."
    )
    df.drop(columns=to_drop_low_target_corr, inplace=True)

# ------------------------------------------------------------------------------
# K. Drop the SF-36 items that are combined into the MH index
# ------------------------------------------------------------------------------
# MH is a synthetic "mental health / psychological distress" index built by
# ISTAT from five SF-36 items: SF9, SF11, SF13, SF14 and SF15 (how often in the
# last 4 weeks the respondent felt calm, discouraged/sad, agitated, full of
# morale, happy). Keeping both the items and their aggregate index MH would
# insert near-perfect redundancy, so the items are removed.
# "errors='ignore'" makes the drop a no-op if some are already gone.
df.drop(["SF9", "SF11", "SF13", "SF14", "SF15"], axis=1, inplace=True, errors="ignore")

# ------------------------------------------------------------------------------
# L. Inspect and impute the remaining NaN cells
# ------------------------------------------------------------------------------
# List the columns that still contain missing values after all the filters.
nan_cols = df.columns[df.isna().any()]
print("Columns with NaN before imputation: {}".format(list(nan_cols)))

# Plot how each of these "to-impute" columns correlates with the two targets,
# as a diagnostic of whether they are worth keeping at all.
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

# ------------------------------------------------------------------------------
# M. Save the preprocessed data
# ------------------------------------------------------------------------------
# Summary of the reduced feature table.
print("\nHere we have {} observations, {} independent variables, which {} of them presents nan values."
    .format(df.shape[0], df.shape[1], len(nan_cols)))

# Save a copy WITH the NaN cells still present, so that other scripts can choose
# a different imputation strategy if they want to.
df_nan = df.copy()
df_nan.to_csv("data/preprocessed_data_with_nan.csv", index=False)

# Impute the remaining NaN values column-wise with the median. This preserves
# every surviving column and row (no further deletions after this point).
medians = df.median(numeric_only=True)
df = df.fillna(medians)

# Final safety check: downstream scripts assume a complete (NaN-free) table.
if df.isna().sum().sum() > 0:
    raise ValueError("NaN values still present after imputation. Please inspect preprocessing steps.")

# Save the final imputed feature table.
df.to_csv("data/preprocessed_data_imputed.csv", index=False)
print("After imputation dataset has {} observations and {} features.".format(df.shape[0], df.shape[1]))
print("Feature reduction: {} -> {} columns.".format(initial_n_cols, df.shape[1]))

# Save the target variables. Because no rows were removed after the initial
# dropna, df keeps the same row order as sprener/spracqua, so reindexing with
# df.index guarantees the pairing is preserved.
sprener[df.index].to_csv("data/tar_sprener.csv", index=False)
spracqua[df.index].to_csv("data/tar_spracqua.csv", index=False)

print("\nPreprocessed datasets saved.")
