import pandas as pd
import time
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
sns.set()

# Read the data
start_load = time.time()
print("\nLoading data...")
raw_df = pd.read_csv("data/AVQ_Microdati_2021.csv")
end_load = time.time()
print("Time to load data: {:.2f} minutes".format((end_load - start_load) / 60))

raw_df.head()
n_obs = raw_df.shape[0]

#%% Target variables
# The target variables are:
# - "SPRENER" (fare attenzione a non sprecare energia elettrica)
# - "SPRACQUA" (fare attenzione a non sprecare acqua)
print("\nTarget variables...")

target_vars = ["SPRENER", "SPRACQUA"]
raw_df[target_vars].describe()

# Check for missing values
raw_df[target_vars].isnull().sum()

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

# Remove variables with a high percentage of missing values, as they may not provide meaningful information. (50%)
nulls = df.isnull().sum() / n_obs
nulls[nulls > 0.3]
# remove variables with more than 30% of missing values
df.drop(nulls[nulls > 0.3].index, axis=1, inplace=True)

# check again for missing values after initial cleaning
nulls_after = df.isnull().sum() / n_obs
nulls_after[nulls_after > 0.3]

# Identify constant variables
constant_vars = df.columns[df.nunique() == 1]
# Variables with constant values across all samples won't contribute to the model. Remove them.
df.drop(constant_vars, axis=1, inplace=True)

# Look at the variance of each variable.
variance = df.var()
# OUTLIER: get the variable with max variance
max_var = variance[variance == variance.max()]
# the var is COEFIN and it's not important for the analysis so can be removed
df.drop("COEFIN", axis=1, inplace=True)

# Variables with low variance (close to zero) are likely to be constant and can be removed
threshold = 0.1
low_var = variance[variance < threshold]
# remove variables with low variance
df.drop(low_var.index, axis=1, inplace=True)

# Remove variables with high correlation
corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
# remove variables with high correlation
df.drop(to_drop, axis=1, inplace=True)

# Remove variables with low correlation with target variables
# compute correlation between target variables and independent variables
corr_sprener = df.corrwith(sprener)
corr_spracqua = df.corrwith(spracqua)
threshold_corr = 0.05
to_drop_sprener = corr_sprener[abs(corr_sprener) < threshold_corr].index
to_drop_spracqua = corr_spracqua[abs(corr_spracqua) < threshold_corr].index
to_drop = list(set(to_drop_sprener) | set(to_drop_spracqua))
# remove variables with low correlation with target variables
df.drop(to_drop, axis=1, inplace=True)

# check for columns who contain empty strings
to_drop = df.columns[df.isin([" "]).any()]
df.drop(to_drop, axis=1, inplace=True)

# NB: by looking at the dataframe visually i can see there are still some columns with empty strings
# selecting all the columns with type object gives those values
empty_df = df.select_dtypes(include=["object"])
# remove those columns
df.drop(empty_df.columns, axis=1, inplace=True)

# remove columns sf9 sf11 sf13 sf14 sf15 which are correlated with variable MH
df.drop(["SF9", "SF11", "SF13", "SF14", "SF15"], axis=1, inplace=True, errors="ignore")

# columns which contains nan
to_drop = df.columns[df.isna().any()]
print("Columns with nan: {}".format(to_drop))

# are these columns important?
# let's see the correlation with the target variables
corr_sprener_to_drop = df[to_drop].corrwith(sprener)
corr_spracqua_to_drop = df[to_drop].corrwith(spracqua)
# plot the correlation
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
corr_sprener_to_drop.plot(kind="bar", ax=ax[0], title="SPRENER")
corr_spracqua_to_drop.plot(kind="bar", ax=ax[1], title="SPRACQUA")
plt.ylabel("Correlation")
plt.xlabel("Variables with NaN to be dropped")
plt.tight_layout()
plt.savefig("exploratory_analysis/to_drop_correlation_with_target_vars.png")


#%% SAVE PREPROCESSED DATA
print("\nHere we have {} observations, {} independent variables, which {} of them presents nan values."
      .format(df.shape[0], df.shape[1], len(to_drop)))

df_nan = df.copy()
df_nan.to_csv("data/preprocessed_data_with_nan.csv", index=False)

# remove rows with nan
df.dropna(axis=0, inplace=True)

# now we remain with 12 indipendent variables
df.to_csv("data/preprocessed_data.csv", index=False)
print("After dropping NaN values dataset has now {} observations.".format(df.shape[0]))

# save target variables
sprener[df.index].to_csv("data/tar_sprener.csv", index=False)
spracqua[df.index].to_csv("data/tar_spracqua.csv", index=False)

print("\nDatasets saved.")

#%% POST PROCESSING ANALYSIS

# correlation matrix
print("\nPlotting correlation matrix...")
# plot correlation matrix
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
plt.tight_layout()
plt.savefig("exploratory_analysis/correlation_matrix_preprocessed.png")

# class distribution of target variables
print("\nPlotting class distribution of target variables...")
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sprener[df.index].value_counts().plot(kind="bar", ax=ax[0], title="SPRENER")
spracqua[df.index].value_counts().plot(kind="bar", ax=ax[1], title="SPRACQUA")
plt.tight_layout()
plt.savefig("exploratory_analysis/target_vars_distribution_preprocessed.png")

