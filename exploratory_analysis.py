import pandas as pd
import time
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import numpy as np

# Read the data
start_load = time.time()
raw_df = pd.read_csv("data/AVQ_Microdati_2021.csv")
end_load = time.time()
print("Time to load data: {:.2f} minutes".format((end_load - start_load) / 60))

raw_df.head()
n_obs = raw_df.shape[0]

#%% Target variables
# The target variables are:
# - "SPRENER" (fare attenzione a non sprecare energia elettrica)
# - "SPRACQUA" (fare attenzione a non sprecare acqua)

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
plt.show()

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
# get the variable with max variance
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




#%% DIMENSIONALITY REDUCTION

# select numerical features
numerical_features = df.select_dtypes(include=["int64", "float64"])


#%% CLUSTERING
from sklearn.cluster import KMeans
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(numerical_features)

# Add cluster labels to the original dataset
data['cluster_label'] = kmeans.labels_

# Now, 'data' contains an additional column 'cluster_label' indicating the cluster assignment for each sample

# To see the cluster assignments and inspect the clusters:
print(data['cluster_label'].value_counts())

# You can further analyze the clusters to identify groups of variables that are redundant
# For example, you can calculate the mean/variance of variables within each cluster and compare them
