import pandas as pd
import time


# Read the data
start_load = time.time()
raw_df = pd.read_csv("data/AVQ_Microdati_2021.csv")
end_load = time.time()
print("Time to load data: {:.2f} minutes".format((end_load - start_load) / 60))

raw_df.head()
n_obs = raw_df.shape[0]

# Remove variables with a high percentage of missing values, as they may not provide meaningful information. (50%)
nulls = raw_df.isnull().sum() / n_obs
nulls[nulls > 0.5]

df = raw_df.dropna(axis=1, thresh=int(0.5 * n_obs)).copy()
df.shape
df.columns

# check again for missing values after initial cleaning
nulls_after = df.isnull().sum() / n_obs
nulls_after[nulls_after > 0.5]

# Identify constant variables
constant_vars = df.columns[df.nunique() == 1]
# Variables with constant values across all samples won't contribute to the model. Remove them.
df.drop(constant_vars, axis=1, inplace=True)

