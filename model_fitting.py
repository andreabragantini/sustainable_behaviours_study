import pandas as pd
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
#from resampling_functions import undersample_df, oversample_df
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Read the data
target = "sprener"
#target = "spracqua"
y = pd.read_csv("data/tar_{}.csv".format(target))
X = pd.read_csv("data/preprocessed_data.csv")



#%% MODEL FITTING: RANDOM FOREST
print("\nFitting Random Forest model...")
start = time.time()

# Split the data into train and test sets
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)

# Fit the model
rf_i = RandomForestClassifier(n_estimators=1000, random_state=42, verbose=1, n_jobs=-1)
#rf_i = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, verbose=1, n_jobs=-1)
rf_i.fit(X_train, y_train)

# predict on test set
y_pred = rf_i.predict(X_test)
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))

# classification report
print("\nClassification report:\n\n", classification_report(y_test, y_pred))

#%% fit Balanced Random Forest Classifier
print("\nFitting initial Balanced Random Forest Classifier...")
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)

# initial model
rf_b = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, verbose=1, n_jobs=-1)
rf_b.fit(X_train, y_train)

# predict the Test set results
y_pred = rf_b.predict(X_test)
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))

# classification report
print("\nClassification report:\n\n", classification_report(y_test, y_pred))

#%% Show features importance

rf_model = rf_b

# view the feature scores
features = X.columns
feature_scores = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

f_i = list(zip(features, rf_model.feature_importances_))
f_i.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(12, 15))

ax.barh([x[0] for x in f_i], [x[1] for x in f_i], align='center', color='purple')
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature Name')
ax.set_title('Random Forest Feature Importance - {}'.format(target))

# Add value labels to the bars
for i, v in enumerate([x[1] for x in f_i]):
    ax.text(v + 0.01, i - 0.1, str(round(v, 3)), color='black', fontsize=10)

plt.savefig('trained_models/features_importance_{}.png'.format(target))

#%% Build Random Forest model on selected features
print("\nFitting final Random Forest Classifier after feature selection...")

# select features with score > 0.01
selected_features = [x[0] for x in f_i if x[1] >= 0.01]

# set features and target
X_sel = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=test_size, stratify=y, random_state=69)

# fit a balanced RF
rf_model = BalancedRandomForestClassifier(n_estimators=1000, random_state=0, verbose=1, n_jobs=-1)
rf_model.fit(X_train, y_train)

# predict the Test set results
y_pred = rf_model.predict(X_test)
print('Reduced Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n\n', cm)

# classification report
print("Classification report:\n\n", classification_report(y_test, y_pred))