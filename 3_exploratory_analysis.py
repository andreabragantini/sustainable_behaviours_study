"""Compact exploratory analysis on the reduced dataset.

The script keeps plots manageable by focusing on a small set of top-ranked
features selected in the previous step.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


FEATURES_PATH = Path("data/preprocessed_data_imputed.csv")
TARGET_PATHS = {
	"sprener": Path("data/tar_sprener.csv"),
	"spracqua": Path("data/tar_spracqua.csv"),
}
FEATURE_REPORT_PATH = Path("exploratory_analysis/reduced_dataset_feature_selection_summary.csv")
OUTPUT_DIR = Path("exploratory_analysis")

TOP_FEATURES_PER_TARGET = 10
LOW_CARDINALITY_LIMIT = 10


def load_data() -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
	"""Load reduced features, targets, and feature-ranking report."""
	X = pd.read_csv(FEATURES_PATH)

	targets: dict[str, pd.Series] = {}
	for name, path in TARGET_PATHS.items():
		target_df = pd.read_csv(path)
		targets[name] = target_df.iloc[:, 0].astype(int)

	feature_report = pd.read_csv(FEATURE_REPORT_PATH)

	for name, y in targets.items():
		if len(y) != len(X):
			raise ValueError(f"Target {name} has {len(y)} rows, expected {len(X)}.")

	return X, targets, feature_report


def select_features_for_eda(feature_report: pd.DataFrame, top_per_target: int) -> list[str]:
	"""Pick a small set of strong features for readable plots."""
	report = feature_report.loc[feature_report["selected"]].copy()
	if report.empty:
		raise ValueError("No selected features found in feature_selection_summary.csv")

	chosen: list[str] = []
	for score_col in ["f_score_sprener", "f_score_spracqua"]:
		top_features = report.sort_values(score_col, ascending=False)["feature"].head(top_per_target)
		for feature in top_features:
			if feature not in chosen:
				chosen.append(feature)

	# Add strong shared features until we reach a plot size that scales with the
	# number of top features considered per target.
	max_univariate_features = max(4, 2 * top_per_target)
	if len(chosen) < max_univariate_features:
		combined_top = report.sort_values("combined_score", ascending=False)["feature"]
		for feature in combined_top:
			if feature not in chosen:
				chosen.append(feature)
			if len(chosen) >= max_univariate_features:
				break

	return chosen[:max_univariate_features]


def plot_target_distributions(targets: dict[str, pd.Series]) -> None:
	"""Plot class counts for each target variable."""
	fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=250)

	for ax, (name, values) in zip(axes, targets.items()):
		order = sorted(values.unique())
		sns.countplot(x=values, order=order, ax=ax, color="#4c78a8")
		ax.set_title(f"Distribution of {name.upper()}")
		ax.set_xlabel("Class")
		ax.set_ylabel("Count")

	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "eda_target_distributions.png")
	plt.close()


def plot_target_joint_distribution(targets: dict[str, pd.Series]) -> None:
	"""Show how the two target variables co-occur."""
	joint = pd.crosstab(targets["sprener"], targets["spracqua"], normalize="index")

	plt.figure(figsize=(6, 5), dpi=250)
	sns.heatmap(joint, annot=True, fmt=".2f", cmap="Blues")
	plt.title("SPRACQUA distribution within SPRENER classes")
	plt.xlabel("SPRACQUA")
	plt.ylabel("SPRENER")
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "eda_target_joint_heatmap.png")
	plt.close()


def plot_univariate_features(X: pd.DataFrame, features: list[str]) -> None:
	"""Plot one distribution per selected feature."""
	n_cols = 2
	n_rows = int(np.ceil(len(features) / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), dpi=250)
	axes = np.atleast_1d(axes).ravel()

	for ax, feature in zip(axes, features):
		series = X[feature]
		unique_count = series.nunique(dropna=True)

		if unique_count <= LOW_CARDINALITY_LIMIT:
			order = sorted(series.dropna().unique())
			sns.countplot(x=series, order=order, ax=ax, color="#72b7b2")
			ax.set_ylabel("Count")
		else:
			sns.histplot(x=series, bins=25, kde=True, ax=ax, color="#72b7b2")
			ax.set_ylabel("Frequency")

		ax.set_title(feature)
		ax.set_xlabel(feature)

	for ax in axes[len(features):]:
		ax.remove()

	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / "eda_univariate_selected_features.png")
	plt.close()


def plot_bivariate_boxplots(X: pd.DataFrame, targets: dict[str, pd.Series], features: list[str]) -> None:
	"""Compare feature distributions across classes of each target."""
	for target_name, y in targets.items():
		plot_df = X[features].copy()
		plot_df[target_name] = y.values

		n_cols = 2
		n_rows = int(np.ceil(len(features) / n_cols))
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), dpi=250)
		axes = np.atleast_1d(axes).ravel()

		for ax, feature in zip(axes, features):
			sns.boxplot(data=plot_df, x=target_name, y=feature, ax=ax, color="#f58518", fliersize=1)
			ax.set_title(f"{feature} vs {target_name.upper()}")
			ax.set_xlabel(target_name.upper())
			ax.set_ylabel(feature)

		for ax in axes[len(features):]:
			ax.remove()

		plt.tight_layout()
		plt.savefig(OUTPUT_DIR / f"eda_bivariate_boxplots_{target_name}.png")
		plt.close()


def plot_group_mean_heatmaps(X: pd.DataFrame, targets: dict[str, pd.Series], features: list[str]) -> None:
	"""Compact summary of how feature means move across target classes."""
	X_scaled = X[features].copy()
	X_scaled = (X_scaled - X_scaled.mean()) / X_scaled.std(ddof=0)
	X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

	for target_name, y in targets.items():
		mean_profile = X_scaled.assign(target=y.values).groupby("target").mean()

		plt.figure(figsize=(10, 4), dpi=250)
		sns.heatmap(mean_profile, cmap="coolwarm", center=0, annot=True, fmt=".2f")
		plt.title(f"Mean standardized feature profile by {target_name.upper()} class")
		plt.xlabel("Feature")
		plt.ylabel(target_name.upper())
		plt.tight_layout()
		plt.savefig(OUTPUT_DIR / f"eda_group_mean_heatmap_{target_name}.png")
		plt.close()


def main() -> None:
	print("Running exploratory analysis on reduced data...")
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	X, targets, feature_report = load_data()
	eda_features = select_features_for_eda(feature_report, TOP_FEATURES_PER_TARGET)
	bivariate_features = eda_features[:max(4, TOP_FEATURES_PER_TARGET)]

	print(f"Reduced dataset shape: {X.shape}")
	print(f"Features used for univariate plots: {eda_features}")
	print(f"Features used for bivariate plots: {bivariate_features}")

	plot_target_distributions(targets)
	plot_target_joint_distribution(targets)
	plot_univariate_features(X, eda_features)
	plot_bivariate_boxplots(X, targets, bivariate_features)
	plot_group_mean_heatmaps(X, targets, bivariate_features)

	print("Exploratory plots saved in exploratory_analysis/.")


if __name__ == "__main__":
	main()


# -----------------------------------------------------------------------------
# CONCLUSIONS FROM THE EXPLORATORY ANALYSIS
# -----------------------------------------------------------------------------
# 1) Both target variables are strongly imbalanced toward class 1.
#    Since:
#    - SPRENER = 1 means high attention to avoiding electricity waste
#    - SPRACQUA = 1 means high attention to avoiding water waste
#    the data suggest that most respondents self-report high attention to both
#    energy-saving and water-saving behaviors.
#
# 2) SPRENER and SPRACQUA are strongly positively associated.
#    The joint heatmap shows a clear diagonal pattern: respondents with low
#    waste-related attention in one target tend to have low attention in the
#    other as well, and the same holds for high-attention respondents.
#    This suggests that the two targets reflect a common latent
#    pro-environmental attitude.
#
# 3) Several selected features show a coherent gradient across both targets.
#    In particular, ALOCAL, ETICHET, BIOLOG, ARUMOR, and USAGETT move
#    systematically from target class 1 to target class 4.
#    This means that respondents who are less careful about saving energy and
#    water also tend to report less sustainable behavior in related domains.
#
# 4) Interpreting the selected features using the survey documentation:
#    - ALOCAL  -> buying local food/products
#    - ETICHET -> reading food-product labels
#    - BIOLOG  -> buying organic food/products
#    - ARUMOR  -> avoiding noisy driving behaviors
#    - USAGETT -> using disposable products
#    the overall pattern is consistent with a broader sustainability profile:
#    attention to energy and water saving is associated with attention to other
#    sustainable consumption and behavior choices.
#
# 5) QTSALE shows a weaker and partly different pattern compared with the other
#    variables. It may be capturing a dietary/health-related dimension rather
#    than the same environmental-behavior dimension captured by SPRENER and
#    SPRACQUA.
#
# 6) Overall conclusion:
#    SPRENER and SPRACQUA do not appear to be isolated habits. They are closely
#    related to one another and also associated with a broader set of everyday
#    sustainable behaviors. This supports using them as meaningful target
#    variables for the subsequent correlation and modelling analyses.
