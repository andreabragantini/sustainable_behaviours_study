"""Feature selection focused on SPRENER and SPRACQUA.

Pipeline goals:
1) Start from the preprocessed feature matrix produced by script 1.
2) Rank features by association with each target variable (ANOVA F-score).
3) Keep top features per target (settings at the top of the script) and merge the two sets.
4) Remove highly correlated duplicates to reduce redundancy.
5) Save a reduced dataset for downstream analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


# -----------------------------
# Configuration
# -----------------------------
INPUT_WITH_NAN = Path("data/preprocessed_data_with_nan.csv")
INPUT_NO_NAN = Path("data/preprocessed_data_imputed.csv")

TARGET_FILES = {
	"sprener": Path("data/tar_sprener.csv"),
	"spracqua": Path("data/tar_spracqua.csv"),
}

OUTPUT_FEATURES = Path("data/preprocessed_data_reduced.csv")
OUTPUT_REPORT = Path("exploratory_analysis/reduced_dataset_feature_selection_summary.csv")

MAX_FEATURES_PER_TARGET = 60
MISSING_COL_THRESHOLD = 0.50
HIGH_CORR_THRESHOLD = 0.90


def load_feature_matrix() -> pd.DataFrame:
	"""Load feature matrix; prefer file with no NaN already worked on."""

	if INPUT_NO_NAN.exists():
		print(f"Loading features from: {INPUT_NO_NAN}")
		return pd.read_csv(INPUT_NO_NAN)
	
	if INPUT_WITH_NAN.exists():
		print(f"Loading features from: {INPUT_WITH_NAN}")
		return pd.read_csv(INPUT_WITH_NAN)

	raise FileNotFoundError(
		"No feature file found. Run 1_data_ingestion_first_dim_reduction.py first."
	)


def load_targets() -> dict[str, pd.Series]:
	"""Load target vectors as 1D integer series."""
	targets: dict[str, pd.Series] = {}
	for name, path in TARGET_FILES.items():
		if not path.exists():
			raise FileNotFoundError(f"Missing target file: {path}")
		target_df = pd.read_csv(path)
		targets[name] = target_df.iloc[:, 0].astype(int)
	return targets


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
	"""Keep numeric usable columns and remove obvious low-information ones."""
	X = X.copy()

	# Try to recover numeric information from object columns.
	for col in X.columns:
		if X[col].dtype == "object":
			X[col] = pd.to_numeric(X[col], errors="coerce")

	X = pd.DataFrame(X.select_dtypes(include=np.number)).copy()

	# Drop columns with too many missing values.
	miss_ratio = X.isna().mean()
	keep_missing_cols = miss_ratio[miss_ratio <= MISSING_COL_THRESHOLD].index.tolist()
	X = pd.DataFrame(X.loc[:, keep_missing_cols]).copy()

	# Drop constant columns.
	nunique = X.nunique(dropna=True)
	X = X.loc[:, nunique > 1]

	# Median imputation for scoring and downstream compatibility.
	X = X.fillna(X.median(numeric_only=True))

	return X


def score_features_anova(X: pd.DataFrame, y: pd.Series) -> pd.Series:
	"""Return ANOVA F-score per feature (higher is better)."""
	f_vals, _ = f_classif(X, y)
	scores = pd.Series(f_vals, index=X.columns)
	scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
	return scores.sort_values(ascending=False)


def remove_highly_correlated(
	X: pd.DataFrame,
	ordered_features: list[str],
	threshold: float,
) -> list[str]:
	"""Greedy pruning: keep high-priority features, skip correlated duplicates."""
	corr = X[ordered_features].corr().abs()
	selected: list[str] = []

	for col in ordered_features:
		if not selected:
			selected.append(col)
			continue

		max_corr_with_selected = corr.loc[col, selected].max()
		if max_corr_with_selected <= threshold:
			selected.append(col)

	# print the removed correlated features for debugging
	removed_features = set(ordered_features) - set(selected)
	if removed_features:
		print(f"Removed {len(removed_features)} highly correlated features:")
		for f in removed_features:
			print(f"  - {f}")
	else:
		print("No highly correlated features were removed.")

	return selected


def main() -> None:
	print("Target-aware feature selection started...")

	# Load data and targets.
	X_raw = load_feature_matrix()
	targets = load_targets()
	print(f"\nInitial features: {X_raw.shape[1]}")


	# Basic row alignment checks.
	n_rows = len(X_raw)
	for name, y in targets.items():
		if len(y) != n_rows:
			raise ValueError(f"Target {name} has {len(y)} rows but X has {n_rows} rows.")

	# Prepare features: keep numeric columns, drop low-information ones, impute missing values.
	X = prepare_features(X_raw)
	print(f"Candidate features after cleaning: {X.shape[1]}")

	# ANOVA:Score by each target and keep top-k per target.
	print(f"\nScoring features by ANOVA F-score per target ...")
	print(f"Keeping at most {MAX_FEATURES_PER_TARGET} features per target.")
	target_scores: dict[str, pd.Series] = {}
	candidate_union: set[str] = set()
	for name, y in targets.items():
		scores = score_features_anova(X, y)
		target_scores[name] = scores
		top_features = scores.head(MAX_FEATURES_PER_TARGET).index.tolist()
		candidate_union.update(top_features)
		print(f"Top features kept for {name}: {len(top_features)}")

	candidates = list(candidate_union)
	if not candidates:
		raise RuntimeError("No candidate feature selected. Check preprocessing and targets.")

	# Build a combined priority score to decide which feature to keep first.
	combined_score = pd.Series(0.0, index=candidates)
	for score_series in target_scores.values():
		combined_score = combined_score.add(score_series.reindex(candidates).fillna(0.0), fill_value=0.0)

	# Remove highly correlated features to reduce redundancy.
	print(f"\nRemoving highly correlated features (threshold: {HIGH_CORR_THRESHOLD})...")	
	ordered_candidates = combined_score.sort_values(ascending=False).index.tolist()
	final_features = remove_highly_correlated(X, ordered_candidates, HIGH_CORR_THRESHOLD)

	# Save the final selected feature matrix and a report of the selection process.
	X_selected = X[final_features].copy()
	X_selected.to_csv(OUTPUT_FEATURES, index=False)

	# REPORT: save a summary of the feature selection process, including scores and selection status.
	report = pd.DataFrame({"feature": X.columns})
	report["f_score_sprener"] = target_scores["sprener"].reindex(report["feature"]).values
	report["f_score_spracqua"] = target_scores["spracqua"].reindex(report["feature"]).values
	report["combined_score"] = (
		report["f_score_sprener"].fillna(0.0) + report["f_score_spracqua"].fillna(0.0)
	)
	report["selected"] = report["feature"].isin(final_features)
	report = report.sort_values(["selected", "combined_score"], ascending=[False, False])
	OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
	report.to_csv(OUTPUT_REPORT, index=False)

	print("\nFeature selection completed.")
	print(f"Selected features: {len(final_features)}")
	print(f"Saved selected matrix to: {OUTPUT_FEATURES}")
	print(f"Saved selection report to: {OUTPUT_REPORT}")


if __name__ == "__main__":
	main()
