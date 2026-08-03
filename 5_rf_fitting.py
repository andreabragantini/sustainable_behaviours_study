"""
Study-oriented random forest analysis for sustainable behaviours.

This script is not meant to build the best predictive model.
Its goal is to use tree-based models as an exploratory tool to identify which survey
features are most strongly associated with two ordered targets:

- SPRACQUA: attention to water-saving behaviours
- SPRENER: attention to energy-saving behaviours

Target coding:
- 1 = high care / high attention
- 4 = no care / no attention

Why use Random Forest here?
- It handles many tabular predictors with nonlinear relations.
- It produces a feature-importance ranking that is useful for screening variables.
- It is a good exploratory step before more interpretable follow-up analyses.

Important limitation:
- Feature importance is not a causal effect.
- Feature importance does not tell the direction of the relationship.
- Because the target is ordinal, a later ordinal model is still recommended for interpretation.
"""

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from feature_descriptions import describe, render_table


# Keep the console output readable.
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used",
    category=UserWarning,
)
sns.set_theme(style="whitegrid")


DATA_PATH = Path("data/preprocessed_data_reduced.csv")
TARGETS = ["spracqua", "sprener"]
OUTPUT_DIR = Path("results") / "rf_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_SIZE = 0.30
RANDOM_STATE = 42
TOP_FEATURES_TO_SAVE = 20   
TOP_FEATURES_TO_PLOT = 10
PROFILE_FEATURES_TO_PLOT = 6
HEATMAP_FEATURES_TO_PLOT = 10


def load_features():
    """Load the full feature matrix and stop if missing values are still present."""
    X = pd.read_csv(DATA_PATH)
    if X.isna().any().any():
        raise ValueError("Input features contain NaN values. Please impute before model fitting.")
    return X


def load_target(target_name):
    """Load a single target from disk as a pandas Series."""
    target_path = Path("data") / "tar_{}.csv".format(target_name)
    y_df = pd.read_csv(target_path)

    if y_df.shape[1] != 1:
        raise ValueError(
            "Target file for {} must contain exactly one column. Found {} columns.".format(
                target_name, y_df.shape[1]
            )
        )

    y = y_df.iloc[:, 0].copy()
    y.name = target_name
    return y


def summarize_target(y, target_name):
    """Save the class distribution as a TXT table so the baseline is easy to read."""
    distribution = pd.DataFrame(
        {
            "count": y.value_counts().sort_index(),
            "proportion": y.value_counts(normalize=True).sort_index(),
        }
    )
    distribution.index.name = "target_level"
    distribution = distribution.reset_index()
    return distribution


def compute_metrics(y_true, y_pred):
    """Keep only a small set of metrics suited to imbalanced multiclass targets."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_models(X_train, X_test, y_train, y_test):
    """Fit a very small set of models for context.

    The predictive comparison is not the core result of the study.
    It is used only to check that the forest learns more than a trivial baseline.
    """
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "balanced_random_forest": BalancedRandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    fitted_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = model_name
        rows.append(metrics)
        fitted_models[model_name] = model

    comparison = pd.DataFrame(rows)
    comparison = comparison[
        ["model", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc"]
    ]
    return comparison, fitted_models


def save_feature_importance_outputs(importance_series, target_name):
    """Save pretty TXT tables of the full importance ranking and of the top
    features (with plain-language descriptions), plus two importance plots."""
    importance_series = importance_series.sort_values(ascending=False)

    importance_table = importance_series.reset_index()
    importance_table.columns = ["feature", "importance"]
    importance_table["rank"] = range(1, len(importance_table) + 1)
    importance_table["description"] = importance_table["feature"].map(describe)
    importance_table = importance_table[["rank", "feature", "description", "importance"]]
    with (OUTPUT_DIR / "feature_importance_{}.txt".format(target_name)).open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                importance_table,
                title="Feature importance ranking - {}".format(target_name),
                formats={"importance": ".6f"},
            )
            + "\n"
        )

    top_features = importance_table.head(TOP_FEATURES_TO_SAVE).reset_index(drop=True)
    with (OUTPUT_DIR / "top_features_{}.txt".format(target_name)).open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                top_features,
                title="Top {} feature importances - {}".format(TOP_FEATURES_TO_SAVE, target_name),
                formats={"importance": ".6f"},
            )
            + "\n"
        )

    full_plot = importance_series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 15))
    ax.barh(full_plot.index.tolist(), full_plot.values.tolist(), color="slateblue")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance - {}".format(target_name))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "features_importance_full_{}.png".format(target_name), dpi=300)
    plt.close(fig)

    top_plot = importance_series.head(TOP_FEATURES_TO_PLOT).sort_values(ascending=True)
    fig_h = max(8, int(TOP_FEATURES_TO_PLOT * 0.25))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(top_plot.index.tolist(), top_plot.values.tolist(), color="teal")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top {} feature importances - {}".format(TOP_FEATURES_TO_PLOT, target_name))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "features_importance_selected_{}.png".format(target_name), dpi=300)
    plt.close(fig)

    return top_features


def save_shared_importance_plot(shared):
    """Compare the importance of shared features across the two targets."""
    shared_plot = shared[shared["present_in_both_top20"]].copy()
    shared_plot = shared_plot.sort_values("mean_importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        shared_plot["importance_spracqua"],
        shared_plot["importance_sprener"],
        color="darkorange",
        alpha=0.8,
    )

    for _, row in shared_plot.head(12).iterrows():
        ax.text(
            row["importance_spracqua"] + 0.0002,
            row["importance_sprener"] + 0.0002,
            row["feature"],
            fontsize=8,
        )

    min_val = min(shared_plot["importance_spracqua"].min(), shared_plot["importance_sprener"].min())
    max_val = max(shared_plot["importance_spracqua"].max(), shared_plot["importance_sprener"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Importance for SPRACQUA")
    ax.set_ylabel("Importance for SPRENER")
    ax.set_title("Shared feature importance across water and energy care")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shared_feature_importance_scatter.png", dpi=300)
    plt.close(fig)


def save_descriptive_feature_plots(X, y, target_name, shared_features):
    """Create descriptive plots for the top shared features.

    These plots are easier to discuss than raw feature importance because they show how
    average feature values change across target levels 1, 2, 3, and 4.

    The features live on very different scales (e.g. 0-4 frequency items vs 0-10
    trust scales), so each feature is z-score standardised before averaging.  The
    group means are then comparable across features and can share a single colorbar
    in the heatmap without the larger-scale features dominating it.
    """
    selected = [feature for feature in shared_features if feature in X.columns]
    if not selected:
        return

    plot_df = pd.concat([y, X[selected]], axis=1)
    scaled = (plot_df[selected] - plot_df[selected].mean()) / plot_df[selected].std()
    scaled_df = pd.concat([y, scaled], axis=1)
    group_means = scaled_df.groupby(y.name)[selected].mean().sort_index()

    means_long = group_means.reset_index().melt(
        id_vars=y.name,
        var_name="feature",
        value_name="mean_standardized",
    )
    means_long["description"] = means_long["feature"].map(describe)
    with (OUTPUT_DIR / "descriptive_group_means_{}.txt".format(target_name)).open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                means_long,
                title="Mean standardized (z-score) shared-feature values by {} level - {}".format(
                    y.name, target_name
                ),
                formats={"mean_standardized": ".4f"},
            )
            + "\n"
        )

    # Line profile for the most important shared features.
    profile_features = selected[:PROFILE_FEATURES_TO_PLOT]
    profile_long = group_means[profile_features].reset_index().melt(
        id_vars=y.name,
        var_name="feature",
        value_name="mean_value",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=profile_long,
        x=y.name,
        y="mean_value",
        hue="feature",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.set_xlabel("Target level (1 = high care, 4 = no care)")
    ax.set_ylabel("Mean standardized (z-score) value")
    ax.set_title("Top shared behavioural profiles by {}".format(target_name))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shared_feature_profiles_{}.png".format(target_name), dpi=300)
    plt.close(fig)

    # Heatmap for a slightly wider set of shared features.
    heatmap_features = selected[:HEATMAP_FEATURES_TO_PLOT]
    heatmap_data = group_means[heatmap_features]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data.T, cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_xlabel("Target level (1 = high care, 4 = no care)")
    ax.set_ylabel("Feature")
    ax.set_title("Mean standardized shared-feature values across {} levels".format(target_name))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shared_feature_heatmap_{}.png".format(target_name), dpi=300)
    plt.close(fig)


def analyse_target(X, target_name):
    """Run the exploratory analysis for one target."""
    print("\n" + "=" * 70)
    print("Analysing target: {}".format(target_name))
    print("=" * 70)

    y = load_target(target_name)
    distribution = summarize_target(y, target_name)
    print("\nClass distribution:\n{}".format(distribution))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )

    comparison, fitted_models = evaluate_models(X_train, X_test, y_train, y_test)
    with (OUTPUT_DIR / "model_comparison_{}.txt".format(target_name)).open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                comparison,
                title="Model comparison - {}".format(target_name),
                formats={
                    "accuracy": ".4f",
                    "balanced_accuracy": ".4f",
                    "f1_macro": ".4f",
                    "f1_weighted": ".4f",
                    "mcc": ".4f",
                },
            )
            + "\n"
        )
    print("\nModel comparison:\n{}".format(comparison))

    # The balanced forest is used as the main screening model because classes are imbalanced.
    # After the comparison step, we refit it on the full sample to obtain a ranking that uses
    # all available respondents.
    exploratory_rf = BalancedRandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    exploratory_rf.fit(X, y)

    importance_series = pd.Series(exploratory_rf.feature_importances_, index=X.columns)
    top_features = save_feature_importance_outputs(importance_series, target_name)
    print("\nTop features for {}:\n{}".format(target_name, top_features))

    return {
        "target": target_name,
        "y": y,
        "distribution": distribution,
        "comparison": comparison,
        "importance": importance_series.sort_values(ascending=False),
        "top_features": top_features,
        "plain_rf": fitted_models["random_forest"],
        "balanced_rf": exploratory_rf,
    }


def build_shared_feature_summary(results_by_target):
    """Highlight the features that matter for both targets."""
    spracqua_top = results_by_target["spracqua"]["top_features"].copy()
    sprener_top = results_by_target["sprener"]["top_features"].copy()

    spracqua_top = spracqua_top.rename(
        columns={
            "rank": "rank_spracqua",
            "importance": "importance_spracqua",
            "description": "description_spracqua",
        }
    )
    sprener_top = sprener_top.rename(
        columns={
            "rank": "rank_sprener",
            "importance": "importance_sprener",
            "description": "description_sprener",
        }
    )

    shared = spracqua_top.merge(sprener_top, on="feature", how="outer")
    shared["description"] = shared["description_spracqua"].fillna(shared["description_sprener"])
    shared["present_in_both_top20"] = shared["rank_spracqua"].notna() & shared["rank_sprener"].notna()
    shared["mean_importance"] = shared[["importance_spracqua", "importance_sprener"]].mean(axis=1)
    shared = shared.sort_values(
        by=["present_in_both_top20", "mean_importance"],
        ascending=[False, False],
    )
    shared = shared[
        [
            "feature",
            "description",
            "rank_spracqua",
            "importance_spracqua",
            "rank_sprener",
            "importance_sprener",
            "present_in_both_top20",
            "mean_importance",
        ]
    ]
    with (OUTPUT_DIR / "shared_top_features.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                shared,
                title="Top features shared across water and energy care",
                formats={
                    "importance_spracqua": ".6f",
                    "importance_sprener": ".6f",
                    "mean_importance": ".6f",
                },
            )
            + "\n"
        )
    return shared


def main():
    start = time.time()
    X = load_features()
    results_by_target = {}

    for target_name in TARGETS:
        results_by_target[target_name] = analyse_target(X, target_name)

    shared = build_shared_feature_summary(results_by_target)
    save_shared_importance_plot(shared)

    shared_features = shared.loc[shared["present_in_both_top20"], "feature"].head(HEATMAP_FEATURES_TO_PLOT).tolist()
    for target_name in TARGETS:
        save_descriptive_feature_plots(
            X,
            results_by_target[target_name]["y"],
            target_name,
            shared_features,
        )

    print("\nShared top features across water and energy care:\n{}".format(shared.head(20)))
    print("\nAnalysis completed in {:.2f} minutes".format((time.time() - start) / 60))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 1) Procedure used in this script:
# ---------------------------------------------------------------------------
#    - For each target, the script first checks class imbalance and compares a dummy baseline,
#      a plain Random Forest, and a Balanced Random Forest.
#    - This predictive comparison is only a diagnostic step.
#    - The main analytical output is the feature-importance ranking from the Balanced RF.
#    - After analysing both targets, the script builds one shared table of top features.
#
# Why this procedure fits the study goal:
#    - The goal is to understand which behaviours and characteristics are associated with
#      carefulness toward water and energy saving.
#    - RF is useful here as a variable-screening device: it helps detect which predictors
#      matter most in separating low-care from high-care respondents.
#    - This is more useful for the study than maximizing predictive accuracy.
#
# ---------------------------------------------------------------------------
# 2) Main results found so far in this project:
# ---------------------------------------------------------------------------
#    - The two targets show a strong overlap in their top-ranked features.
#    - Variables such as USAGETT, ARUMOR, ALOCAL, ETICHET, BIOLOG, TRASPO,
#      PUNTIFI10, PUNTIFI3, ETAMi, and LIBFAM appear near the top for both targets. 
#      This suggests that water-saving care and energy-saving care belong to a broader
#      behavioural profile rather than being isolated attitudes.
#    - USAGETT is the one exception: unlike the other top behaviour items, it moves
#      against the good-behaviour gradient, while the two similarly-phrased negative
#      items (GCARTE, DOPFIL) correlate negatively with the targets as expected.
#
# How these outputs should be used in the thesis or report:
#    - Use top_features_<target>.txt to identify the most relevant correlates
#      (with plain-language descriptions of each feature).
#    - Use shared_top_features.txt to focus on behaviours that matter for both targets.
#    - Use shared_feature_profiles_<target>.png and shared_feature_heatmap_<target>.png
#      to discuss how the top shared variables change across target levels 1, 2, 3, and 4.
#      These plots use z-score standardised feature values so features with different
#      scales (0-4 vs 0-10 items) can be compared on the same axis/colorbar.
#    - The next best step is an ordinal and more interpretable follow-up model.
#
# ---------------------------------------------------------------------------
# 3) Cautions:
# ---------------------------------------------------------------------------
#    - Feature importance is associative, not causal.
#    - Correlated predictors can share or distort importance.
#    - RF importance does not tell whether a variable is positively or negatively related
#      to carefulness.
#    - For direction and effect size, use descriptive tables and an ordinal model afterward.
