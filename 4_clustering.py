"""Survey respondent clustering for tentative behavioral profiling.

This script clusters respondents, not variables.  The goal is to see whether the
survey contains a few coherent citizen profiles and then inspect how SPRENER and
SPRACQUA vary across those profiles.

Rework notes (vs the previous version):
  * The clustering now runs on the FULL 299-feature matrix
    (``data/preprocessed_data_imputed.csv``) instead of the ANOVA-reduced
    65-feature subset, so socio-demographic, attitude and spatial information
    can drive the profiles.
  * A bootstrap stability check (average adjusted Rand index over resamples)
    is added next to the classical internal validation metrics, so the choice
    of ``k`` does not rest on a single weak silhouette value.
  * Feature codes in every output table are expanded with plain-language
    descriptions (see ``feature_descriptions.py``).
  * Result tables are written as aligned plain-text files; raw membership data
    stays in CSV.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from feature_descriptions import describe, render_table


RANDOM_STATE = 42
FEATURES_PATH = Path("data/preprocessed_data_imputed.csv")
TARGET_PATHS = {
    "sprener": Path("data/tar_sprener.csv"),
    "spracqua": Path("data/tar_spracqua.csv"),
}
OUTPUT_DIR = Path("results") / "clustering"
K_RANGE = range(3, 9)
N_INIT = 10
TOP_PROFILE_FEATURES = 12
N_BOOTSTRAP = 12
BOOTSTRAP_FRACTION = 0.7


sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load features and both target vectors."""
    X = pd.read_csv(FEATURES_PATH)

    targets: dict[str, pd.Series] = {}
    for name, path in TARGET_PATHS.items():
        target_df = pd.read_csv(path)
        targets[name] = target_df.iloc[:, 0].astype(int)

    for name, y in targets.items():
        if len(y) != len(X):
            raise ValueError(f"Target {name} has {len(y)} rows, expected {len(X)}.")

    return X, targets


def preprocess_features(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep numeric columns and standardize them (input is already imputed)."""
    numeric = X.select_dtypes(include=np.number).copy()
    if numeric.empty:
        raise ValueError("No numeric features found in the imputed dataset.")

    dropped = [c for c in X.columns if c not in numeric.columns]
    if dropped:
        print(f"Note: {len(dropped)} non-numeric columns excluded from clustering: {dropped}")

    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(numeric),
        columns=numeric.columns,
        index=numeric.index,
    )
    return numeric, numeric_scaled


def reduce_dimensions_for_clustering(X_scaled: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """Compress the feature space before KMeans to reduce noise and collinearity."""
    print("Performing PCA to reduce dimensionality ...")
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca


def evaluate_k_values(X_reduced: np.ndarray, k_values: range) -> pd.DataFrame:
    """Score candidate cluster counts with standard internal validation metrics."""
    records: list[dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X_reduced)
        counts = np.bincount(labels)
        records.append(
            {
                "k": float(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X_reduced, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_reduced, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_reduced, labels)),
                "min_cluster_size": float(counts.min()),
                "max_cluster_size": float(counts.max()),
            }
        )
    return pd.DataFrame(records)


def compute_bootstrap_stability(X_reduced: np.ndarray, k_values: range) -> dict[int, tuple[float, float]]:
    """Average adjusted Rand index between full-data labels and resample labels.

    A high value means the k-cluster solution is reproduced on independent
    subsamples (stable structure); a value close to zero means the solution
    changes with the sample (no real structure at that k).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_reduced)
    stability: dict[int, tuple[float, float]] = {}
    for k in k_values:
        full_labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_reduced)
        aris: list[float] = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, size=int(n * BOOTSTRAP_FRACTION), replace=False)
            boot_labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit_predict(X_reduced[idx])
            aris.append(adjusted_rand_score(full_labels[idx], boot_labels))
        stability[k] = (float(np.mean(aris)), float(np.std(aris)))
        print(f"  k={k}: stability ARI = {stability[k][0]:.3f} (+/- {stability[k][1]:.3f})")
    return stability


def choose_k(metrics: pd.DataFrame, stability: dict[int, tuple[float, float]]) -> int:
    """Pick the cluster count with the best silhouette that is also reproducible.

    Among the k values whose bootstrap stability is at least half of the best
    stability, pick the one with the highest silhouette score.
    """
    best_stab = max(stab for stab, _ in stability.values())
    candidates = [
        int(row["k"])
        for _, row in metrics.iterrows()
        if stability[int(row["k"])][0] >= 0.5 * best_stab
    ]
    pool = metrics.loc[metrics["k"].isin(candidates)]
    return int(pool.sort_values("silhouette", ascending=False).iloc[0]["k"])


def fit_final_model(X_reduced: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """Fit the final KMeans model."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(X_reduced)
    return labels, model


def plot_cluster_selection(metrics: pd.DataFrame) -> None:
    """Plot how the validation metrics change across candidate k values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=250)
    axes[0].plot(metrics["k"], metrics["silhouette"], marker="o")
    axes[0].set_title("Silhouette score")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("score")

    axes[1].plot(metrics["k"], metrics["calinski_harabasz"], marker="o", color="#f58518")
    axes[1].set_title("Calinski-Harabasz")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("score")

    axes[2].plot(metrics["k"], metrics["davies_bouldin"], marker="o", color="#54a24b")
    axes[2].set_title("Davies-Bouldin")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("lower is better")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_selection_metrics.png")
    plt.close()


def plot_pca_scatter(X_scaled: pd.DataFrame, labels: np.ndarray) -> None:
    """Visualize the clusters on the first two PCA components."""
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca_2d.fit_transform(X_scaled)

    plt.figure(figsize=(9, 8), dpi=250)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Respondent clusters in 2D PCA space")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_pca_scatter.png")
    plt.close()


def plot_cluster_profiles(cluster_means: pd.DataFrame, output_path: Path) -> list[str]:
    """Plot the most distinctive standardized features for each cluster."""
    profile_dispersion = cluster_means.std(axis=0).sort_values(ascending=False)
    top_features = profile_dispersion.head(TOP_PROFILE_FEATURES).index.tolist()

    plt.figure(figsize=(12, 6), dpi=250)
    sns.heatmap(cluster_means[top_features], cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Cluster profile on standardized features")
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return top_features


def plot_target_distributions(labels: np.ndarray, targets: dict[str, pd.Series]) -> None:
    """Show how SPRENER and SPRACQUA response classes are distributed in each cluster."""
    for target_name, y in targets.items():
        ctab = pd.crosstab(labels, y, normalize="index")
        ctab = ctab.reindex(columns=sorted(y.unique()), fill_value=0.0)

        ax = ctab.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis", width=0.85)
        ax.set_title(f"{target_name.upper()} response distribution by cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Within-cluster proportion")
        ax.legend(title=target_name.upper(), bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cluster_{target_name}_distribution.png")
        plt.close()


def build_cluster_summary(
    labels: np.ndarray,
    X_scaled: pd.DataFrame,
    targets: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cluster-level summary tables."""
    summary = pd.DataFrame({"cluster": np.arange(labels.max() + 1)})
    summary["size"] = summary["cluster"].map(pd.Series(labels).value_counts()).fillna(0).astype(int)

    cluster_means = X_scaled.assign(cluster=labels).groupby("cluster").mean().sort_index()

    for target_name, y in targets.items():
        summary[f"mean_{target_name}"] = summary["cluster"].map(
            pd.Series(y.values).groupby(labels).mean()
        )

    summary["dominant_sprener"] = summary["cluster"].map(
        pd.Series(targets["sprener"].values).groupby(labels).agg(lambda s: s.mode().iloc[0])
    )
    summary["dominant_spracqua"] = summary["cluster"].map(
        pd.Series(targets["spracqua"].values).groupby(labels).agg(lambda s: s.mode().iloc[0])
    )

    return summary, cluster_means


def build_profile_table(cluster_means: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    """Long-form profile table with plain-language feature descriptions."""
    records = []
    for feature in top_features:
        row = {"feature": feature, "description": describe(feature)}
        for cluster in cluster_means.index:
            row[f"cluster_{cluster}"] = cluster_means.loc[cluster, feature]
        records.append(row)
    return pd.DataFrame(records)


def save_target_tables(labels: np.ndarray, targets: dict[str, pd.Series]) -> None:
    """Save target-by-cluster count and proportion tables as TXT (CSV kept for data)."""
    for target_name, y in targets.items():
        counts = pd.crosstab(labels, y).sort_index()
        proportions = counts.div(counts.sum(axis=1), axis=0)

        with (OUTPUT_DIR / f"cluster_{target_name}_counts.txt").open("w", encoding="utf-8") as fh:
            fh.write(render_table(counts, title=f"{target_name.upper()} counts by cluster") + "\n")
        with (OUTPUT_DIR / f"cluster_{target_name}_proportions.txt").open("w", encoding="utf-8") as fh:
            fh.write(
                render_table(
                    proportions,
                    title=f"{target_name.upper()} within-cluster proportions",
                    formats={c: ".4f" for c in proportions.columns},
                )
                + "\n"
            )

        counts.to_csv(OUTPUT_DIR / f"cluster_{target_name}_counts.csv")
        proportions.to_csv(OUTPUT_DIR / f"cluster_{target_name}_proportions.csv")


def main() -> KMeans | None:
    print("Running survey respondent clustering for behavioral profiling...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_raw, targets = load_data()
    _, X_scaled = preprocess_features(X_raw)
    print(f"Loaded {len(X_scaled)} respondents and {X_scaled.shape[1]} features.")

    X_reduced, pca = reduce_dimensions_for_clustering(X_scaled)
    print(f"PCA retained {pca.n_components_} components to explain 90% of the variance.")

    metrics = evaluate_k_values(X_reduced, K_RANGE)
    print("\nBootstrap stability (adjusted Rand index):")
    stability = compute_bootstrap_stability(X_reduced, K_RANGE)

    metrics["stability_ari"] = metrics["k"].map(lambda k: stability[int(k)][0])
    metrics["stability_sd"] = metrics["k"].map(lambda k: stability[int(k)][1])
    with (OUTPUT_DIR / "cluster_validation_metrics.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                metrics,
                title="Cluster validation metrics (silhouette/CH/DB + bootstrap stability)",
                formats={
                    "k": ".0f",
                    "inertia": ".0f",
                    "silhouette": ".4f",
                    "calinski_harabasz": ".1f",
                    "davies_bouldin": ".4f",
                    "stability_ari": ".4f",
                    "stability_sd": ".4f",
                    "min_cluster_size": ".0f",
                    "max_cluster_size": ".0f",
                },
            )
            + "\n"
        )
    metrics.to_csv(OUTPUT_DIR / "cluster_validation_metrics.csv", index=False)
    plot_cluster_selection(metrics)

    best_k = choose_k(metrics, stability)
    best_metrics = metrics.loc[metrics["k"] == best_k].iloc[0]
    print(
        "\nCandidate clustering quality: "
        f"best k={best_k}, silhouette={best_metrics['silhouette']:.3f}, "
        f"CH={best_metrics['calinski_harabasz']:.1f}, "
        f"DB={best_metrics['davies_bouldin']:.3f}, "
        f"stability ARI={best_metrics['stability_ari']:.3f}"
    )

    labels, model = fit_final_model(X_reduced, best_k)
    cluster_sizes = pd.Series(labels, name="cluster").value_counts().sort_index()
    print("Cluster sizes:")
    print(cluster_sizes.to_string())

    cluster_summary, cluster_means = build_cluster_summary(labels, X_scaled, targets)
    with (OUTPUT_DIR / "cluster_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                cluster_summary,
                title="Cluster summary (targets coded 1 = high care, 4 = no care)",
                formats={c: ".4f" for c in cluster_summary.columns if c.startswith("mean_")},
            )
            + "\n"
        )
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    cluster_means.to_csv(OUTPUT_DIR / "cluster_feature_means_standardized.csv")

    save_target_tables(labels, targets)
    plot_target_distributions(labels, targets)
    top_features = plot_cluster_profiles(cluster_means, OUTPUT_DIR / "cluster_feature_profile_heatmap.png")
    plot_pca_scatter(X_scaled, labels)

    profile_table = build_profile_table(cluster_means, top_features)
    with (OUTPUT_DIR / "cluster_profiles.txt").open("w", encoding="utf-8") as fh:
        fmt = {f"cluster_{c}": ".2f" for c in cluster_means.index}
        fh.write(render_table(profile_table, title="Most distinctive cluster features (standardized means)", formats=fmt) + "\n")
    profile_table.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)

    membership = pd.DataFrame(
        {
            "cluster": labels,
            "sprener": targets["sprener"].values,
            "spracqua": targets["spracqua"].values,
        }
    )
    membership.to_csv(OUTPUT_DIR / "cluster_membership.csv", index=False)

    print("\nMost distinctive standardized features in the cluster profile:")
    for feature in top_features:
        print(f"  - {feature}: {describe(feature)}")
    print(f"\nSaved clustering outputs to: {OUTPUT_DIR}")
    print(
        "Note: SPRENER and SPRACQUA are coded from 1 to 4, where 1 means "
        "more attention to saving behavior and 4 means never."
    )

    return model


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# CONCLUSIONS FROM THE RESPONDENT CLUSTERING
# -----------------------------------------------------------------------------
# 1) The survey does not contain well-separated citizen profiles. k=3 is chosen
#    (best silhouette among the stable k values), but every k in 3-8 has a very
#    low silhouette (0.008-0.027). The bootstrap stability is high (ARI ~0.97),
#    so the weak partition is reproducible, but the clusters are soft and the
#    respondent space is nearly continuous rather than grouped.
#
# 2) The clusters are driven by age and digital/internet usage, not by
#    sustainability. The 12 most distinctive features are INTTEMPO, PCTEMPO,
#    PCOPE_FILE, PCOPEWO, ETAMi (age), PWEB, INTATT11, PCOPE_SF, PCOPE_PASTE,
#    INTATT30A, INTATT16 and PCOPEPH. None of the behaviour items that drive the
#    target gradient in 3_exploratory_analysis.py (ALOCAL, ETICHET, BIOLOG,
#    ARUMOR, USAGETT) appear among the distinguishing features.
#
# 3) SPRENER and SPRACQUA barely differ across clusters: mean values range only
#    from ~1.40 to ~1.53 and every cluster is dominated by class 1 (proportions
#    0.66-0.74). The younger/digitally-savvy cluster is marginally the most
#    attentive and the lowest-usage cluster the least, but the spread is small.
#
# 4) Interpretation 
#    The coherent behavioural gradient found in the exploratory analysis 
#    (3_exploratory_analysis.py) operates at the level of
#    individuals and is weak/diffuse (see also 5_rf_fitting.py). Unsupervised
#    clustering on the full 299-feature matrix therefore separates respondents
#    by age and digital usage rather than by sustainability attitude. No
#    separate "sustainable-citizen" segment emerges; sustainability behaves as
#    a diffuse, cross-cutting latent disposition instead of a defining cluster
#    dimension.
# -----------------------------------------------------------------------------
