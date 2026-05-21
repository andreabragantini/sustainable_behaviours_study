"""Respondent clustering for behavioral profiling.

This script clusters respondents, not variables. The goal is to see whether the
survey contains a few coherent citizen profiles and then inspect how SPRENER and
SPRACQUA vary across those profiles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
FEATURES_PATH = Path("data/preprocessed_data.csv")
TARGET_PATHS = {
    "sprener": Path("data/tar_sprener.csv"),
    "spracqua": Path("data/tar_spracqua.csv"),
}
OUTPUT_DIR = Path("exploratory_analysis") / "clustering"
K_RANGE = range(3, 9)
N_INIT = 20
TOP_PROFILE_FEATURES = 12


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
    """Keep numeric columns, impute missing values, and standardize them."""
    numeric = X.select_dtypes(include=np.number).copy()
    if numeric.empty:
        raise ValueError("No numeric features found in preprocessed_data.csv")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    numeric_imputed = pd.DataFrame(
        imputer.fit_transform(numeric),
        columns=numeric.columns,
        index=numeric.index,
    )
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(numeric_imputed),
        columns=numeric.columns,
        index=numeric.index,
    )

    return numeric_imputed, numeric_scaled


def reduce_dimensions_for_clustering(X_scaled: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """Compress the feature space before KMeans to reduce noise and collinearity."""
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


def choose_k(metrics: pd.DataFrame) -> int:
    """Pick the cluster count with the best silhouette score."""
    best_row = metrics.sort_values(["silhouette", "calinski_harabasz"], ascending=[False, False]).iloc[0]
    return int(best_row["k"])


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


def save_target_tables(labels: np.ndarray, targets: dict[str, pd.Series]) -> None:
    """Save target-by-cluster count and proportion tables."""
    for target_name, y in targets.items():
        counts = pd.crosstab(labels, y).sort_index()
        proportions = counts.div(counts.sum(axis=1), axis=0)

        counts.to_csv(OUTPUT_DIR / f"cluster_{target_name}_counts.csv")
        proportions.to_csv(OUTPUT_DIR / f"cluster_{target_name}_proportions.csv")


def main() -> KMeans | None:
    print("Running respondent clustering for behavioral profiling...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_raw, targets = load_data()
    _, X_scaled = preprocess_features(X_raw)
    X_reduced, pca = reduce_dimensions_for_clustering(X_scaled)

    print(f"Loaded {len(X_scaled)} respondents and {X_scaled.shape[1]} features.")
    print(f"PCA retained {pca.n_components_} components to explain 90% of the variance.")

    metrics = evaluate_k_values(X_reduced, K_RANGE)
    metrics.to_csv(OUTPUT_DIR / "cluster_validation_metrics.csv", index=False)
    plot_cluster_selection(metrics)

    best_k = choose_k(metrics)
    best_metrics = metrics.loc[metrics["k"] == best_k].iloc[0]
    print(
        "Candidate clustering quality: "
        f"best k={best_k}, silhouette={best_metrics['silhouette']:.3f}, "
        f"CH={best_metrics['calinski_harabasz']:.1f}, DB={best_metrics['davies_bouldin']:.3f}"
    )

    labels, model = fit_final_model(X_reduced, best_k)
    cluster_sizes = pd.Series(labels, name="cluster").value_counts().sort_index()
    print("Cluster sizes:")
    print(cluster_sizes.to_string())

    cluster_summary, cluster_means = build_cluster_summary(labels, X_scaled, targets)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    cluster_means.to_csv(OUTPUT_DIR / "cluster_feature_means_standardized.csv")
    save_target_tables(labels, targets)
    plot_target_distributions(labels, targets)
    top_features = plot_cluster_profiles(cluster_means, OUTPUT_DIR / "cluster_feature_profile_heatmap.png")
    plot_pca_scatter(X_scaled, labels)

    membership = pd.DataFrame(
        {
            "cluster": labels,
            "sprener": targets["sprener"].values,
            "spracqua": targets["spracqua"].values,
        }
    )
    membership.to_csv(OUTPUT_DIR / "cluster_membership.csv", index=False)

    print("\nMost distinctive standardized features in the cluster profile:")
    print(", ".join(top_features))
    print(f"\nSaved clustering outputs to: {OUTPUT_DIR}")
    print(
        "Interpretation note: SPRENER and SPRACQUA are coded from 1 to 4, where 1 means "
        "more attention to saving behavior and 4 means never.")

    # Keep the fitted model accessible if the script is imported interactively.
    return model


if __name__ == "__main__":
    main()


'''
Simple respondent-segmentation analysis and ran it successfully. It now clusters people, not variables, 
using standardized features plus PCA denoising, chooses k by internal validation, and exports cluster summaries
and plots under clustering.

The result is usable as an exploratory profile analysis, but the structure is weak: the best solution was k=3 
with silhouette 0.070, so the clusters are only mildly separated. 
Still, they are informative for your research goal: one cluster is clearly more pro-saving than the others,
while another is less uniformly pro-saving, and all three are still dominated by the most sustainable response class. 
That means the data seems to contain behavioral subgroups, but not strong, clean “types” of Italian citizens. 
So clustering is useful here as a descriptive segmentation tool, not as primary evidence of which features cause 
or predict SPRENER/SPRACQUA.
'''