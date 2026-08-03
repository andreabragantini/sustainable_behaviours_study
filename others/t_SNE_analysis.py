"""
This script performs feature selection using t-SNE for dimensionality reduction and KMeans for clustering.
The goal is to identify groups of variables that are redundant and can be removed from the dataset.
"""
import pandas as pd
"""Dedicated t-SNE analysis for the AVQ preprocessed dataset.

This script builds one shared t-SNE embedding from the feature matrix and then
colors it by each target variable (sprener and spracqua). It also computes
simple quantitative checks to estimate whether class labels are separable in
the 2D embedding.
"""

from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGETS = ["sprener", "spracqua"]
PERPLEXITY = 30
N_ITER = 1000
PCA_COMPONENTS = 50
OUT_DIR = Path("exploratory_analysis") / "tSNE_analysis"


def load_targets(target_names: list[str]) -> dict[str, pd.Series]:
    """Load all target vectors as 1D integer series."""
    targets: dict[str, pd.Series] = {}
    for target_name in target_names:
        target_df = pd.read_csv(f"data/tar_{target_name}.csv")
        target_series = target_df.iloc[:, 0].astype(int)
        targets[target_name] = target_series
    return targets


def build_tsne_embedding(features_df: pd.DataFrame) -> np.ndarray:
    """Scale, optionally denoise with PCA, then embed with t-SNE."""
    numeric_df = features_df.select_dtypes(include=np.number).copy()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    scaled = StandardScaler().fit_transform(numeric_df)

    if scaled.shape[1] > PCA_COMPONENTS:
        reducer = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        tsne_input = reducer.fit_transform(scaled)
    else:
        tsne_input = scaled

    valid_perplexity = min(PERPLEXITY, max(5, (tsne_input.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=valid_perplexity,
        n_iter=N_ITER,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    )
    return tsne.fit_transform(tsne_input)


def evaluate_target_separation(embedding: np.ndarray, labels: pd.Series) -> dict[str, float]:
    """Compute simple metrics that quantify class separation in t-SNE space."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    knn = KNeighborsClassifier(n_neighbors=25)
    dummy = DummyClassifier(strategy="most_frequent")

    knn_score = cross_val_score(knn, embedding, labels, cv=cv, scoring="balanced_accuracy").mean()
    dummy_score = cross_val_score(dummy, embedding, labels, cv=cv, scoring="balanced_accuracy").mean()

    f_values, p_values = f_classif(embedding, labels)

    # Silhouette with true labels: near 0 means labels are not geometrically separated.
    sil = float(silhouette_score(embedding, labels))

    return {
        "balanced_accuracy_knn": float(knn_score),
        "balanced_accuracy_dummy": float(dummy_score),
        "knn_minus_dummy": float(knn_score - dummy_score),
        "f_tsne1": float(f_values[0]),
        "f_tsne2": float(f_values[1]),
        "p_tsne1": float(p_values[0]),
        "p_tsne2": float(p_values[1]),
        "silhouette_by_true_labels": sil,
    }


def plot_embedding(embedding: np.ndarray, labels: pd.Series, target_name: str) -> None:
    """Save a color-coded scatter plot of the shared t-SNE embedding."""
    unique_labels = np.sort(labels.unique())

    plt.figure(figsize=(10, 10), dpi=300)
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        s=5,
        cmap="viridis",
        alpha=0.85,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_ticks(unique_labels)
    cbar.set_label(target_name)

    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.title(f"t-SNE visualization for {target_name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"tsne_{target_name}.png")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building shared t-SNE embedding from preprocessed features...")
    start = time.time()

    X = pd.read_csv("data/preprocessed_data_final.csv")
    targets = load_targets(TARGETS)

    row_count = len(X)
    for target_name, target_values in targets.items():
        if len(target_values) != row_count:
            raise ValueError(
                f"Target {target_name} has {len(target_values)} rows, expected {row_count}."
            )

    embedding = build_tsne_embedding(X)
    elapsed_min = (time.time() - start) / 60
    print(f"t-SNE completed in {elapsed_min:.2f} minutes.")

    results: list[dict[str, Any]] = []

    for target_name, target_values in targets.items():
        print(f"\nAnalyzing target: {target_name}")
        plot_embedding(embedding, target_values, target_name)
        metrics: dict[str, Any] = evaluate_target_separation(embedding, target_values)
        metrics["target"] = target_name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "target",
            "balanced_accuracy_knn",
            "balanced_accuracy_dummy",
            "knn_minus_dummy",
            "silhouette_by_true_labels",
            "f_tsne1",
            "p_tsne1",
            "f_tsne2",
            "p_tsne2",
        ]
    ]

    out_csv = OUT_DIR / "tsne_target_separation_metrics.csv"
    results_df.to_csv(out_csv, index=False)

    print("\nTarget separation metrics:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved metrics to: {out_csv}")
    print("Saved figures to exploratory_analysis/tSNE_analysis/tsne_<target>.png")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# QUICK CONCLUSIONS FROM THIS ANALYSIS
# -----------------------------------------------------------------------------
# 1) The elongated t-SNE geometry suggests a dominant low-dimensional continuum
#    in the feature space (a few latent gradients explain much of local layout).
# 2) For both targets (sprener and spracqua), colors are mixed across the shape:
#    there is no clear target-specific region in the embedding.
# 3) Quantitative checks support this visual result:
#    - KNN balanced accuracy is almost equal to dummy baseline.
#    - Silhouette by true labels is negative (class overlap dominates).
# 4) Practical interpretation:
#    t-SNE is useful here for visualization of structure, but it does not provide
#    evidence of strong target separability/predictive signal for these labels.
# 5) Caution:
#    t-SNE preserves local neighborhoods better than global geometry, so do not
#    over-interpret the exact global shape as a causal data-generating mechanism.
