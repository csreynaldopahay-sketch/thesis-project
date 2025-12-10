"""End-to-end data preparation and analysis pipeline for the AMR dataset.

This script implements the phases described in the project plan:
- Phase 0: cleaning, encoding, target creation, and stratified data splitting
- Phase 1: unsupervised clustering, dimensionality reduction, and association rule mining
- Phase 2: supervised classification for MAR index prediction and species identification

The script is designed to be run as a CLI and to keep all changes minimal while
allowing reproducible processing of the `rawdata.csv` file found in the repo.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


RANDOM_STATE = 42
ANTIBIOTIC_ENCODING = {"s": 0, "i": 1, "r": 2}
INTERPRETATION_SUFFIX = "_int"


def clean_interpretation_value(value: str) -> Optional[str]:
    """Normalize raw interpretation strings to s/i/r."""
    if value is None:
        return None
    val = str(value).strip().lower()
    if not val:
        return None
    match = re.search(r"[sir]", val)
    return match.group(0) if match else None


def normalize_metadata(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure consistent text formatting for metadata columns."""
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )
    return df


def fill_missing_interpretations(df: pd.DataFrame, interpretation_cols: List[str]) -> pd.DataFrame:
    """Fill missing antibiotic interpretation values with column modes."""
    for col in interpretation_cols:
        mode_val = df[col].mode(dropna=True)
        fill_value = mode_val.iloc[0] if not mode_val.empty else "s"
        df[col] = df[col].fillna(fill_value)
    return df


def drop_rows_with_excess_missing(
    df: pd.DataFrame, interpretation_cols: List[str], max_missing_fraction: float
) -> pd.DataFrame:
    """Remove rows that exceed the allowed missing fraction for interpretation columns."""
    missing_fraction = df[interpretation_cols].isna().mean(axis=1)
    return df.loc[missing_fraction <= max_missing_fraction].copy()


def encode_interpretations(df: pd.DataFrame, interpretation_cols: List[str]) -> pd.DataFrame:
    """Ordinally encode s/i/r to 0/1/2."""
    encoded = df[interpretation_cols].replace(ANTIBIOTIC_ENCODING)
    return encoded.astype(int)


def prepare_dataset(
    raw_path: Path,
    mar_threshold: float = 0.3,
    max_missing_fraction: float = 0.4,
    min_species_size: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Complete Phase 0: cleaning, encoding, targets, and labeling."""
    df = pd.read_csv(raw_path)
    interpretation_cols = [c for c in df.columns if c.endswith(INTERPRETATION_SUFFIX)]
    mic_cols = [c for c in df.columns if c.endswith("_mic")]

    # Remove irrelevant columns (MIC values and non-essential identifiers)
    cols_to_drop = set(mic_cols + ["isolate_code", "replicate", "colony", "esbl"])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Normalize categorical metadata
    df = normalize_metadata(
        df,
        ["bacterial_species", "administrative_region", "national_site", "local_site", "sample_source"],
    )

    # Standardize interpretation values
    for col in interpretation_cols:
        df[col] = df[col].apply(clean_interpretation_value)

    df = drop_rows_with_excess_missing(df, interpretation_cols, max_missing_fraction)
    df = fill_missing_interpretations(df, interpretation_cols)
    encoded_features = encode_interpretations(df, interpretation_cols)

    # Targets
    df["mar_index"] = pd.to_numeric(df["mar_index"], errors="coerce")
    df["high_mar_target"] = (df["mar_index"] > mar_threshold).astype(int)

    species_counts = df["bacterial_species"].value_counts()
    df["species_target"] = df["bacterial_species"].where(
        df["bacterial_species"].map(species_counts) >= min_species_size, "other"
    )

    return df, encoded_features, df["high_mar_target"], df["species_target"], interpretation_cols


def stratified_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.1, val_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train/validation/test with requested proportions and stratification."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=RANDOM_STATE
    )
    adjusted_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=adjusted_test_size, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_clustering(encoded_features: pd.DataFrame, output_dir: Path) -> None:
    """Execute KMeans, hierarchical clustering, and DBSCAN on encoded resistance profiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled = StandardScaler().fit_transform(encoded_features)

    max_k = min(10, encoded_features.shape[0])
    k_range = range(2, max_k)
    inertia = []
    silhouette = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = kmeans.fit_predict(scaled)
        inertia.append(kmeans.inertia_)
        if len(set(labels)) > 1:
            silhouette.append(silhouette_score(scaled, labels))
        else:
            silhouette.append(float("nan"))

    elbow_data = pd.DataFrame({"k": list(k_range), "inertia": inertia, "silhouette": silhouette})
    elbow_data.to_csv(output_dir / "cluster_selection_metrics.csv", index=False)

    best_k_idx = (
        np.nanargmax(silhouette) if any(not math.isnan(v) for v in silhouette) else 0
    )
    best_k = list(k_range)[best_k_idx]

    # Final clustering models
    kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init="auto").fit(scaled)
    agglomerative = AgglomerativeClustering(n_clusters=best_k).fit(scaled)
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(scaled)

    cluster_df = pd.DataFrame(
        {
            "kmeans_cluster": kmeans_final.labels_,
            "hierarchical_cluster": agglomerative.labels_,
            "dbscan_cluster": dbscan.labels_,
        }
    )
    cluster_df.to_csv(output_dir / "cluster_labels.csv", index=False)


def run_dimensionality_reduction(encoded_features: pd.DataFrame, labels: pd.Series, output_dir: Path) -> None:
    """Generate PCA, t-SNE, and UMAP embeddings for visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded_features)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_coords, columns=["pc1", "pc2"])
    pca_df["label"] = labels.values
    pca_df.to_csv(output_dir / "pca_embedding.csv", index=False)

    plt.figure()
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="label", s=30)
    plt.title("PCA (variance explained: {:.2f}%)".format(pca.explained_variance_ratio_.sum() * 100))
    plt.tight_layout()
    plt.savefig(output_dir / "pca_plot.png")
    plt.close()

    tsne_coords = TSNE(n_components=2, random_state=RANDOM_STATE, init="random").fit_transform(scaled)
    tsne_df = pd.DataFrame(tsne_coords, columns=["tsne1", "tsne2"])
    tsne_df["label"] = labels.values
    tsne_df.to_csv(output_dir / "tsne_embedding.csv", index=False)

    if umap is not None:
        reducer = umap.UMAP(random_state=RANDOM_STATE)
        umap_coords = reducer.fit_transform(scaled)
        umap_df = pd.DataFrame(umap_coords, columns=["umap1", "umap2"])
        umap_df["label"] = labels.values
        umap_df.to_csv(output_dir / "umap_embedding.csv", index=False)


def run_association_rule_mining(encoded_features: pd.DataFrame, output_dir: Path) -> None:
    """Identify co-resistance patterns using Apriori and FP-Growth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resistant_binary = encoded_features == ANTIBIOTIC_ENCODING["r"]

    apriori_df = apriori(resistant_binary, min_support=0.02, use_colnames=True)
    apriori_rules = association_rules(apriori_df, metric="confidence", min_threshold=0.6)
    apriori_rules = apriori_rules[apriori_rules["lift"] > 1]
    apriori_rules.to_csv(output_dir / "apriori_rules.csv", index=False)

    fpg_df = fpgrowth(resistant_binary, min_support=0.02, use_colnames=True)
    fpg_rules = association_rules(fpg_df, metric="confidence", min_threshold=0.6)
    fpg_rules = fpg_rules[fpg_rules["lift"] > 1]
    fpg_rules.to_csv(output_dir / "fpgrowth_rules.csv", index=False)


def build_supervised_models(num_classes: int) -> Dict[str, object]:
    """Set up the supervised models for MAR and species classification."""
    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=500, multi_class="auto"),
        "svm": SVC(kernel="rbf", probability=True),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE
        ),
    }
    if xgb is not None:
        models["xgboost"] = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=300,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        )
    return models


def evaluate_model_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary"
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    average: str,
) -> pd.DataFrame:
    """Fit models and evaluate them on the validation set."""
    models = build_supervised_models(num_classes=len(np.unique(y_train)))
    results = []
    for name, model in models.items():
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        metrics = evaluate_model_predictions(y_val, preds, average=average)
        metrics["model"] = name
        results.append(metrics)
    return pd.DataFrame(results)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_file: Path) -> None:
    """Persist confusion matrix as CSV."""
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(output_file, index=False)


def run_supervised_tasks(
    encoded_features: pd.DataFrame,
    mar_target: pd.Series,
    species_target: pd.Series,
    output_dir: Path,
) -> None:
    """Execute Phase 2 supervised modelling for MAR and species classification."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # High MAR classification
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(encoded_features, mar_target)
    mar_results = train_and_evaluate(X_train, y_train, X_val, y_val, average="binary")
    mar_results.to_csv(output_dir / "mar_model_comparison.csv", index=False)

    # Evaluate best model on test set (by F1)
    best_model_name = mar_results.sort_values("f1", ascending=False).iloc[0]["model"]
    best_model = build_supervised_models(num_classes=2)[best_model_name]
    mar_pipeline = make_pipeline(StandardScaler(), best_model)
    mar_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    mar_preds = mar_pipeline.predict(X_test)
    save_confusion_matrix(y_test, mar_preds, output_dir / "mar_confusion_matrix.csv")

    # Species classification (multi-class)
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = stratified_split(
        encoded_features, species_target
    )
    species_results = train_and_evaluate(X_train_s, y_train_s, X_val_s, y_val_s, average="weighted")
    species_results.to_csv(output_dir / "species_model_comparison.csv", index=False)

    best_species_model_name = species_results.sort_values("f1", ascending=False).iloc[0]["model"]
    best_species_model = build_supervised_models(num_classes=species_target.nunique())[
        best_species_model_name
    ]
    species_pipeline = make_pipeline(StandardScaler(), best_species_model)
    species_pipeline.fit(pd.concat([X_train_s, X_val_s]), pd.concat([y_train_s, y_val_s]))
    species_preds = species_pipeline.predict(X_test_s)
    save_confusion_matrix(
        y_test_s, species_preds, output_dir / "species_confusion_matrix.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMR analysis pipeline")
    parser.add_argument("--raw-path", type=Path, default=Path("rawdata.csv"), help="Path to raw CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Directory for generated artifacts"
    )
    parser.add_argument("--mar-threshold", type=float, default=0.3, help="MAR index threshold")
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.4,
        help="Max fraction of missing interpretation columns before dropping a row",
    )
    parser.add_argument(
        "--min-species-size",
        type=int,
        default=10,
        help="Minimum samples per species before merging into 'other'",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only run Phase 0 (preprocessing) and print dataset summaries",
    )
    parser.add_argument(
        "--skip-unsupervised", action="store_true", help="Skip clustering/dimensionality reduction"
    )
    parser.add_argument(
        "--skip-supervised", action="store_true", help="Skip supervised model training"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, encoded_features, mar_target, species_target, interpretation_cols = prepare_dataset(
        args.raw_path,
        mar_threshold=args.mar_threshold,
        max_missing_fraction=args.max_missing_fraction,
        min_species_size=args.min_species_size,
    )

    print(f"Loaded {len(df)} cleaned isolates with {len(interpretation_cols)} antibiotics.")
    print("Class balance (high MAR):")
    print(mar_target.value_counts(normalize=True))
    print("Species label distribution:")
    print(species_target.value_counts().head())

    if args.summary_only:
        return

    if not args.skip_unsupervised:
        run_clustering(encoded_features, args.output_dir / "unsupervised")
        run_dimensionality_reduction(encoded_features, species_target, args.output_dir / "unsupervised")
        run_association_rule_mining(encoded_features, args.output_dir / "unsupervised")

    if not args.skip_supervised:
        run_supervised_tasks(encoded_features, mar_target, species_target, args.output_dir / "supervised")


if __name__ == "__main__":
    main()
