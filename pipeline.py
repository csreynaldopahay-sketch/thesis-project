from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


RANDOM_STATE = 42
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "rawdata.csv"
OUTPUT_DIR = BASE_PATH / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

ENCODING_MAP = {"s": 0, "i": 1, "r": 2}
DEFAULT_INTERPRETATION_FILL = "s"
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 5
APRIORI_MIN_SUPPORT = 0.02
FPGROWTH_MIN_SUPPORT = 0.03
ASSOCIATION_MIN_CONFIDENCE = 0.6
ASSOCIATION_MIN_LIFT = 1.0


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class BestModel(TypedDict):
    name: str
    model: Pipeline
    metrics: Dict[str, float]


def top_species_proportions(series: pd.Series) -> Dict[str, float]:
    counts = series.value_counts(normalize=True).head(3)
    return counts.to_dict()


def ensure_output_dirs() -> None:
    for folder in (OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, ARTIFACTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _normalize_interpretation(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    text = text.replace("*", "")
    text = re.sub(r"[^sir]", "", text)
    return text if text in {"s", "i", "r"} else np.nan


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "bacterial_species",
        "administrative_region",
        "national_site",
        "local_site",
        "sample_source",
    ):
        if col in df:
            df[col] = (
                df[col].astype(str).str.strip().str.lower().replace({"nan": np.nan})
            )
    return df


def prepare_dataset(
    df: pd.DataFrame,
    row_missing_threshold: float = 0.4,
    column_missing_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, List[str]]:
    df = standardize_labels(df)
    mic_cols = [col for col in df.columns if col.endswith("_mic")]
    drop_cols = [col for col in mic_cols + ["isolate_code", "replicate", "colony"] if col in df]
    df = df.drop(columns=drop_cols)

    interpretation_cols = [col for col in df.columns if col.endswith("_int")]
    if not interpretation_cols:
        raise ValueError("No interpretation columns found.")

    missing_by_col = df[interpretation_cols].isna().mean()
    feature_cols = [c for c in interpretation_cols if missing_by_col[c] < column_missing_threshold]

    df[feature_cols] = df[feature_cols].apply(lambda col: col.map(_normalize_interpretation))
    row_missing = df[feature_cols].isna().mean(axis=1)
    df = df.loc[row_missing <= row_missing_threshold].copy()

    for col in feature_cols:
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else DEFAULT_INTERPRETATION_FILL
        df[col] = df[col].fillna(fill_value)

    df[feature_cols] = df[feature_cols].replace(ENCODING_MAP).astype(int)
    return df, feature_cols


def add_targets(
    df: pd.DataFrame,
    mar_threshold: float = 0.3,
    species_min_count: int = 20,
) -> pd.DataFrame:
    df = df.copy()
    df["high_mar"] = (df["mar_index"] > mar_threshold).astype(int)
    species_counts = df["bacterial_species"].value_counts()
    common_species = species_counts[species_counts >= species_min_count].index
    df["species_grouped"] = df["bacterial_species"].where(
        df["bacterial_species"].isin(common_species), "other"
    )
    return df


def encode_species_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    encoder = LabelEncoder()
    df = df.copy()
    df["species_encoded"] = encoder.fit_transform(df["species_grouped"])
    return df, encoder


def split_dataset(
    df: pd.DataFrame, feature_cols: List[str], target: str
) -> DatasetSplits:
    X = df[feature_cols]
    y = df[target]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio = 0.2 / 0.9
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )
    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def build_pipeline(model, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_model_suite(num_classes: int) -> Dict[str, Pipeline]:
    is_binary = num_classes == 2
    return {
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced" if is_binary else None
            ),
            scale=False,
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic" if is_binary else "multi:softprob",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
        "logistic_regression": build_pipeline(
            LogisticRegression(
                max_iter=500,
                class_weight="balanced" if is_binary else None,
                multi_class="auto",
            ),
            scale=True,
        ),
        "svm": build_pipeline(
            SVC(
                probability=True,
                class_weight="balanced" if is_binary else None,
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        "knn": build_pipeline(KNeighborsClassifier(n_neighbors=7), scale=True),
        # MultinomialNB fits the ordinal encoded (0/1/2) resistance categories without scaling.
        "naive_bayes": build_pipeline(MultinomialNB(), scale=False),
    }


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, average: str
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def evaluate_models(
    splits: DatasetSplits, num_classes: int, task_name: str, average: str
) -> Tuple[List[Dict[str, float]], BestModel]:
    models = get_model_suite(num_classes)
    results: List[Dict[str, float]] = []
    best: BestModel | None = None

    for name, model in models.items():
        model.fit(splits.X_train, splits.y_train)
        preds = model.predict(splits.X_val)
        metrics = compute_metrics(splits.y_val, preds, average=average)
        results.append({"model": name, **metrics})
        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {"name": name, "model": model, "metrics": metrics}

    if best is None:
        raise RuntimeError(f"No models evaluated for task {task_name}.")

    pd.DataFrame(results).sort_values(by="f1", ascending=False).to_csv(
        REPORTS_DIR / f"{task_name}_validation_metrics.csv", index=False
    )
    return results, best


def evaluate_on_test(
    best_model: Pipeline,
    splits: DatasetSplits,
    average: str,
    task_name: str,
    labels: List[str] | None = None,
) -> Dict[str, float]:
    combined_X = pd.concat([splits.X_train, splits.X_val])
    combined_y = pd.concat([splits.y_train, splits.y_val])
    best_model.fit(combined_X, combined_y)
    preds = best_model.predict(splits.X_test)
    metrics = compute_metrics(splits.y_test, preds, average=average)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(REPORTS_DIR / f"{task_name}_test_metrics.csv", index=False)
    save_confusion_matrix(splits.y_test, preds, task_name, labels=labels)
    return metrics


def save_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, task_name: str, labels: List[str] | None = None
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    display_labels = labels if labels is not None else np.unique(y_true)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix - {task_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{task_name}_confusion_matrix.png", dpi=200)
    plt.close()


def run_unsupervised(df: pd.DataFrame, feature_cols: List[str]) -> None:
    imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()
    features = scaler.fit_transform(imputer.fit_transform(df[feature_cols]))

    ks = list(range(2, 9))
    inertias, silhouettes = [], []
    best_k, best_silhouette = None, -np.inf
    best_labels = None

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = kmeans.fit_predict(features)
        inertia = kmeans.inertia_
        sil = silhouette_score(features, labels)
        inertias.append(inertia)
        silhouettes.append(sil)
        if sil > best_silhouette:
            best_silhouette, best_k, best_labels = sil, k, labels

    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow method (K-Means)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ks, silhouettes, marker="o", color="darkgreen")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette scores (K-Means)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_silhouette.png", dpi=200)
    plt.close()

    if best_labels is None or best_k is None:
        raise RuntimeError("Failed to identify K-Means clusters.")

    agg = AgglomerativeClustering(n_clusters=best_k)
    agg_labels = agg.fit_predict(features)

    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    db_labels = dbscan.fit_predict(features)

    df_clusters = df.copy()
    df_clusters["kmeans_cluster"] = best_labels
    df_clusters["agg_cluster"] = agg_labels
    df_clusters["dbscan_cluster"] = db_labels

    cluster_summary = (
        df_clusters.groupby("kmeans_cluster")
        .agg(
            count=("kmeans_cluster", "size"),
            mar_mean=("mar_index", "mean"),
            mar_median=("mar_index", "median"),
        )
        .reset_index()
    )
    cluster_summary["top_species"] = (
        df_clusters.groupby("kmeans_cluster")["bacterial_species"]
        .apply(top_species_proportions)
        .reset_index(drop=True)
    )
    cluster_summary.to_csv(REPORTS_DIR / "cluster_summary.csv", index=False)

    plot_embeddings(features, df_clusters, best_labels)


def plot_embeddings(
    features: np.ndarray, df_clusters: pd.DataFrame, cluster_labels: np.ndarray
) -> None:
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_components = pca.fit_transform(features)
    pca_df = pd.DataFrame(pca_components[:, :2], columns=["pc1", "pc2"])
    pca_df["species"] = df_clusters["bacterial_species"].values
    pca_df["high_mar"] = df_clusters["high_mar"].values
    pca_df["cluster"] = cluster_labels

    plt.figure(figsize=(6, 4))
    plt.plot(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        marker="o",
    )
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_variance.png", dpi=200)
    plt.close()

    scatter_plot(pca_df, "pc1", "pc2", "species", "PCA by species", "pca_species.png")
    scatter_plot(pca_df, "pc1", "pc2", "high_mar", "PCA by MAR", "pca_mar.png")
    scatter_plot(
        pca_df, "pc1", "pc2", "cluster", "PCA by K-Means cluster", "pca_clusters.png"
    )

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=30,
        learning_rate="auto",
        init="pca",
    )
    tsne_components = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_components, columns=["tsne1", "tsne2"])
    tsne_df["species"] = df_clusters["bacterial_species"].values
    tsne_df["high_mar"] = df_clusters["high_mar"].values
    tsne_df["cluster"] = cluster_labels
    scatter_plot(
        tsne_df, "tsne1", "tsne2", "species", "t-SNE by species", "tsne_species.png"
    )
    scatter_plot(
        tsne_df, "tsne1", "tsne2", "high_mar", "t-SNE by MAR", "tsne_mar.png"
    )
    scatter_plot(
        tsne_df,
        "tsne1",
        "tsne2",
        "cluster",
        "t-SNE by K-Means cluster",
        "tsne_clusters.png",
    )

    umap_components = umap.UMAP(
        n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1
    ).fit_transform(features)
    umap_df = pd.DataFrame(umap_components, columns=["umap1", "umap2"])
    umap_df["species"] = df_clusters["bacterial_species"].values
    umap_df["high_mar"] = df_clusters["high_mar"].values
    umap_df["cluster"] = cluster_labels
    scatter_plot(
        umap_df, "umap1", "umap2", "species", "UMAP by species", "umap_species.png"
    )
    scatter_plot(
        umap_df, "umap1", "umap2", "high_mar", "UMAP by MAR", "umap_mar.png"
    )
    scatter_plot(
        umap_df,
        "umap1",
        "umap2",
        "cluster",
        "UMAP by K-Means cluster",
        "umap_clusters.png",
    )


def scatter_plot(
    df: pd.DataFrame, x: str, y: str, hue: str, title: str, filename: str
) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="tab10", s=50, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()


def mine_association_rules(df: pd.DataFrame, feature_cols: List[str]) -> None:
    resistant = (df[feature_cols] == 2).astype(bool)
    apriori_frequent = apriori(
        resistant, min_support=APRIORI_MIN_SUPPORT, use_colnames=True
    )
    apriori_rules = association_rules(
        apriori_frequent, metric="confidence", min_threshold=ASSOCIATION_MIN_CONFIDENCE
    )
    apriori_rules = apriori_rules[apriori_rules["lift"] > ASSOCIATION_MIN_LIFT]
    apriori_rules = format_rules(apriori_rules)
    apriori_rules.head(20).to_csv(REPORTS_DIR / "apriori_rules.csv", index=False)

    fpg_frequent = fpgrowth(
        resistant, min_support=FPGROWTH_MIN_SUPPORT, use_colnames=True
    )
    fpg_rules = association_rules(
        fpg_frequent, metric="confidence", min_threshold=ASSOCIATION_MIN_CONFIDENCE
    )
    fpg_rules = fpg_rules[fpg_rules["lift"] > ASSOCIATION_MIN_LIFT]
    fpg_rules = format_rules(fpg_rules)
    fpg_rules.head(20).to_csv(REPORTS_DIR / "fpgrowth_rules.csv", index=False)


def format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    formatted = rules.copy()
    formatted["antecedents"] = formatted["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    formatted["consequents"] = formatted["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    return formatted.sort_values(by="lift", ascending=False)


def save_model_artifact(model: Pipeline, feature_cols: List[str], name: str) -> None:
    artifact = {"model": model, "feature_columns": feature_cols, "encoding": ENCODING_MAP}
    joblib.dump(artifact, ARTIFACTS_DIR / f"{name}.pkl")


def run_pipeline() -> None:
    ensure_output_dirs()
    df_raw = pd.read_csv(DATA_PATH)
    cleaned, feature_cols = prepare_dataset(df_raw)
    cleaned = add_targets(cleaned)
    cleaned, species_encoder = encode_species_labels(cleaned)

    splits_mar = split_dataset(cleaned, feature_cols, target="high_mar")
    _, best_mar = evaluate_models(
        splits_mar, num_classes=2, task_name="high_mar", average="binary"
    )
    mar_test_metrics = evaluate_on_test(
        best_mar["model"], splits_mar, average="binary", task_name="high_mar"
    )
    save_model_artifact(best_mar["model"], feature_cols, "high_mar_classifier")

    splits_species = split_dataset(cleaned, feature_cols, target="species_encoded")
    _, best_species = evaluate_models(
        splits_species,
        num_classes=cleaned["species_encoded"].nunique(),
        task_name="species",
        average="macro",
    )
    species_test_metrics = evaluate_on_test(
        best_species["model"],
        splits_species,
        average="macro",
        task_name="species",
        labels=species_encoder.classes_.tolist(),
    )

    metrics_summary = pd.DataFrame(
        [
            {"task": "high_mar", **mar_test_metrics, "best_model": best_mar["name"]},
            {
                "task": "species",
                **species_test_metrics,
                "best_model": best_species["name"],
            },
        ]
    )
    metrics_summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    run_unsupervised(cleaned, feature_cols)
    mine_association_rules(cleaned, feature_cols)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    run_pipeline()
