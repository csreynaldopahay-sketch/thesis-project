# AMR Analysis Pipeline

This repository contains the cleaned dataset (`rawdata.csv`) and a Python pipeline that implements the phases outlined in the project plan:

- **Phase 0** – Data cleaning, handling missing values, ordinal encoding of antibiotic interpretations (s=0, i=1, r=2), MAR target creation (>0.3), rare-species merging, and stratified 70/20/10 train/validation/test splits.
- **Phase 1** – Unsupervised analyses: K-means, hierarchical clustering, DBSCAN, PCA/t-SNE/UMAP embeddings, and association rule mining (Apriori + FP-Growth) on resistant phenotypes.
- **Phase 2** – Supervised modelling for high-MAR prediction and species classification across six classifiers (Random Forest, XGBoost, Logistic Regression, SVM, KNN, Naive Bayes).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

Run all phases (unsupervised + supervised):

```bash
python analysis_pipeline.py --raw-path rawdata.csv --output-dir outputs
```

Quick sanity check (Phase 0 only):

```bash
python analysis_pipeline.py --summary-only
```

You can skip heavy sections:

- `--skip-unsupervised` to omit clustering/embeddings/rule mining
- `--skip-supervised` to omit model training

Artifacts (plots, embeddings, rule tables, model comparisons, confusion matrices) are written under `outputs/`.
