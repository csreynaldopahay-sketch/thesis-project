# AMR Pattern Recognition

A comprehensive machine learning pipeline for analyzing antimicrobial resistance (AMR) patterns in bacterial isolates.

## Overview

This project implements a complete data analysis pipeline for AMR research, including:

- **Phase 0**: Data Understanding & Preparation
- **Phase 1**: Unsupervised Pattern Recognition (Clustering, Dimensionality Reduction, Association Rules)
- **Phase 2**: Supervised Pattern Recognition (MDR Prediction, Species Classification)
- **Phase 3**: Model Comparison & Interpretation
- **Phase 4**: Deployment (Streamlit & FastAPI)
- **Phase 5**: Documentation & Reporting

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd thesis-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

```bash
python main.py
```

### Run Specific Phases

```bash
# Run only data preparation
python main.py --phases 0

# Run preprocessing and unsupervised analysis
python main.py --phases 0 1

# Run all supervised learning phases
python main.py --phases 0 2 3
```

### Custom Data File

```bash
python main.py --data mydata.csv --output results/
```

### Command Line Options

```
-d, --data      Path to raw data CSV file (default: rawdata.csv)
-o, --output    Output directory (default: outputs)
-p, --phases    Specific phases to run (0-5)
```

## Project Structure

```
thesis-project/
├── main.py                     # Main pipeline entry point
├── requirements.txt            # Python dependencies
├── rawdata.csv                 # Input data (AMR susceptibility data)
├── src/
│   ├── __init__.py
│   ├── phase0_data_preparation.py   # Data cleaning & encoding
│   ├── phase1_unsupervised.py       # Clustering, PCA, t-SNE, UMAP, Association Rules
│   ├── phase2_supervised.py         # Classification models
│   ├── phase3_comparison.py         # Model comparison
│   ├── phase4_deployment.py         # Streamlit & FastAPI apps
│   └── phase5_documentation.py      # Report generation
├── outputs/                    # Generated outputs
│   ├── processed_data.csv
│   ├── mar_train.csv, mar_val.csv, mar_test.csv
│   ├── species_train.csv, species_val.csv, species_test.csv
│   ├── *.png                   # Visualization plots
│   ├── models/                 # Saved model files (.pkl)
│   ├── app.py                  # Streamlit application
│   ├── api.py                  # FastAPI application
│   └── thesis_report.md        # Generated report
└── README.md
```

## Data Format

The input CSV file should contain:

- **bacterial_species**: Species name
- **isolate_code**: Unique identifier
- **administrative_region**: Geographic region
- **sample_source**: Source of sample
- **[antibiotic]_int**: Interpretation columns (s/i/r)
- **mar_index**: Multiple Antibiotic Resistance index (optional, will be calculated)

### Antibiotic Interpretation Values

- `s` = Susceptible
- `i` = Intermediate
- `r` = Resistant

## Pipeline Phases

### Phase 0: Data Preparation

- Load and clean raw data
- Handle missing values
- Encode antibiotic interpretations (s=0, i=1, r=2)
- Create target variables (High MAR, Species)
- Split data (70% train, 20% validation, 10% test)

### Phase 1: Unsupervised Learning

- **Clustering**: K-means, Hierarchical, DBSCAN
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Association Rules**: Apriori, FP-Growth

### Phase 2: Supervised Learning

Six classifiers for each task:
- Random Forest
- XGBoost
- Logistic Regression
- SVM
- K-Nearest Neighbors
- Naive Bayes

Tasks:
- **MDR Prediction**: Binary classification (High MAR vs Low MAR)
- **Species Classification**: Multi-class classification

### Phase 3: Model Comparison

- Compare all models using accuracy, precision, recall, F1-score
- Select best model based on F1-score
- Generate feature importance analysis
- Provide biological interpretation

### Phase 4: Deployment

#### Streamlit Application

```bash
cd outputs
streamlit run app.py
```

Features:
- Manual input of resistance profiles
- File upload for batch predictions
- Interactive visualizations

#### FastAPI REST API

```bash
cd outputs
uvicorn api:app --reload
```

Endpoints:
- `GET /`: Health check
- `POST /predict/mdr`: MDR prediction
- `POST /predict/species`: Species prediction
- `POST /predict/all`: Combined predictions

### Phase 5: Documentation

Generates comprehensive report including:
- Introduction to AMR
- Dataset description
- Preprocessing workflow
- Analysis results
- Model comparison
- Deployment guide
- Discussion and conclusions

## Output Files

After running the pipeline, the following files are generated:

| File | Description |
|------|-------------|
| `processed_data.csv` | Cleaned and encoded data |
| `mar_train/val/test.csv` | Data splits for MAR prediction |
| `species_train/val/test.csv` | Data splits for species classification |
| `elbow_silhouette.png` | Optimal cluster selection plot |
| `dendrogram.png` | Hierarchical clustering dendrogram |
| `pca_variance.png` | PCA variance explained |
| `embeddings.png` | PCA, t-SNE, UMAP visualizations |
| `association_rules.csv` | Co-resistance rules |
| `mar_confusion_matrices.png` | MDR model confusion matrices |
| `species_confusion_matrices.png` | Species model confusion matrices |
| `mar_feature_importance.png` | Important antibiotics for MDR |
| `species_feature_importance.png` | Important antibiotics for species |
| `mar_roc_curves.png` | ROC curves for MDR models |
| `model_comparison.png` | Model performance comparison |
| `thesis_report.md` | Complete analysis report |
| `models/*.pkl` | Trained model files |

## Models

Trained models are saved in `outputs/models/`:

- `mar_random_forest.pkl`
- `mar_xgboost.pkl`
- `mar_logistic_regression.pkl`
- `mar_svm.pkl`
- `mar_knn.pkl`
- `mar_naive_bayes.pkl`
- `species_random_forest.pkl`
- `species_xgboost.pkl`
- (etc.)

## API Example

```python
import requests

# Predict MDR status
response = requests.post(
    "http://localhost:8000/predict/mdr",
    json={
        "antibiotics": {
            "ampicillin": "r",
            "gentamicin": "s",
            "tetracycline": "r",
            "ciprofloxacin": "i"
        }
    }
)
print(response.json())
```

## Dependencies

Core libraries:
- pandas, numpy (data manipulation)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- umap-learn (dimensionality reduction)
- mlxtend (association rules)
- matplotlib, seaborn, plotly (visualization)
- streamlit (web app)
- fastapi, uvicorn (REST API)
- joblib (model serialization)

## License

This project is part of academic research on antimicrobial resistance pattern recognition.

## Citation

If you use this pipeline in your research, please cite:

```
AMR Pattern Recognition Pipeline
Thesis Project - [Year]
```

## Contact

For questions or issues, please open a GitHub issue.
