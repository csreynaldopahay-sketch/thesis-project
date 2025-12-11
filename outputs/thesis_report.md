
---
Generated: 2025-12-11 01:24:21
AMR Pattern Recognition Analysis Report
---

# AMR PATTERN RECOGNITION ANALYSIS REPORT

## 1. Introduction to Antimicrobial Resistance (AMR)

Antimicrobial resistance (AMR) is a critical global health challenge where bacteria, 
viruses, fungi, and parasites evolve to resist the drugs designed to kill them. This 
phenomenon leads to:

- Increased mortality and morbidity
- Longer hospital stays
- Higher medical costs
- Limited treatment options

### 1.1 Pattern Recognition in AMR Analysis

Pattern recognition techniques offer powerful tools for:
- Identifying resistance patterns
- Predicting multi-drug resistance (MDR)
- Classifying bacterial species based on resistance profiles
- Discovering co-resistance associations

This analysis employs both unsupervised and supervised machine learning methods to 
extract meaningful patterns from antimicrobial susceptibility testing (AST) data.


## 2. Dataset Description

### 2.1 Data Overview

- **Total Samples**: 526
- **Number of Species**: 13
- **Number of Regions**: 3
- **Number of Antibiotics Tested**: 23

### 2.2 MAR Index Statistics

The Multiple Antibiotic Resistance (MAR) Index is calculated as:

MAR = (Number of antibiotics to which isolate is resistant) / (Total number of antibiotics tested)

- **Mean MAR Index**: 0.1116
- **Standard Deviation**: 0.1071
- **Min**: 0.0000
- **Max**: 0.7143

### 2.3 Species Distribution

The dataset includes isolates from multiple bacterial species, with the most common being:

- escherichia_coli: 228
- klebsiella_pneumoniae_ssp_pneumoniae: 146
- enterobacter_cloacae_complex: 69
- enterobacter_aerogenes: 23
- salmonella_group: 22
- pseudomonas_aeruginosa: 17
- vibrio_fluvialis: 8
- vibrio_cholerae: 6
- klebsiella_pneumoniae_ssp_ozaenae: 2
- salmonella_enterica_spp_diarizonae: 2

### 2.4 MDR Classification Distribution

Using MAR > 0.3 as the threshold for high MAR (MDR):

- 0: 485
- 1: 41


## 3. Preprocessing Workflow

### 3.1 Data Cleaning

The following cleaning steps were applied:

1. **Remove irrelevant columns**: MIC values were removed, keeping only interpretation columns
2. **Handle missing values**: Rows with >70% missing antibiotic data were removed
3. **Standardize interpretations**: Values were standardized to s (susceptible), i (intermediate), r (resistant)
4. **Clean labels**: Species names, regions, and other categorical variables were standardized

### 3.2 Feature Encoding

Antibiotic interpretations were encoded using ordinal encoding:
- s (susceptible) = 0
- i (intermediate) = 1
- r (resistant) = 2

### 3.3 Target Variable Creation

Two supervised learning targets were created:

1. **High MAR Classification**:
   - Threshold: MAR > 0.3
   - Binary: High MAR = 1, Low MAR = 0

2. **Species Classification**:
   - Species with <10 samples merged into "Other" category

### 3.4 Data Splitting

Data was split using stratified sampling:
- **Training Set**: 70%
- **Validation Set**: 20%
- **Test Set**: 10%


## 4. Unsupervised Analysis Results

### 4.1 Clustering Analysis

#### K-Means Clustering
- **Optimal K**: 7 (determined by silhouette score)
- Clusters reveal distinct groups of isolates with similar resistance patterns

#### Hierarchical Clustering
- Dendrogram analysis reveals hierarchical relationships between resistance profiles
- Useful for identifying nested cluster structures

#### DBSCAN
- Identifies outliers (rare resistance phenotypes)
- Useful for detecting unusual resistance combinations

### 4.2 Dimensionality Reduction

#### PCA (Principal Component Analysis)
- Linear projection preserving maximum variance
- First 2-3 components typically explain significant variance

#### t-SNE
- Non-linear visualization revealing local cluster structure
- Useful for identifying distinct resistance phenotype groups

#### UMAP
- Preserves both local and global structure
- Often provides best visualization for AMR phenotype clustering

### 4.3 Association Rule Mining

Co-resistance patterns identified using FP-Growth algorithm:
- Minimum support: 0.02
- Minimum confidence: 0.5
- Minimum lift: 1.0

Key findings reveal which antibiotic resistances tend to co-occur, potentially 
indicating shared resistance mechanisms or genetic linkage.


## 5. Supervised Model Comparison

### 5.1 MDR (High MAR) Prediction

Six classifiers were trained and evaluated for MDR prediction:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | See outputs | See outputs | See outputs | See outputs |
| XGBoost | See outputs | See outputs | See outputs | See outputs |
| Logistic Regression | See outputs | See outputs | See outputs | See outputs |
| SVM | See outputs | See outputs | See outputs | See outputs |
| KNN | See outputs | See outputs | See outputs | See outputs |
| Naive Bayes | See outputs | See outputs | See outputs | See outputs |

**Feature Importance Analysis**: Random Forest and XGBoost identify which antibiotics 
are most predictive of MDR status.

### 5.2 Species Classification

The same six classifiers were applied for multi-class species prediction:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | See outputs | See outputs | See outputs | See outputs |
| XGBoost | See outputs | See outputs | See outputs | See outputs |
| Logistic Regression | See outputs | See outputs | See outputs | See outputs |
| SVM | See outputs | See outputs | See outputs | See outputs |
| KNN | See outputs | See outputs | See outputs | See outputs |
| Naive Bayes | See outputs | See outputs | See outputs | See outputs |

**Per-Class Performance**: Detailed confusion matrices show classification performance 
for each bacterial species.


## 6. Deployment Procedure

### 6.1 Model Serialization

Best-performing models are saved using joblib:
- `mar_random_forest.pkl` - MDR prediction model
- `species_random_forest.pkl` - Species classification model

### 6.2 Web Application

A Streamlit web application (`app.py`) provides:
- Manual input of resistance profiles
- File upload for batch predictions
- Interactive visualizations
- Real-time MDR and species predictions

### 6.3 REST API

A FastAPI application (`api.py`) provides:
- `/predict/mdr` - MDR prediction endpoint
- `/predict/species` - Species prediction endpoint
- `/predict/all` - Combined prediction endpoint

### 6.4 Deployment Options

Recommended deployment platforms:
1. **HuggingFace Spaces** - Free hosting for Streamlit apps
2. **Heroku** - Easy deployment with Docker support
3. **AWS/GCP/Azure** - Scalable cloud deployment

### 6.5 Application Features

The deployed application offers:
- Upload resistance profile
- Get MDR prediction with confidence score
- Get species prediction with top alternatives
- View feature importance
- Interactive UMAP visualization


## 7. Discussion and Conclusions

### 7.1 Key Findings

1. **Clustering Analysis** reveals distinct groups of bacterial isolates with similar 
   resistance patterns, potentially indicating common resistance mechanisms or sources.

2. **Association Rules** identify co-resistance patterns that may reflect:
   - Shared genetic elements (plasmids, transposons)
   - Co-selection pressure from antibiotic combinations
   - Intrinsic resistance mechanisms

3. **MDR Prediction** models achieve reasonable accuracy, suggesting resistance 
   patterns are indicative of overall MDR status.

4. **Species Classification** performance varies, with some species having distinctive 
   resistance profiles while others show overlapping patterns.

### 7.2 Implications for AMR Surveillance

- Pattern recognition can supplement traditional AST methods
- Automated MDR screening based on partial resistance data
- Rapid species identification from resistance profiles alone
- Identification of emerging resistance patterns through clustering

### 7.3 Conclusions

This study demonstrates the utility of machine learning pattern recognition for:
1. Understanding AMR epidemiology
2. Predicting MDR status
3. Classifying bacterial species from phenotypic data
4. Identifying co-resistance relationships

The deployed models provide a practical tool for rapid AMR assessment.


## 8. Limitations and Future Work

### 8.1 Limitations

1. **Dataset Size**: Limited samples may affect model generalizability
2. **Class Imbalance**: Some species/resistance patterns underrepresented
3. **Temporal Bias**: Data from specific time period may not reflect current trends
4. **Geographic Scope**: Regional data may not generalize to other areas
5. **Feature Limitations**: Only phenotypic resistance data, no genetic information

### 8.2 Future Work

1. **Expand Dataset**: Include more samples, species, and time periods
2. **Integrate Genomic Data**: Combine phenotypic and genotypic features
3. **Deep Learning**: Explore neural network architectures
4. **Temporal Analysis**: Model resistance trend evolution
5. **Transfer Learning**: Adapt models to new geographic regions
6. **Real-time Integration**: Connect to laboratory information systems
7. **Mobile Application**: Develop field-deployable prediction tools

### 8.3 Recommendations for Implementation

1. Validate models on independent datasets before clinical use
2. Regularly retrain models with updated resistance data
3. Combine with expert clinical judgment
4. Monitor model performance over time
