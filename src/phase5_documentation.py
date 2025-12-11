"""
PHASE 5: DOCUMENTATION & REPORTING

This module provides:
- Generate comprehensive analysis report
- Create visualizations summary
- Export results in various formats
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """
    PHASE 5: Documentation & Reporting
    
    Generate comprehensive thesis report and documentation.
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory containing analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.report_sections = {}
        
    def generate_introduction(self) -> str:
        """Generate introduction section."""
        return """
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
"""
    
    def generate_dataset_description(self, stats: Dict) -> str:
        """
        Generate dataset description section.
        
        Args:
            stats: Statistics dictionary from Phase 0
        """
        return f"""
## 2. Dataset Description

### 2.1 Data Overview

- **Total Samples**: {stats.get('total_samples', 'N/A')}
- **Number of Species**: {stats.get('num_species', 'N/A')}
- **Number of Regions**: {stats.get('num_regions', 'N/A')}
- **Number of Antibiotics Tested**: {stats.get('num_antibiotics', 'N/A')}

### 2.2 MAR Index Statistics

The Multiple Antibiotic Resistance (MAR) Index is calculated as:

MAR = (Number of antibiotics to which isolate is resistant) / (Total number of antibiotics tested)

- **Mean MAR Index**: {stats.get('mar_index_stats', {}).get('mean', 'N/A'):.4f}
- **Standard Deviation**: {stats.get('mar_index_stats', {}).get('std', 'N/A'):.4f}
- **Min**: {stats.get('mar_index_stats', {}).get('min', 'N/A'):.4f}
- **Max**: {stats.get('mar_index_stats', {}).get('max', 'N/A'):.4f}

### 2.3 Species Distribution

The dataset includes isolates from multiple bacterial species, with the most common being:

{self._format_distribution(stats.get('species_distribution', {}))}

### 2.4 MDR Classification Distribution

Using MAR > 0.3 as the threshold for high MAR (MDR):

{self._format_distribution(stats.get('high_mar_distribution', {}))}
"""
    
    def _format_distribution(self, dist: Dict, top_n: int = 10) -> str:
        """Format a distribution dictionary for display."""
        if not dist:
            return "N/A"
        
        lines = []
        for i, (key, value) in enumerate(dist.items()):
            if i >= top_n:
                break
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def generate_preprocessing_section(self) -> str:
        """Generate preprocessing workflow section."""
        return """
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
"""
    
    def generate_unsupervised_section(self, phase1_results: Dict = None) -> str:
        """
        Generate unsupervised analysis results section.
        
        Args:
            phase1_results: Results dictionary from Phase 1
        """
        optimal_k = phase1_results.get('clustering', {}).get('optimal_k', 3) if phase1_results else 3
        
        return f"""
## 4. Unsupervised Analysis Results

### 4.1 Clustering Analysis

#### K-Means Clustering
- **Optimal K**: {optimal_k} (determined by silhouette score)
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
"""
    
    def generate_supervised_section(self, phase2_results: Dict = None) -> str:
        """
        Generate supervised model comparison section.
        
        Args:
            phase2_results: Results dictionary from Phase 2
        """
        return """
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
"""
    
    def generate_deployment_section(self) -> str:
        """Generate deployment procedure section."""
        return """
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
"""
    
    def generate_discussion_section(self) -> str:
        """Generate discussion and conclusions section."""
        return """
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
"""
    
    def generate_limitations_section(self) -> str:
        """Generate limitations and future work section."""
        return """
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
"""
    
    def generate_full_report(self, 
                            stats: Dict = None,
                            phase1_results: Dict = None,
                            phase2_results: Dict = None) -> str:
        """
        Generate complete thesis report.
        
        Args:
            stats: Statistics from Phase 0
            phase1_results: Results from Phase 1
            phase2_results: Results from Phase 2
            
        Returns:
            Complete report as string
        """
        sections = [
            self.generate_introduction(),
            self.generate_dataset_description(stats or {}),
            self.generate_preprocessing_section(),
            self.generate_unsupervised_section(phase1_results),
            self.generate_supervised_section(phase2_results),
            self.generate_deployment_section(),
            self.generate_discussion_section(),
            self.generate_limitations_section(),
        ]
        
        # Add metadata
        header = f"""
---
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AMR Pattern Recognition Analysis Report
---
"""
        
        return header + "\n".join(sections)
    
    def save_report(self, report: str, filename: str = 'thesis_report.md'):
        """
        Save report to file.
        
        Args:
            report: Report content
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Saved report to {filepath}")
        
        return str(filepath)
    
    def export_results_summary(self, results: Dict, filename: str = 'results_summary.json'):
        """
        Export results summary as JSON.
        
        Args:
            results: Results dictionary to export
            filename: Output filename
        """
        from datetime import datetime
        
        # Convert to JSON-serializable format
        def convert(obj):
            """Convert non-JSON-serializable objects to serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                # For custom objects, try to get their dict representation
                return f"<{type(obj).__name__}>"
            else:
                # For any other type, convert to string representation
                try:
                    return str(obj)
                except Exception:
                    return f"<non-serializable: {type(obj).__name__}>"
        
        filepath = self.output_dir / filename
        
        # Extract serializable parts
        summary = {}
        
        if 'stats' in results:
            summary['statistics'] = results['stats']
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, default=convert, indent=2)
        
        print(f"Saved results summary to {filepath}")
        
        return str(filepath)


def run_phase5(stats: Dict = None,
               phase1_results: Dict = None,
               phase2_results: Dict = None,
               output_dir: str = 'outputs') -> Dict:
    """
    Run complete Phase 5: Documentation & Reporting
    
    Args:
        stats: Statistics from Phase 0
        phase1_results: Results from Phase 1
        phase2_results: Results from Phase 2
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with report paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 5: DOCUMENTATION & REPORTING")
    print("=" * 60)
    
    generator = ReportGenerator(output_dir)
    
    # Generate report
    report = generator.generate_full_report(stats, phase1_results, phase2_results)
    
    # Save report
    report_path = generator.save_report(report, 'thesis_report.md')
    
    # Export results summary
    results = {'stats': stats} if stats else {}
    summary_path = generator.export_results_summary(results, 'results_summary.json')
    
    print("\n" + "-" * 40)
    print("Documentation Created")
    print("-" * 40)
    print(f"""
Files created:
- {report_path} (Main thesis report)
- {summary_path} (Results summary JSON)

Output directory contents:
- processed_data.csv
- mar_train/val/test.csv
- species_train/val/test.csv
- *.png (visualization plots)
- models/*.pkl (trained models)
    """)
    
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)
    
    return {
        'report_path': report_path,
        'summary_path': summary_path
    }


if __name__ == "__main__":
    run_phase5(output_dir='outputs')
