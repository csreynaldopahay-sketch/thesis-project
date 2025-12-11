"""
AMR Pattern Recognition - Source Package

This package contains modules for analyzing antimicrobial resistance patterns.

Modules:
    phase0_data_preparation: Data cleaning, encoding, and splitting
    phase1_unsupervised: Clustering, dimensionality reduction, association rules
    phase2_supervised: Classification models for MDR and species prediction
    phase3_comparison: Model comparison and interpretation
    phase4_deployment: Streamlit and FastAPI deployment
    phase5_documentation: Report generation
"""

from .phase0_data_preparation import DataPreparation, run_phase0
from .phase1_unsupervised import (
    ClusteringAnalysis, 
    DimensionalityReduction, 
    AssociationRuleMining, 
    run_phase1
)
from .phase2_supervised import SupervisedClassification, run_phase2
from .phase3_comparison import ModelComparison, run_phase3
from .phase4_deployment import AMRPredictionPipeline, run_phase4
from .phase5_documentation import ReportGenerator, run_phase5

__version__ = '1.0.0'
__author__ = 'AMR Research Team'

__all__ = [
    'DataPreparation',
    'ClusteringAnalysis',
    'DimensionalityReduction',
    'AssociationRuleMining',
    'SupervisedClassification',
    'ModelComparison',
    'AMRPredictionPipeline',
    'ReportGenerator',
    'run_phase0',
    'run_phase1',
    'run_phase2',
    'run_phase3',
    'run_phase4',
    'run_phase5'
]
