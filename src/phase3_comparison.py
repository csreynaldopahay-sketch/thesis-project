"""
PHASE 3: MODEL COMPARISON & INTERPRETATION

This module handles:
- Compare all supervised models
- Select final model based on F1-score, precision/recall balance
- Provide biological interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelComparison:
    """
    PHASE 3: Model Comparison & Interpretation
    
    Compare all supervised models and select the best one.
    """
    
    def __init__(self):
        """Initialize model comparison."""
        self.mar_comparison = None
        self.species_comparison = None
        self.final_mar_model = None
        self.final_species_model = None
        
    def compare_models(self, 
                       mar_results: Dict,
                       species_results: Dict) -> Dict:
        """
        Compare all models for both tasks.
        
        Args:
            mar_results: Results from MAR prediction
            species_results: Results from species classification
            
        Returns:
            Dictionary with comparison results
        """
        self.mar_comparison = mar_results['comparison']
        self.species_comparison = species_results['comparison']
        
        return {
            'mar_comparison': self.mar_comparison,
            'species_comparison': self.species_comparison
        }
    
    def select_best_model(self,
                          comparison_df: pd.DataFrame,
                          primary_metric: str = 'F1-Score',
                          secondary_metric: str = 'Precision') -> str:
        """
        Select the best model based on specified metrics.
        
        Args:
            comparison_df: DataFrame with model comparison results
            primary_metric: Primary metric for selection
            secondary_metric: Secondary metric for tie-breaking
            
        Returns:
            Name of the best model
        """
        # Sort by primary metric, then secondary
        sorted_df = comparison_df.sort_values(
            by=[primary_metric, secondary_metric],
            ascending=[False, False]
        )
        
        best_model = sorted_df.iloc[0]['Model']
        
        return best_model
    
    def generate_comparison_table(self, 
                                  comparison_df: pd.DataFrame,
                                  task_name: str) -> pd.DataFrame:
        """
        Generate a formatted comparison table.
        
        Args:
            comparison_df: DataFrame with model comparison results
            task_name: Name of the task (for display)
            
        Returns:
            Formatted DataFrame
        """
        table = comparison_df.copy()
        
        # Round numeric columns
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for col in numeric_cols:
            if col in table.columns:
                table[col] = table[col].apply(lambda x: f"{x:.4f}")
        
        # Add ranking
        table['Rank'] = range(1, len(table) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Model'] + numeric_cols
        table = table[[c for c in cols if c in table.columns]]
        
        return table
    
    def plot_model_comparison(self,
                              mar_comparison: pd.DataFrame,
                              species_comparison: pd.DataFrame,
                              output_path: str = None) -> plt.Figure:
        """
        Plot model comparison for both tasks.
        
        Args:
            mar_comparison: MAR prediction comparison
            species_comparison: Species classification comparison
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # MAR Prediction
        x = np.arange(len(mar_comparison))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i * width, mar_comparison[metric], width, label=metric)
        
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Score')
        axes[0].set_title('MAR/MDR Prediction - Model Comparison')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(mar_comparison['Model'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        # Species Classification
        x = np.arange(len(species_comparison))
        
        for i, metric in enumerate(metrics):
            axes[1].bar(x + i * width, species_comparison[metric], width, label=metric)
        
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Species Classification - Model Comparison')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(species_comparison['Model'], rotation=45, ha='right')
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved model comparison plot to {output_path}")
        
        return fig
    
    def generate_interpretation(self,
                                mar_results: Dict,
                                species_results: Dict,
                                feature_names: List[str]) -> Dict:
        """
        Generate biological interpretation of results.
        
        Args:
            mar_results: Results from MAR prediction
            species_results: Results from species classification
            feature_names: List of antibiotic feature names
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {
            'mar_prediction': [],
            'species_classification': [],
            'key_antibiotics': [],
            'recommendations': []
        }
        
        # MAR Prediction interpretations
        best_mar = mar_results['best_model']
        mar_comparison = mar_results['comparison']
        best_mar_f1 = mar_comparison[mar_comparison['Model'] == best_mar]['F1-Score'].values[0]
        
        interpretations['mar_prediction'].append(
            f"The {best_mar} model achieves the best performance for MDR prediction "
            f"with an F1-score of {best_mar_f1:.4f}."
        )
        
        if best_mar_f1 > 0.8:
            interpretations['mar_prediction'].append(
                "The model shows strong predictive power, suggesting that antibiotic "
                "resistance patterns are highly indicative of MDR status."
            )
        elif best_mar_f1 > 0.6:
            interpretations['mar_prediction'].append(
                "The model shows moderate predictive power. Additional features "
                "(e.g., genetic markers) might improve predictions."
            )
        else:
            interpretations['mar_prediction'].append(
                "The model shows limited predictive power, possibly due to class "
                "imbalance or complex resistance patterns."
            )
        
        # Species Classification interpretations
        best_species = species_results['best_model']
        species_comparison = species_results['comparison']
        best_species_f1 = species_comparison[species_comparison['Model'] == best_species]['F1-Score'].values[0]
        
        interpretations['species_classification'].append(
            f"The {best_species} model achieves the best performance for species "
            f"classification with an F1-score of {best_species_f1:.4f}."
        )
        
        if best_species_f1 > 0.7:
            interpretations['species_classification'].append(
                "This suggests distinct antibiotic resistance profiles among bacterial "
                "species, which could be exploited for rapid species identification."
            )
        else:
            interpretations['species_classification'].append(
                "Lower performance may indicate overlapping resistance profiles among "
                "species or insufficient training data for some classes."
            )
        
        # Key antibiotics interpretation
        interpretations['key_antibiotics'].append(
            "Feature importance analysis can identify which antibiotics are most "
            "predictive of MDR status and species identity."
        )
        
        # Recommendations
        interpretations['recommendations'] = [
            "1. Consider ensemble methods combining the top-performing models.",
            "2. Investigate misclassified samples for patterns.",
            "3. Validate findings with external datasets.",
            "4. Consider temporal validation (train on older, test on newer data).",
            "5. Explore the biological basis of important features."
        ]
        
        return interpretations
    
    def generate_report(self,
                        mar_results: Dict,
                        species_results: Dict,
                        output_path: str = None) -> str:
        """
        Generate a comprehensive comparison report.
        
        Args:
            mar_results: Results from MAR prediction
            species_results: Results from species classification
            output_path: Path to save report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("PHASE 3: MODEL COMPARISON & INTERPRETATION REPORT")
        report.append("=" * 70)
        
        # MAR Prediction
        report.append("\n1. HIGH MAR (MDR) PREDICTION\n")
        report.append("-" * 40)
        mar_table = self.generate_comparison_table(mar_results['comparison'], 'MAR Prediction')
        report.append(mar_table.to_string(index=False))
        report.append(f"\nBest Model: {mar_results['best_model']}")
        
        # Species Classification
        report.append("\n\n2. SPECIES CLASSIFICATION\n")
        report.append("-" * 40)
        species_table = self.generate_comparison_table(species_results['comparison'], 'Species Classification')
        report.append(species_table.to_string(index=False))
        report.append(f"\nBest Model: {species_results['best_model']}")
        
        # Selection Criteria
        report.append("\n\n3. MODEL SELECTION CRITERIA\n")
        report.append("-" * 40)
        report.append("""
The final model selection is based on:
- F1-score (primary): Best balance of precision and recall
- Precision/Recall balance: Important for AMR data with potential class imbalance
- Biological plausibility: Results should make biological sense
- Interpretability: Ability to understand model decisions
        """)
        
        # Final Selections
        report.append("\n4. FINAL MODEL SELECTIONS\n")
        report.append("-" * 40)
        report.append(f"MAR/MDR Prediction: {mar_results['best_model']}")
        report.append(f"Species Classification: {species_results['best_model']}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Saved comparison report to {output_path}")
        
        return report_text


def run_phase3(mar_results: Dict,
               species_results: Dict,
               output_dir: str = 'outputs') -> Dict:
    """
    Run complete Phase 3: Model Comparison & Interpretation
    
    Args:
        mar_results: Results from Phase 2 MAR prediction
        species_results: Results from Phase 2 species classification
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing all Phase 3 results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 3: MODEL COMPARISON & INTERPRETATION")
    print("=" * 60)
    
    comparison = ModelComparison()
    
    # Compare models
    comparison.compare_models(mar_results, species_results)
    
    # Select best models
    best_mar = comparison.select_best_model(mar_results['comparison'])
    best_species = comparison.select_best_model(species_results['comparison'])
    
    print(f"\nBest MAR Prediction Model: {best_mar}")
    print(f"Best Species Classification Model: {best_species}")
    
    # Plot comparison
    comparison.plot_model_comparison(
        mar_results['comparison'],
        species_results['comparison'],
        f'{output_dir}/model_comparison.png'
    )
    
    # Generate interpretation
    interpretations = comparison.generate_interpretation(
        mar_results,
        species_results,
        mar_results['feature_cols']
    )
    
    print("\nInterpretations:")
    for category, items in interpretations.items():
        print(f"\n{category.upper()}:")
        for item in items:
            print(f"  - {item}")
    
    # Generate report
    report = comparison.generate_report(
        mar_results,
        species_results,
        f'{output_dir}/phase3_report.txt'
    )
    
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    
    return {
        'comparison': comparison,
        'best_mar_model': best_mar,
        'best_species_model': best_species,
        'interpretations': interpretations,
        'report': report
    }


if __name__ == "__main__":
    # Example usage
    from phase0_data_preparation import run_phase0
    from phase2_supervised import run_phase2
    
    # Run previous phases
    phase0_result = run_phase0('rawdata.csv', 'outputs')
    phase2_result = run_phase2(
        phase0_result['mar_splits'],
        phase0_result['species_splits'],
        'outputs'
    )
    
    # Run Phase 3
    phase3_result = run_phase3(
        phase2_result['mar_prediction'],
        phase2_result['species_classification'],
        'outputs'
    )
