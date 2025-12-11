"""
PHASE 2: SUPERVISED PATTERN RECOGNITION

This module handles:
- 2.1 Predict High MAR Index / MDR (6 classifiers)
- 2.2 Species Classification from Resistance Profiles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


class SupervisedClassification:
    """
    PHASE 2: Supervised Classification
    
    Train and evaluate 6 classifiers for:
    - High MAR (MDR) prediction
    - Species classification
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize supervised classification.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = {}
        self.label_encoder = None
        
    def _initialize_models(self) -> Dict:
        """
        Initialize all 6 classifiers with default hyperparameters.
        
        Note: Hyperparameters are set to reasonable defaults for AMR data:
        - Random Forest: n_estimators=100 provides good balance of accuracy and speed,
          max_depth=10 prevents overfitting on high-dimensional resistance data
        - XGBoost: Similar settings for consistency and interpretability
        - For production use, consider GridSearchCV or RandomizedSearchCV for tuning
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,  # Sufficient trees for stable feature importance
                max_depth=10,      # Prevents overfitting with many antibiotic features
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf', probability=True, random_state=self.random_state
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5
            ),
            'naive_bayes': GaussianNB()
        }
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=self.random_state, use_label_encoder=False,
                eval_metric='logloss'
            )
        
        return models
    
    def train_model(self, 
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame = None,
                    y_val: pd.Series = None) -> Dict:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        
        # Training performance
        train_pred = model.predict(X_train)
        train_results = {
            'accuracy': accuracy_score(y_train, train_pred),
            'f1': f1_score(y_train, train_pred, average='weighted')
        }
        
        # Validation performance
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_results = {
                'accuracy': accuracy_score(y_val, val_pred),
                'f1': f1_score(y_val, val_pred, average='weighted')
            }
            print(f"  Train Accuracy: {train_results['accuracy']:.4f}, "
                  f"Val Accuracy: {val_results['accuracy']:.4f}")
            return {'train': train_results, 'val': val_results}
        
        print(f"  Train Accuracy: {train_results['accuracy']:.4f}")
        return {'train': train_results}
    
    def train_all_models(self,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: pd.DataFrame = None,
                         y_val: pd.Series = None) -> Dict:
        """
        Train all 6 classifiers.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with all training results
        """
        all_results = {}
        
        # Check if labels need encoding for XGBoost
        is_string_labels = y_train.dtype == 'object' or y_train.dtype.name == 'category'
        
        if is_string_labels:
            # Create and fit label encoder
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val) if y_val is not None else None
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val
        
        for model_name in self.models.keys():
            # Use encoded labels for XGBoost
            if model_name == 'xgboost' and is_string_labels:
                results = self.train_model(model_name, X_train, y_train_encoded, 
                                          X_val, y_val_encoded)
            else:
                results = self.train_model(model_name, X_train, y_train, X_val, y_val)
            all_results[model_name] = results
        
        return all_results
    
    def evaluate_model(self,
                       model_name: str,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       is_multiclass: bool = False) -> Dict:
        """
        Evaluate a trained model on test set.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            is_multiclass: Whether this is a multi-class problem
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # For XGBoost with encoded labels, decode predictions
        y_test_eval = y_test
        y_pred_eval = y_pred
        if model_name == 'xgboost' and self.label_encoder is not None and is_multiclass:
            y_test_eval = self.label_encoder.transform(y_test)
        
        # Basic metrics
        avg_method = 'weighted' if is_multiclass else 'binary'
        
        results = {
            'accuracy': accuracy_score(y_test_eval, y_pred_eval),
            'precision': precision_score(y_test_eval, y_pred_eval, average=avg_method, zero_division=0),
            'recall': recall_score(y_test_eval, y_pred_eval, average=avg_method, zero_division=0),
            'f1': f1_score(y_test_eval, y_pred_eval, average=avg_method, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test_eval, y_pred_eval),
            'classification_report': classification_report(y_test_eval, y_pred_eval, zero_division=0),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_).flatten()
        
        self.results[model_name] = results
        
        return results
    
    def evaluate_all_models(self,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           is_multiclass: bool = False) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            is_multiclass: Whether this is a multi-class problem
            
        Returns:
            DataFrame with comparison of all models
        """
        comparison = []
        
        for model_name in self.trained_models.keys():
            results = self.evaluate_model(model_name, X_test, y_test, is_multiclass)
            
            comparison.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'f1') -> Tuple[str, object]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for selection ('accuracy', 'precision', 'recall', 'f1')
            
        Returns:
            Tuple of (model_name, model_object)
        """
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.results.items():
            score = results.get(metric, results.get('f1'))
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, self.trained_models.get(best_model_name)
    
    def get_feature_importance(self, 
                               model_name: str,
                               feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.results[model_name]
        
        if 'feature_importance' not in results:
            return pd.DataFrame()
        
        importance = results['feature_importance']
        
        # Handle multi-class case where importance might be 2D
        if len(importance.shape) > 1:
            importance = np.mean(np.abs(importance), axis=0)
        
        # Ensure lengths match
        if len(importance) != len(feature_names):
            print(f"Warning: Feature importance length ({len(importance)}) != feature names ({len(feature_names)})")
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_confusion_matrices(self,
                                class_names: List[str] = None,
                                output_path: str = None) -> plt.Figure:
        """
        Plot confusion matrices for all models.
        
        Args:
            class_names: List of class labels
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_models = len(self.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if idx < len(axes):
                cm = results['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=class_names, yticklabels=class_names)
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
                axes[idx].set_title(f'{model_name}\nF1: {results["f1"]:.3f}')
        
        # Hide unused axes
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrices to {output_path}")
        
        return fig
    
    def plot_feature_importance(self,
                                feature_names: List[str],
                                top_n: int = 15,
                                output_path: str = None) -> plt.Figure:
        """
        Plot feature importance for Random Forest and XGBoost.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        models_with_importance = ['random_forest', 'xgboost']
        available_models = [m for m in models_with_importance if m in self.results]
        
        if not available_models:
            print("No models with feature importance available")
            return None
        
        fig, axes = plt.subplots(1, len(available_models), figsize=(6 * len(available_models), 6))
        if len(available_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(available_models):
            importance_df = self.get_feature_importance(model_name, feature_names)
            
            if len(importance_df) > 0:
                top_features = importance_df.head(top_n)
                
                axes[idx].barh(range(len(top_features)), top_features['importance'].values)
                axes[idx].set_yticks(range(len(top_features)))
                axes[idx].set_yticklabels(top_features['feature'].values)
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name} - Feature Importance')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved feature importance to {output_path}")
        
        return fig
    
    def plot_roc_curves(self,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        output_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for binary classification.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model_name, model in self.trained_models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                
                # Binary classification
                if y_pred_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved ROC curves to {output_path}")
        
        return fig
    
    def save_models(self, output_dir: str = 'models', prefix: str = ''):
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
            prefix: Prefix for model filenames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filename = f'{output_dir}/{prefix}{model_name}.pkl'
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
    
    def load_model(self, filepath: str, model_name: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            model_name: Name to assign to the loaded model
        """
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"Loaded model from {filepath}")


def run_phase2(mar_splits: Dict,
               species_splits: Dict,
               output_dir: str = 'outputs') -> Dict:
    """
    Run complete Phase 2: Supervised Pattern Recognition
    
    Args:
        mar_splits: Data splits for MAR prediction from Phase 0
        species_splits: Data splits for species classification from Phase 0
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing all Phase 2 results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    
    print("=" * 60)
    print("PHASE 2: SUPERVISED PATTERN RECOGNITION")
    print("=" * 60)
    
    results = {}
    
    # 2.1 High MAR Prediction
    print("\n" + "-" * 40)
    print("2.1 Predict High MAR Index / MDR")
    print("-" * 40)
    
    feature_cols = mar_splits['feature_cols']
    
    X_train = mar_splits['train'][feature_cols]
    y_train = mar_splits['train']['high_mar']
    X_val = mar_splits['val'][feature_cols]
    y_val = mar_splits['val']['high_mar']
    X_test = mar_splits['test'][feature_cols]
    y_test = mar_splits['test']['high_mar']
    
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    
    # Train all models
    mar_classifier = SupervisedClassification()
    mar_classifier.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    mar_comparison = mar_classifier.evaluate_all_models(X_test, y_test, is_multiclass=False)
    print("\nMAR Prediction - Model Comparison:")
    print(mar_comparison.to_string())
    mar_comparison.to_csv(f'{output_dir}/mar_model_comparison.csv', index=False)
    
    # Get best model
    best_mar_model_name, best_mar_model = mar_classifier.get_best_model('f1')
    print(f"\nBest model for MAR prediction: {best_mar_model_name}")
    
    # Plot results
    mar_classifier.plot_confusion_matrices(
        class_names=['Low MAR', 'High MAR'],
        output_path=f'{output_dir}/mar_confusion_matrices.png'
    )
    
    mar_classifier.plot_feature_importance(
        feature_names=feature_cols,
        output_path=f'{output_dir}/mar_feature_importance.png'
    )
    
    mar_classifier.plot_roc_curves(X_test, y_test, f'{output_dir}/mar_roc_curves.png')
    
    # Save models
    mar_classifier.save_models(f'{output_dir}/models', prefix='mar_')
    
    results['mar_prediction'] = {
        'classifier': mar_classifier,
        'comparison': mar_comparison,
        'best_model': best_mar_model_name,
        'feature_cols': feature_cols
    }
    
    # 2.2 Species Classification
    print("\n" + "-" * 40)
    print("2.2 Species Classification from Resistance Profiles")
    print("-" * 40)
    
    feature_cols = species_splits['feature_cols']
    
    X_train = species_splits['train'][feature_cols]
    y_train = species_splits['train']['species_target']
    X_val = species_splits['val'][feature_cols]
    y_val = species_splits['val']['species_target']
    X_test = species_splits['test'][feature_cols]
    y_test = species_splits['test']['species_target']
    
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"Number of classes: {y_train.nunique()}")
    
    # Train all models
    species_classifier = SupervisedClassification()
    species_classifier.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    species_comparison = species_classifier.evaluate_all_models(X_test, y_test, is_multiclass=True)
    print("\nSpecies Classification - Model Comparison:")
    print(species_comparison.to_string())
    species_comparison.to_csv(f'{output_dir}/species_model_comparison.csv', index=False)
    
    # Get best model
    best_species_model_name, best_species_model = species_classifier.get_best_model('f1')
    print(f"\nBest model for species classification: {best_species_model_name}")
    
    # Plot results
    class_names = sorted(y_test.unique())
    species_classifier.plot_confusion_matrices(
        class_names=class_names,
        output_path=f'{output_dir}/species_confusion_matrices.png'
    )
    
    species_classifier.plot_feature_importance(
        feature_names=feature_cols,
        output_path=f'{output_dir}/species_feature_importance.png'
    )
    
    # Save models
    species_classifier.save_models(f'{output_dir}/models', prefix='species_')
    
    results['species_classification'] = {
        'classifier': species_classifier,
        'comparison': species_comparison,
        'best_model': best_species_model_name,
        'feature_cols': feature_cols,
        'class_names': class_names
    }
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Example usage
    from phase0_data_preparation import run_phase0
    
    # Run Phase 0 first
    phase0_result = run_phase0('rawdata.csv', 'outputs')
    
    # Run Phase 2
    phase2_result = run_phase2(
        phase0_result['mar_splits'],
        phase0_result['species_splits'],
        'outputs'
    )
