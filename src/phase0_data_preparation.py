"""
PHASE 0: DATA UNDERSTANDING & PREPARATION

This module handles:
- 0.1 Data Cleaning
- 0.2 Feature Encoding
- 0.3 Creating supervised target variables
- 0.4 Data Splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class DataPreparation:
    """Class for data cleaning, encoding, and preparation for AMR analysis."""
    
    def __init__(self, filepath: str):
        """
        Initialize with path to raw data CSV.
        
        Args:
            filepath: Path to the rawdata.csv file
        """
        self.filepath = filepath
        self.raw_data = None
        self.cleaned_data = None
        self.encoded_data = None
        
        # Define antibiotic interpretation columns (ending with _int)
        self.antibiotic_int_cols = []
        
        # Define metadata columns (non-antibiotic columns)
        self.metadata_cols = [
            'bacterial_species', 'isolate_code', 'administrative_region',
            'national_site', 'local_site', 'sample_source', 'replicate', 
            'colony', 'esbl', 'scored_resistance', 'num_antibiotics_tested', 
            'mar_index'
        ]
        
        # Ordinal encoding map for antibiotic susceptibility interpretations
        # s (susceptible) = 0: Bacteria killed/inhibited by standard antibiotic dose
        # i (intermediate) = 1: May respond to higher doses or site-specific concentrations  
        # r (resistant) = 2: Bacteria survive standard antibiotic dose
        # This encoding reflects increasing resistance levels (0 < 1 < 2)
        self.encoding_map = {'s': 0, 'i': 1, 'r': 2}
        
    def load_data(self) -> pd.DataFrame:
        """Load the raw data from CSV."""
        self.raw_data = pd.read_csv(self.filepath)
        print(f"Loaded data with shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        return self.raw_data
    
    def _identify_antibiotic_columns(self) -> Tuple[List[str], List[str]]:
        """
        Identify antibiotic MIC and interpretation columns.
        
        Returns:
            Tuple of (mic_columns, int_columns)
        """
        mic_cols = [col for col in self.raw_data.columns if col.endswith('_mic')]
        int_cols = [col for col in self.raw_data.columns if col.endswith('_int')]
        self.antibiotic_int_cols = int_cols
        return mic_cols, int_cols
    
    def clean_data(self, 
                   max_missing_ratio: float = 0.7,
                   min_species_samples: int = 5) -> pd.DataFrame:
        """
        PHASE 0.1: Data Cleaning
        
        - Remove irrelevant columns (MIC values, keep only interpretations)
        - Handle missing values in antibiotic interpretation columns
        - Treat 'i' (intermediate) as its own category
        - Drop rows with too many missing values
        - Ensure consistency in species labels, region names, etc.
        
        Args:
            max_missing_ratio: Maximum ratio of missing values allowed per row
            min_species_samples: Minimum samples required per species
            
        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            self.load_data()
            
        df = self.raw_data.copy()
        
        # Identify antibiotic columns
        mic_cols, int_cols = self._identify_antibiotic_columns()
        print(f"Found {len(mic_cols)} MIC columns and {len(int_cols)} interpretation columns")
        
        # Remove MIC columns (keep interpretation columns only for analysis)
        df = df.drop(columns=mic_cols, errors='ignore')
        print(f"Removed MIC columns. New shape: {df.shape}")
        
        # Clean antibiotic interpretation values
        # Standardize values: lowercase, strip whitespace, handle variations
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                # Handle variations like '*r', 'r*', etc.
                df[col] = df[col].replace({
                    '*r': 'r', 'r*': 'r', '*s': 's', 's*': 's', '*i': 'i', 'i*': 'i',
                    'nan': np.nan, '': np.nan, 'none': np.nan
                })
                # Ensure only s, i, r are valid values
                df.loc[~df[col].isin(['s', 'i', 'r']), col] = np.nan
        
        # Calculate missing ratio per row for antibiotic columns
        antibiotic_cols_in_df = [col for col in int_cols if col in df.columns]
        df['missing_ratio'] = df[antibiotic_cols_in_df].isna().sum(axis=1) / len(antibiotic_cols_in_df)
        
        # Drop rows with too many missing values
        initial_rows = len(df)
        df = df[df['missing_ratio'] <= max_missing_ratio]
        dropped_rows = initial_rows - len(df)
        print(f"Dropped {dropped_rows} rows with missing ratio > {max_missing_ratio}")
        df = df.drop(columns=['missing_ratio'])
        
        # Clean species labels: lowercase, replace spaces with underscores
        if 'bacterial_species' in df.columns:
            df['bacterial_species'] = df['bacterial_species'].astype(str).str.lower().str.strip()
            df['bacterial_species'] = df['bacterial_species'].str.replace(' ', '_')
            # Remove empty species
            df = df[df['bacterial_species'].notna() & (df['bacterial_species'] != '')]
            df = df[df['bacterial_species'] != 'nan']
        
        # Clean region names
        if 'administrative_region' in df.columns:
            df['administrative_region'] = df['administrative_region'].astype(str).str.lower().str.strip()
        
        # Clean other categorical columns
        for col in ['national_site', 'local_site', 'sample_source']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        
        # Ensure numeric columns are numeric
        for col in ['scored_resistance', 'num_antibiotics_tested', 'mar_index', 'replicate', 'colony']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.cleaned_data = df
        print(f"Cleaned data shape: {df.shape}")
        print(f"Species distribution:\n{df['bacterial_species'].value_counts()}")
        
        return df
    
    def encode_features(self) -> pd.DataFrame:
        """
        PHASE 0.2: Feature Encoding
        
        Convert antibiotic interpretations (s, i, r) into numeric form:
        - Ordinal encoding: s=0, i=1, r=2
        
        Returns:
            DataFrame with encoded antibiotic features
        """
        if self.cleaned_data is None:
            self.clean_data()
            
        df = self.cleaned_data.copy()
        
        # Get antibiotic interpretation columns
        antibiotic_cols = [col for col in self.antibiotic_int_cols if col in df.columns]
        
        # Create encoded columns
        for col in antibiotic_cols:
            encoded_col = col.replace('_int', '_encoded')
            df[encoded_col] = df[col].map(self.encoding_map)
        
        self.encoded_data = df
        print(f"Created {len(antibiotic_cols)} encoded antibiotic features")
        
        return df
    
    def create_target_variables(self, 
                                mar_threshold: float = 0.17,
                                min_species_samples: int = 10,
                                other_label: str = 'other') -> pd.DataFrame:
        """
        PHASE 0.3: Create the supervised target variables
        
        1. High MAR index prediction (MDR - Multi-Drug Resistance):
           - Set threshold (MAR > 0.17) - aligned with traditional MDR definition
           - A bacterium is considered multi-drug resistant if MAR > 0.17
           - This threshold corresponds to resistance to ~4 antibiotics out of 22-23 tested
           - Convert into binary target (MDR/High MAR = 1, Non-MDR/Low MAR = 0)
           
        2. Species classification:
           - Keep only species with enough samples
           - Merge very rare species into "Other" if needed
        
        Args:
            mar_threshold: Threshold for high MAR classification
            min_species_samples: Minimum samples required per species
            other_label: Label for merged rare species
            
        Returns:
            DataFrame with target variables added
        """
        if self.encoded_data is None:
            self.encode_features()
            
        df = self.encoded_data.copy()
        
        # 1. Create High MAR target variable (MDR classification)
        if 'mar_index' in df.columns:
            df['high_mar'] = (df['mar_index'] > mar_threshold).astype(int)
            high_mar_count = df['high_mar'].sum()
            low_mar_count = len(df) - high_mar_count
            print(f"MDR Classification (MAR > {mar_threshold}):")
            print(f"  MDR (High MAR): {high_mar_count} samples")
            print(f"  Non-MDR (Low MAR): {low_mar_count} samples")
        else:
            print("Warning: mar_index column not found. Cannot create high_mar target.")
            df['high_mar'] = np.nan
        
        # 2. Create species classification target
        if 'bacterial_species' in df.columns:
            species_counts = df['bacterial_species'].value_counts()
            
            # Identify rare species (fewer than min_species_samples)
            rare_species = species_counts[species_counts < min_species_samples].index.tolist()
            
            # Create species_target column
            df['species_target'] = df['bacterial_species'].copy()
            df.loc[df['bacterial_species'].isin(rare_species), 'species_target'] = other_label
            
            print(f"\nSpecies classification:")
            print(f"Original species count: {len(species_counts)}")
            print(f"Rare species merged into 'other': {len(rare_species)}")
            print(f"Final species distribution:\n{df['species_target'].value_counts()}")
        else:
            print("Warning: bacterial_species column not found.")
            df['species_target'] = np.nan
        
        self.encoded_data = df
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of encoded antibiotic feature column names."""
        return [col for col in self.encoded_data.columns if col.endswith('_encoded')]
    
    def split_data(self, 
                   target_col: str,
                   test_size: float = 0.1,
                   val_size: float = 0.2,
                   random_state: int = 42,
                   stratify: bool = True) -> Dict[str, pd.DataFrame]:
        """
        PHASE 0.4: Data Splitting
        
        Split data into:
        - 70% Training
        - 20% Validation  
        - 10% Final Test
        
        with stratification (if specified)
        
        Args:
            target_col: Name of target column for stratification
            test_size: Proportion for final test set (default 0.1 = 10%)
            val_size: Proportion for validation set (default 0.2 = 20%)
            random_state: Random seed for reproducibility
            stratify: Whether to use stratified splitting
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames and feature columns
        """
        if self.encoded_data is None:
            raise ValueError("Data not encoded yet. Run create_target_variables() first.")
        
        df = self.encoded_data.copy()
        
        # Remove rows with missing target
        df_valid = df[df[target_col].notna()].copy()
        print(f"Valid samples for target '{target_col}': {len(df_valid)}")
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        
        # Remove samples with all NaN features
        df_valid = df_valid.dropna(subset=feature_cols, how='all')
        print(f"Samples after removing all-NaN features: {len(df_valid)}")
        
        # Fill remaining NaN in features with mode (most common value) 
        for col in feature_cols:
            if df_valid[col].isna().any():
                mode_value = df_valid[col].mode()
                if len(mode_value) > 0:
                    df_valid[col] = df_valid[col].fillna(mode_value[0])
                else:
                    df_valid[col] = df_valid[col].fillna(0)
        
        # Prepare for splitting
        X = df_valid[feature_cols]
        y = df_valid[target_col]
        
        # First split: separate test set (10%)
        stratify_col = y if stratify else None
        
        # Handle stratification with small classes
        if stratify:
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            if min_class_count < 2:
                print(f"Warning: Class with only {min_class_count} sample(s). Disabling stratification.")
                stratify_col = None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
        
        # Adjust validation size for the remaining data
        # val_size is proportion of original data, need to calculate proportion of remaining
        val_size_adjusted = val_size / (1 - test_size)
        
        stratify_temp = y_temp if stratify and stratify_col is not None else None
        if stratify_temp is not None:
            class_counts_temp = y_temp.value_counts()
            if class_counts_temp.min() < 2:
                stratify_temp = None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=stratify_temp
        )
        
        # Create DataFrames with all columns
        train_df = df_valid.loc[X_train.index].copy()
        val_df = df_valid.loc[X_val.index].copy()
        test_df = df_valid.loc[X_test.index].copy()
        
        print(f"\nData split summary:")
        print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df_valid)*100:.1f}%)")
        print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df_valid)*100:.1f}%)")
        print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df_valid)*100:.1f}%)")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'full_processed': df_valid
        }
    
    def get_summary_statistics(self) -> Dict:
        """Generate summary statistics for the processed data."""
        if self.encoded_data is None:
            return {}
        
        df = self.encoded_data
        feature_cols = self.get_feature_columns()
        
        stats = {
            'total_samples': len(df),
            'num_species': df['bacterial_species'].nunique() if 'bacterial_species' in df.columns else 0,
            'num_regions': df['administrative_region'].nunique() if 'administrative_region' in df.columns else 0,
            'num_antibiotics': len(feature_cols),
            'mar_index_stats': df['mar_index'].describe().to_dict() if 'mar_index' in df.columns else {},
            'species_distribution': df['bacterial_species'].value_counts().to_dict() if 'bacterial_species' in df.columns else {},
            'high_mar_distribution': df['high_mar'].value_counts().to_dict() if 'high_mar' in df.columns else {},
        }
        
        return stats


def run_phase0(filepath: str, output_dir: str = 'outputs') -> Dict:
    """
    Run complete Phase 0: Data Understanding & Preparation
    
    Args:
        filepath: Path to rawdata.csv
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary containing processed data splits and metadata
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 0: DATA UNDERSTANDING & PREPARATION")
    print("=" * 60)
    
    # Initialize data preparation
    prep = DataPreparation(filepath)
    
    # 0.1 Data Cleaning
    print("\n" + "-" * 40)
    print("0.1 Data Cleaning")
    print("-" * 40)
    prep.load_data()
    prep.clean_data()
    
    # 0.2 Feature Encoding
    print("\n" + "-" * 40)
    print("0.2 Feature Encoding")
    print("-" * 40)
    prep.encode_features()
    
    # 0.3 Create target variables
    print("\n" + "-" * 40)
    print("0.3 Creating Supervised Target Variables")
    print("-" * 40)
    prep.create_target_variables()
    
    # Save cleaned and encoded data
    prep.encoded_data.to_csv(f'{output_dir}/processed_data.csv', index=False)
    print(f"\nSaved processed data to {output_dir}/processed_data.csv")
    
    # 0.4 Data Splitting - for MAR prediction
    print("\n" + "-" * 40)
    print("0.4 Data Splitting - High MAR Prediction")
    print("-" * 40)
    mar_splits = prep.split_data(target_col='high_mar')
    
    # Save MAR splits
    mar_splits['train'].to_csv(f'{output_dir}/mar_train.csv', index=False)
    mar_splits['val'].to_csv(f'{output_dir}/mar_val.csv', index=False)
    mar_splits['test'].to_csv(f'{output_dir}/mar_test.csv', index=False)
    
    # 0.4 Data Splitting - for Species classification
    print("\n" + "-" * 40)
    print("0.4 Data Splitting - Species Classification")
    print("-" * 40)
    species_splits = prep.split_data(target_col='species_target')
    
    # Save species splits
    species_splits['train'].to_csv(f'{output_dir}/species_train.csv', index=False)
    species_splits['val'].to_csv(f'{output_dir}/species_val.csv', index=False)
    species_splits['test'].to_csv(f'{output_dir}/species_test.csv', index=False)
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary Statistics")
    print("-" * 40)
    stats = prep.get_summary_statistics()
    for key, value in stats.items():
        if isinstance(value, dict) and len(value) > 10:
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("PHASE 0 COMPLETE")
    print("=" * 60)
    
    return {
        'prep': prep,
        'mar_splits': mar_splits,
        'species_splits': species_splits,
        'stats': stats
    }


if __name__ == "__main__":
    # Run Phase 0
    result = run_phase0('rawdata.csv', 'outputs')
