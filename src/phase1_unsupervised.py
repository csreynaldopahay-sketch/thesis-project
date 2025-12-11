"""
PHASE 1: UNSUPERVISED PATTERN RECOGNITION

This module handles:
- 1.1 Clustering (K-means, Hierarchical, DBSCAN)
- 1.2 Dimensionality Reduction (PCA, t-SNE, UMAP)
- 1.3 Association Rule Mining (Apriori, FP-Growth)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("mlxtend not available. Install with: pip install mlxtend")


class ClusteringAnalysis:
    """
    PHASE 1.1: Clustering Analysis
    
    Group isolates with similar resistance patterns using:
    - K-means
    - Hierarchical Clustering
    - DBSCAN
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        """
        Initialize clustering analysis.
        
        Args:
            data: DataFrame with encoded antibiotic features
            feature_cols: List of feature column names to use for clustering
        """
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.X = data[feature_cols].values
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.cluster_labels = {}
        self.models = {}
        
    def find_optimal_k(self, k_range: range = range(2, 11)) -> Tuple[int, Dict]:
        """
        Determine optimal number of clusters using Elbow method and Silhouette score.
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Tuple of (optimal_k, results_dict)
        """
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                sil_score = silhouette_score(self.X_scaled, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find optimal k based on silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        results = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
        
        print(f"Optimal k based on silhouette score: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.4f}")
        
        return optimal_k, results
    
    def run_kmeans(self, n_clusters: int = 3) -> np.ndarray:
        """
        Run K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X_scaled)
        
        self.cluster_labels['kmeans'] = labels
        self.models['kmeans'] = kmeans
        
        sil_score = silhouette_score(self.X_scaled, labels)
        print(f"K-means ({n_clusters} clusters) - Silhouette Score: {sil_score:.4f}")
        
        return labels
    
    def run_hierarchical(self, n_clusters: int = 3, linkage_method: str = 'ward') -> np.ndarray:
        """
        Run Hierarchical (Agglomerative) Clustering.
        
        Args:
            n_clusters: Number of clusters
            linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            
        Returns:
            Cluster labels
        """
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = agg.fit_predict(self.X_scaled)
        
        self.cluster_labels['hierarchical'] = labels
        self.models['hierarchical'] = agg
        
        sil_score = silhouette_score(self.X_scaled, labels)
        print(f"Hierarchical ({n_clusters} clusters, {linkage_method}) - Silhouette Score: {sil_score:.4f}")
        
        return labels
    
    def run_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Run DBSCAN clustering (detects outliers/rare resistant isolates).
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Cluster labels (-1 indicates outliers)
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.X_scaled)
        
        self.cluster_labels['dbscan'] = labels
        self.models['dbscan'] = dbscan
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = (labels == -1).sum()
        
        print(f"DBSCAN - Clusters: {n_clusters}, Outliers: {n_outliers}")
        
        if n_clusters > 1:
            # Calculate silhouette excluding outliers
            mask = labels != -1
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                sil_score = silhouette_score(self.X_scaled[mask], labels[mask])
                print(f"DBSCAN Silhouette Score (excluding outliers): {sil_score:.4f}")
        
        return labels
    
    def describe_clusters(self, method: str = 'kmeans') -> pd.DataFrame:
        """
        Describe clusters in terms of species composition, MAR index, and resistance patterns.
        
        Args:
            method: Clustering method to describe
            
        Returns:
            DataFrame with cluster summaries
        """
        if method not in self.cluster_labels:
            raise ValueError(f"Run {method} clustering first")
        
        df = self.data.copy()
        df['cluster'] = self.cluster_labels[method]
        
        summaries = []
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            summary = {
                'cluster': cluster_id,
                'n_samples': len(cluster_data),
                'proportion': len(cluster_data) / len(df)
            }
            
            # MAR index statistics
            if 'mar_index' in cluster_data.columns:
                summary['mar_mean'] = cluster_data['mar_index'].mean()
                summary['mar_std'] = cluster_data['mar_index'].std()
                summary['high_mar_pct'] = (cluster_data['mar_index'] > 0.3).mean() * 100
            
            # Species composition (top 3)
            if 'bacterial_species' in cluster_data.columns:
                species_dist = cluster_data['bacterial_species'].value_counts(normalize=True).head(3)
                summary['top_species'] = ', '.join([f"{s}({v:.1%})" for s, v in species_dist.items()])
            
            # Location distribution
            if 'administrative_region' in cluster_data.columns:
                region_dist = cluster_data['administrative_region'].value_counts(normalize=True).head(3)
                summary['top_regions'] = ', '.join([f"{r}({v:.1%})" for r, v in region_dist.items()])
            
            # Average resistance per antibiotic
            for col in self.feature_cols[:5]:  # Top 5 antibiotics
                antibiotic_name = col.replace('_encoded', '')
                summary[f'avg_{antibiotic_name}'] = cluster_data[col].mean()
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def plot_elbow_silhouette(self, results: Dict, output_path: str = None) -> plt.Figure:
        """Plot Elbow curve and Silhouette scores."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow plot
        axes[0].plot(results['k_values'], results['inertias'], 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].axvline(x=results['optimal_k'], color='r', linestyle='--', label=f'Optimal k={results["optimal_k"]}')
        axes[0].legend()
        
        # Silhouette plot
        axes[1].plot(results['k_values'], results['silhouette_scores'], 'go-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score vs k')
        axes[1].axvline(x=results['optimal_k'], color='r', linestyle='--', label=f'Optimal k={results["optimal_k"]}')
        axes[1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved elbow/silhouette plot to {output_path}")
        
        return fig
    
    def plot_dendrogram(self, output_path: str = None) -> plt.Figure:
        """Plot hierarchical clustering dendrogram."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        Z = linkage(self.X_scaled, method='ward')
        dendrogram(Z, ax=ax, truncate_mode='level', p=5)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        ax.set_title('Hierarchical Clustering Dendrogram')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved dendrogram to {output_path}")
        
        return fig


class DimensionalityReduction:
    """
    PHASE 1.2: Dimensionality Reduction
    
    Visualize high-dimensional resistance data using:
    - PCA
    - t-SNE
    - UMAP
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        """
        Initialize dimensionality reduction.
        
        Args:
            data: DataFrame with encoded antibiotic features
            feature_cols: List of feature column names
        """
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.X = data[feature_cols].values
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.embeddings = {}
        
    def run_pca(self, n_components: int = 2) -> Tuple[np.ndarray, Dict]:
        """
        Run PCA dimensionality reduction.
        
        Args:
            n_components: Number of components to keep
            
        Returns:
            Tuple of (embedding, pca_info)
        """
        # First run full PCA to get variance explained
        pca_full = PCA()
        pca_full.fit(self.X_scaled)
        
        # Then get requested components
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(self.X_scaled)
        
        self.embeddings['pca'] = embedding
        
        pca_info = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'total_variance_explained': sum(pca.explained_variance_ratio_),
            'all_variance_ratios': pca_full.explained_variance_ratio_,
            'components': pca.components_
        }
        
        print(f"PCA: {n_components} components explain {pca_info['total_variance_explained']*100:.2f}% variance")
        
        return embedding, pca_info
    
    def run_tsne(self, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        """
        Run t-SNE dimensionality reduction.
        
        Args:
            n_components: Number of dimensions (usually 2 or 3)
            perplexity: t-SNE perplexity parameter
            
        Returns:
            t-SNE embedding
        """
        # Adjust perplexity if dataset is small
        n_samples = len(self.X_scaled)
        perplexity = min(perplexity, (n_samples - 1) / 3)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, max_iter=1000)
        embedding = tsne.fit_transform(self.X_scaled)
        
        self.embeddings['tsne'] = embedding
        print(f"t-SNE: Reduced to {n_components}D (perplexity={perplexity:.1f})")
        
        return embedding
    
    def run_umap(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """
        Run UMAP dimensionality reduction.
        
        Args:
            n_components: Number of dimensions
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            
        Returns:
            UMAP embedding
        """
        if not UMAP_AVAILABLE:
            print("UMAP not available. Skipping.")
            return None
        
        # Adjust n_neighbors if dataset is small
        n_samples = len(self.X_scaled)
        n_neighbors = min(n_neighbors, n_samples - 1)
        
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(self.X_scaled)
        
        self.embeddings['umap'] = embedding
        print(f"UMAP: Reduced to {n_components}D (n_neighbors={n_neighbors})")
        
        return embedding
    
    def plot_variance_explained(self, pca_info: Dict, output_path: str = None) -> plt.Figure:
        """Plot PCA variance explained."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        n_components = len(pca_info['all_variance_ratios'])
        x = range(1, min(n_components, 20) + 1)
        
        # Individual variance
        axes[0].bar(x, pca_info['all_variance_ratios'][:len(x)])
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance Explained Ratio')
        axes[0].set_title('Individual Variance Explained by PCA')
        
        # Cumulative variance
        cumsum = np.cumsum(pca_info['all_variance_ratios'][:len(x)])
        axes[1].plot(x, cumsum, 'bo-')
        axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title('Cumulative Variance Explained by PCA')
        axes[1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved PCA variance plot to {output_path}")
        
        return fig
    
    def plot_embeddings(self, 
                       color_by: str = 'species_target',
                       cluster_labels: Dict[str, np.ndarray] = None,
                       output_path: str = None) -> plt.Figure:
        """
        Plot 2D embeddings from all methods.
        
        Args:
            color_by: Column name to color points by
            cluster_labels: Dict of clustering labels to add as subplot
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_methods = len(self.embeddings)
        if cluster_labels:
            n_methods += len(cluster_labels)
        
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        idx = 0
        
        # Get color values
        if color_by in self.data.columns:
            colors = self.data[color_by]
            if colors.dtype == 'object':
                color_codes = pd.Categorical(colors).codes
            else:
                color_codes = colors
        else:
            color_codes = np.zeros(len(self.data))
        
        # Plot dimensionality reduction methods
        for method, embedding in self.embeddings.items():
            if idx < len(axes):
                scatter = axes[idx].scatter(embedding[:, 0], embedding[:, 1], 
                                           c=color_codes, cmap='tab10', alpha=0.6, s=30)
                axes[idx].set_xlabel(f'{method.upper()} Dimension 1')
                axes[idx].set_ylabel(f'{method.upper()} Dimension 2')
                axes[idx].set_title(f'{method.upper()} - colored by {color_by}')
                idx += 1
        
        # Plot with cluster labels
        if cluster_labels:
            for method, labels in cluster_labels.items():
                if idx < len(axes) and 'pca' in self.embeddings:
                    embedding = self.embeddings['pca']
                    scatter = axes[idx].scatter(embedding[:, 0], embedding[:, 1],
                                               c=labels, cmap='viridis', alpha=0.6, s=30)
                    axes[idx].set_xlabel('PCA Dimension 1')
                    axes[idx].set_ylabel('PCA Dimension 2')
                    axes[idx].set_title(f'PCA - {method} clusters')
                    idx += 1
        
        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved embeddings plot to {output_path}")
        
        return fig


class AssociationRuleMining:
    """
    PHASE 1.3: Association Rule Mining
    
    Find co-resistance patterns using:
    - Apriori
    - FP-Growth
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        """
        Initialize association rule mining.
        
        Args:
            data: DataFrame with encoded antibiotic features
            feature_cols: List of feature column names (encoded)
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError("mlxtend is required for association rule mining")
        
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.binary_data = None
        self.frequent_itemsets = None
        self.rules = None
        
    def prepare_binary_data(self, threshold: int = 2) -> pd.DataFrame:
        """
        Convert encoded resistance data to binary format for association rules.
        
        Args:
            threshold: Value at or above which to consider resistant (default: 2 = 'r')
            
        Returns:
            Binary DataFrame where True = resistant
        """
        binary_df = pd.DataFrame()
        
        for col in self.feature_cols:
            antibiotic_name = col.replace('_encoded', '_resistant')
            binary_df[antibiotic_name] = self.data[col] >= threshold
        
        self.binary_data = binary_df
        print(f"Created binary resistance data with {len(binary_df.columns)} antibiotics")
        print(f"Resistance rates per antibiotic:")
        for col in binary_df.columns:
            rate = binary_df[col].mean() * 100
            print(f"  {col}: {rate:.1f}%")
        
        return binary_df
    
    def run_apriori(self, min_support: float = 0.02, max_len: int = 5) -> pd.DataFrame:
        """
        Run Apriori algorithm to find frequent itemsets.
        
        Args:
            min_support: Minimum support threshold
            max_len: Maximum itemset length
            
        Returns:
            DataFrame of frequent itemsets
        """
        if self.binary_data is None:
            self.prepare_binary_data()
        
        self.frequent_itemsets = apriori(self.binary_data, min_support=min_support,
                                         use_colnames=True, max_len=max_len)
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets with min_support={min_support}")
        
        return self.frequent_itemsets
    
    def run_fpgrowth(self, min_support: float = 0.02, max_len: int = 5) -> pd.DataFrame:
        """
        Run FP-Growth algorithm to find frequent itemsets.
        
        Args:
            min_support: Minimum support threshold
            max_len: Maximum itemset length
            
        Returns:
            DataFrame of frequent itemsets
        """
        if self.binary_data is None:
            self.prepare_binary_data()
        
        self.frequent_itemsets = fpgrowth(self.binary_data, min_support=min_support,
                                          use_colnames=True, max_len=max_len)
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets with min_support={min_support}")
        
        return self.frequent_itemsets
    
    def generate_rules(self, 
                       min_confidence: float = 0.6,
                       min_lift: float = 1.0) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
            
        Returns:
            DataFrame of association rules
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            print("No frequent itemsets found. Run apriori or fpgrowth first.")
            return pd.DataFrame()
        
        # Filter for itemsets with at least 2 items for rules
        multi_item = self.frequent_itemsets[self.frequent_itemsets['itemsets'].apply(len) >= 2]
        
        if len(multi_item) == 0:
            print("No multi-item frequent itemsets found.")
            return pd.DataFrame()
        
        try:
            # For newer versions of mlxtend, we need to include all itemsets
            self.rules = association_rules(self.frequent_itemsets, metric='confidence', 
                                           min_threshold=min_confidence)
            
            # Filter by lift
            self.rules = self.rules[self.rules['lift'] >= min_lift]
            
            # Sort by lift
            self.rules = self.rules.sort_values('lift', ascending=False)
            
            print(f"Generated {len(self.rules)} rules with confidence>={min_confidence} and lift>={min_lift}")
        except Exception as e:
            print(f"Error generating rules: {e}")
            print("Attempting alternative method...")
            try:
                # Try with support_only first, then calculate metrics
                self.rules = association_rules(self.frequent_itemsets, metric='support', 
                                               min_threshold=0.01)
                # Filter by confidence and lift
                self.rules = self.rules[(self.rules['confidence'] >= min_confidence) & 
                                        (self.rules['lift'] >= min_lift)]
                self.rules = self.rules.sort_values('lift', ascending=False)
                print(f"Generated {len(self.rules)} rules using alternative method")
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                self.rules = pd.DataFrame()
        
        return self.rules
    
    def get_top_rules(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N association rules by lift.
        
        Args:
            n: Number of top rules to return
            
        Returns:
            DataFrame with top rules
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        
        top_rules = self.rules.head(n).copy()
        
        # Format for readability
        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Select key columns
        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        
        return top_rules[display_cols]
    
    def interpret_rules(self) -> List[str]:
        """
        Generate biological interpretations of top rules.
        
        Returns:
            List of interpretation strings
        """
        if self.rules is None or len(self.rules) == 0:
            return ["No rules to interpret"]
        
        interpretations = []
        
        for _, rule in self.rules.head(10).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            
            interpretation = (
                f"If {antecedents}, then {consequents} "
                f"(lift={rule['lift']:.2f}, conf={rule['confidence']:.2f})"
            )
            interpretations.append(interpretation)
        
        return interpretations


def run_phase1(processed_data: pd.DataFrame, 
               feature_cols: List[str],
               output_dir: str = 'outputs') -> Dict:
    """
    Run complete Phase 1: Unsupervised Pattern Recognition
    
    Args:
        processed_data: DataFrame from Phase 0
        feature_cols: List of encoded feature columns
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing all Phase 1 results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 1: UNSUPERVISED PATTERN RECOGNITION")
    print("=" * 60)
    
    results = {}
    
    # 1.1 Clustering
    print("\n" + "-" * 40)
    print("1.1 Clustering Analysis")
    print("-" * 40)
    
    clustering = ClusteringAnalysis(processed_data, feature_cols)
    
    # Find optimal k
    optimal_k, k_results = clustering.find_optimal_k()
    clustering.plot_elbow_silhouette(k_results, f'{output_dir}/elbow_silhouette.png')
    
    # Run clustering algorithms
    clustering.run_kmeans(n_clusters=optimal_k)
    clustering.run_hierarchical(n_clusters=optimal_k)
    clustering.run_dbscan(eps=1.5, min_samples=3)
    
    # Describe clusters
    kmeans_summary = clustering.describe_clusters('kmeans')
    print("\nK-means Cluster Summary:")
    print(kmeans_summary.to_string())
    kmeans_summary.to_csv(f'{output_dir}/kmeans_cluster_summary.csv', index=False)
    
    # Plot dendrogram
    clustering.plot_dendrogram(f'{output_dir}/dendrogram.png')
    
    results['clustering'] = {
        'model': clustering,
        'optimal_k': optimal_k,
        'k_results': k_results,
        'cluster_labels': clustering.cluster_labels,
        'kmeans_summary': kmeans_summary
    }
    
    # 1.2 Dimensionality Reduction
    print("\n" + "-" * 40)
    print("1.2 Dimensionality Reduction")
    print("-" * 40)
    
    dim_red = DimensionalityReduction(processed_data, feature_cols)
    
    # PCA
    pca_embedding, pca_info = dim_red.run_pca(n_components=2)
    dim_red.plot_variance_explained(pca_info, f'{output_dir}/pca_variance.png')
    
    # t-SNE
    tsne_embedding = dim_red.run_tsne(n_components=2)
    
    # UMAP
    umap_embedding = dim_red.run_umap(n_components=2)
    
    # Plot embeddings
    dim_red.plot_embeddings(color_by='species_target', 
                           cluster_labels=clustering.cluster_labels,
                           output_path=f'{output_dir}/embeddings.png')
    
    results['dim_reduction'] = {
        'model': dim_red,
        'pca_info': pca_info,
        'embeddings': dim_red.embeddings
    }
    
    # 1.3 Association Rule Mining
    print("\n" + "-" * 40)
    print("1.3 Association Rule Mining")
    print("-" * 40)
    
    if MLXTEND_AVAILABLE:
        arm = AssociationRuleMining(processed_data, feature_cols)
        arm.prepare_binary_data()
        
        # Try FP-Growth (usually faster)
        arm.run_fpgrowth(min_support=0.02)
        rules = arm.generate_rules(min_confidence=0.5, min_lift=1.0)
        
        if len(rules) > 0:
            top_rules = arm.get_top_rules(20)
            print("\nTop 20 Co-resistance Rules:")
            print(top_rules.to_string())
            top_rules.to_csv(f'{output_dir}/association_rules.csv', index=False)
            
            print("\nRule Interpretations:")
            interpretations = arm.interpret_rules()
            for interp in interpretations:
                print(f"  - {interp}")
        
        results['association_rules'] = {
            'model': arm,
            'rules': rules,
            'interpretations': interpretations if len(rules) > 0 else []
        }
    else:
        print("Skipping association rule mining (mlxtend not available)")
        results['association_rules'] = None
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Example usage
    from phase0_data_preparation import run_phase0
    
    # Run Phase 0 first
    phase0_result = run_phase0('rawdata.csv', 'outputs')
    
    # Get processed data
    processed_data = phase0_result['mar_splits']['full_processed']
    feature_cols = phase0_result['mar_splits']['feature_cols']
    
    # Run Phase 1
    phase1_result = run_phase1(processed_data, feature_cols, 'outputs')
