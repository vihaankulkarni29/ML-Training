"""
Clustering Model Evaluation
Comprehensive metrics for unsupervised clustering quality assessment.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)


def load_results(results_path: str, matrix_path: str):
    """Load clustering results and original data."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix not found: {matrix_path}")
    
    results = pd.read_csv(results_path)
    matrix = pd.read_csv(matrix_path, index_col=0)
    
    # Extract PCA coordinates and labels
    X_pca = results[['PC1', 'PC2', 'PC3']].values
    labels = results['Clan'].values
    
    return X_pca, labels, results, matrix


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute comprehensive clustering quality metrics.
    
    Metrics:
    - Silhouette Score: [-1, 1], higher is better
      Measures how similar objects are to their own cluster vs other clusters
      
    - Davies-Bouldin Index: [0, inf], lower is better
      Average similarity ratio of each cluster with its most similar cluster
      
    - Calinski-Harabasz Score: [0, inf], higher is better
      Ratio of between-cluster dispersion to within-cluster dispersion
      
    - Inertia: Within-cluster sum of squares (lower is better)
    """
    metrics = {}
    
    # Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    metrics['silhouette_score'] = silhouette_avg
    
    # Davies-Bouldin Index (lower is better)
    db_index = davies_bouldin_score(X, labels)
    metrics['davies_bouldin_index'] = db_index
    
    # Calinski-Harabasz Score (higher is better)
    ch_score = calinski_harabasz_score(X, labels)
    metrics['calinski_harabasz_score'] = ch_score
    
    # Compute inertia manually
    inertia = 0.0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - centroid) ** 2)
    metrics['inertia'] = inertia
    
    return metrics


def compute_cluster_statistics(results: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster statistics."""
    stats_list = []
    
    for clan in sorted(results['Clan'].unique()):
        clan_data = results[results['Clan'] == clan]
        clan_samples = clan_data['BioSample'].values
        
        # Get original gene matrix for this clan
        clan_matrix = matrix.loc[matrix.index.isin(clan_samples)]
        
        stats = {
            'Clan': clan,
            'Size': len(clan_data),
            'Percentage': (len(clan_data) / len(results)) * 100,
            'Unique_Locations': clan_data['Location'].nunique(),
            'Top_Location': clan_data['Location'].mode()[0] if len(clan_data) > 0 else 'N/A',
            'Avg_Genes_Per_Sample': clan_matrix.sum(axis=1).mean(),
            'Total_Unique_Genes': (clan_matrix.sum(axis=0) > 0).sum(),
            'PC1_Mean': clan_data['PC1'].mean(),
            'PC2_Mean': clan_data['PC2'].mean(),
            'PC3_Mean': clan_data['PC3'].mean(),
            'PC1_Std': clan_data['PC1'].std(),
            'PC2_Std': clan_data['PC2'].std(),
            'PC3_Std': clan_data['PC3'].std()
        }
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def plot_silhouette_analysis(X: np.ndarray, labels: np.ndarray, output_path: str):
    """Create silhouette plot for each cluster."""
    n_clusters = len(np.unique(labels))
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i, color in zip(range(n_clusters), colors):
        # Get silhouette values for cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        
        # Label the silhouette plots with their cluster numbers
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Clan {i}')
        
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Plot (Avg Score: {silhouette_avg:.3f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)
    
    # Vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, label=f'Average: {silhouette_avg:.3f}')
    ax.legend()
    
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved silhouette plot -> {output_path}")
    plt.close()


def plot_cluster_sizes(cluster_stats: pd.DataFrame, output_path: str):
    """Visualize cluster size distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax1.bar(cluster_stats['Clan'], cluster_stats['Size'], color='steelblue', edgecolor='black')
    ax1.set_xlabel('Clan ID', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    ax2.pie(
        cluster_stats['Size'],
        labels=[f"Clan {c}" for c in cluster_stats['Clan']],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10(np.linspace(0, 1, len(cluster_stats)))
    )
    ax2.set_title('Cluster Percentage Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved cluster sizes plot -> {output_path}")
    plt.close()


def print_evaluation_report(metrics: dict, cluster_stats: pd.DataFrame):
    """Print comprehensive evaluation report."""
    print("\n" + "=" * 70)
    print("CLUSTERING EVALUATION REPORT")
    print("=" * 70)
    
    print("\n1. GLOBAL CLUSTERING QUALITY METRICS:")
    print("-" * 70)
    print(f"  Silhouette Score:          {metrics['silhouette_score']:.4f}")
    print(f"    Range: [-1, 1] | Higher is better | >0.5 = Good")
    print(f"    Interpretation: {get_silhouette_interpretation(metrics['silhouette_score'])}")
    
    print(f"\n  Davies-Bouldin Index:      {metrics['davies_bouldin_index']:.4f}")
    print(f"    Range: [0, ∞] | Lower is better | <1.0 = Good")
    print(f"    Interpretation: {get_db_interpretation(metrics['davies_bouldin_index'])}")
    
    print(f"\n  Calinski-Harabasz Score:   {metrics['calinski_harabasz_score']:.2f}")
    print(f"    Range: [0, ∞] | Higher is better")
    print(f"    Interpretation: Good separation between clusters")
    
    print(f"\n  Inertia (WCSS):            {metrics['inertia']:.2f}")
    print(f"    Within-Cluster Sum of Squares | Lower is better")
    
    print("\n2. PER-CLUSTER STATISTICS:")
    print("-" * 70)
    print(cluster_stats.to_string(index=False))
    
    print("\n3. CLUSTER QUALITY ASSESSMENT:")
    print("-" * 70)
    assess_cluster_quality(metrics, cluster_stats)
    
    print("\n" + "=" * 70)


def get_silhouette_interpretation(score: float) -> str:
    """Interpret silhouette score."""
    if score > 0.7:
        return "Excellent - Strong cluster structure"
    elif score > 0.5:
        return "Good - Reasonable cluster structure"
    elif score > 0.25:
        return "Fair - Weak cluster structure"
    else:
        return "Poor - Overlapping clusters or incorrect K"


def get_db_interpretation(score: float) -> str:
    """Interpret Davies-Bouldin index."""
    if score < 0.5:
        return "Excellent - Well-separated clusters"
    elif score < 1.0:
        return "Good - Decent cluster separation"
    elif score < 1.5:
        return "Fair - Moderate cluster overlap"
    else:
        return "Poor - High cluster overlap"


def assess_cluster_quality(metrics: dict, cluster_stats: pd.DataFrame):
    """Overall assessment of clustering quality."""
    issues = []
    strengths = []
    
    # Check silhouette score
    if metrics['silhouette_score'] > 0.5:
        strengths.append(f"✓ Strong silhouette score ({metrics['silhouette_score']:.3f})")
    elif metrics['silhouette_score'] > 0.25:
        issues.append(f"⚠ Moderate silhouette score ({metrics['silhouette_score']:.3f}) - clusters may overlap")
    else:
        issues.append(f"✗ Low silhouette score ({metrics['silhouette_score']:.3f}) - poor cluster quality")
    
    # Check Davies-Bouldin
    if metrics['davies_bouldin_index'] < 1.0:
        strengths.append(f"✓ Low DB index ({metrics['davies_bouldin_index']:.3f}) - good separation")
    else:
        issues.append(f"⚠ High DB index ({metrics['davies_bouldin_index']:.3f}) - cluster overlap")
    
    # Check cluster size imbalance
    size_std = cluster_stats['Size'].std()
    size_mean = cluster_stats['Size'].mean()
    cv = size_std / size_mean
    if cv > 1.5:
        issues.append(f"⚠ High cluster size imbalance (CV={cv:.2f})")
    else:
        strengths.append(f"✓ Balanced cluster sizes (CV={cv:.2f})")
    
    # Check for very small clusters
    min_size = cluster_stats['Size'].min()
    if min_size < len(cluster_stats) * 50:
        issues.append(f"⚠ Very small cluster detected (min size: {min_size})")
    
    print("\nStrengths:")
    for s in strengths:
        print(f"  {s}")
    
    if issues:
        print("\nPotential Issues:")
        for i in issues:
            print(f"  {i}")
    else:
        print("\n✓ No major issues detected")


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering model quality")
    parser.add_argument("--results", default="results/cluster_results.csv", help="Path to cluster results")
    parser.add_argument("--matrix", default="data/genotype_matrix.csv", help="Path to genotype matrix")
    parser.add_argument("--out-silhouette", default="results/silhouette_plot.png", help="Output silhouette plot")
    parser.add_argument("--out-sizes", default="results/cluster_sizes.png", help="Output cluster sizes plot")
    parser.add_argument("--out-stats", default="results/cluster_statistics.csv", help="Output stats CSV")
    args = parser.parse_args()
    
    print("=" * 70)
    print("LOADING DATA...")
    print("=" * 70)
    
    # Load data
    X_pca, labels, results, matrix = load_results(args.results, args.matrix)
    
    # Compute metrics
    print("\nComputing clustering metrics...")
    metrics = compute_clustering_metrics(X_pca, labels)
    
    # Compute per-cluster stats
    print("Computing per-cluster statistics...")
    cluster_stats = compute_cluster_statistics(results, matrix)
    
    # Save statistics
    os.makedirs(os.path.dirname(args.out_stats), exist_ok=True)
    cluster_stats.to_csv(args.out_stats, index=False)
    print(f"Saved cluster statistics -> {args.out_stats}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_silhouette_analysis(X_pca, labels, args.out_silhouette)
    plot_cluster_sizes(cluster_stats, args.out_sizes)
    
    # Print report
    print_evaluation_report(metrics, cluster_stats)


if __name__ == "__main__":
    main()
