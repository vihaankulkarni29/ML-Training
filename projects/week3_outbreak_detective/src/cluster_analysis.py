"""
Outbreak Detective: Bacterial Clan Clustering
Uses PCA + K-Means to identify hidden genomic clusters in Asian bacterial samples.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(matrix_path: str, metadata_path: str):
    """Load genotype matrix and metadata."""
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix not found: {matrix_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    # Load genotype matrix (BioSample x Genes)
    X = pd.read_csv(matrix_path, index_col=0)
    
    # Load metadata (BioSample -> Location)
    meta = pd.read_csv(metadata_path, index_col=0)
    
    print(f"Loaded matrix: {X.shape[0]} samples x {X.shape[1]} genes")
    print(f"Loaded metadata: {meta.shape[0]} samples")
    
    return X, meta


def apply_pca(X: pd.DataFrame, n_components: int = 3) -> tuple:
    """Reduce dimensionality using PCA."""
    print(f"\nApplying PCA: {X.shape[1]} genes -> {n_components} components")
    
    # Standardize features (mean=0, variance=1) - important for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print variance explained
    variance_explained = pca.explained_variance_ratio_
    print(f"Variance explained by each PC: {variance_explained}")
    print(f"Total variance captured: {variance_explained.sum():.2%}")
    
    return X_pca, pca


def find_optimal_k(X_pca: np.ndarray, k_range: range = range(2, 11)) -> int:
    """Use elbow method to find optimal number of clusters."""
    print("\nFinding optimal K using elbow method...")
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)
    
    # Simple heuristic: find elbow (could be improved with silhouette score)
    # For now, return a reasonable default
    optimal_k = 6  # Good balance for visualization
    print(f"Selected K={optimal_k} clusters")
    
    return optimal_k


def cluster_samples(X_pca: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster samples using K-Means."""
    print(f"\nClustering samples into {n_clusters} clans...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_pca)
    
    unique, counts = np.unique(labels, return_counts=True)
    print("Clan sizes:")
    for clan, count in zip(unique, counts):
        print(f"  Clan {clan}: {count} samples")
    
    return labels


def visualize_3d(X_pca: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, 
                 output_path: str = "results/outbreak_clans_3d.png"):
    """Create 3D scatter plot with Matplotlib."""
    print(f"\nGenerating 3D visualization...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
    
    # Plot each cluster
    for i, color in enumerate(colors):
        mask = labels == i
        ax.scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1], 
            X_pca[mask, 2],
            c=[color],
            label=f'Clan {i} (n={mask.sum()})',
            s=30,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.3
        )
    
    # Labels and title
    ax.set_xlabel('PC1 (Genetic Variance)', fontsize=12, labelpad=10)
    ax.set_ylabel('PC2 (Genetic Variance)', fontsize=12, labelpad=10)
    ax.set_zlabel('PC3 (Genetic Variance)', fontsize=12, labelpad=10)
    ax.set_title('Bacterial Genomic Clans: Asia Outbreak Detective\nPCA + K-Means Clustering', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, frameon=True)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D plot -> {output_path}")
    
    plt.show()


def save_results(X: pd.DataFrame, labels: np.ndarray, X_pca: np.ndarray, 
                 metadata: pd.DataFrame, output_path: str):
    """Save clustering results to CSV."""
    results = pd.DataFrame({
        'BioSample': X.index,
        'Clan': labels,
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2]
    })
    
    # Merge with metadata
    results = results.set_index('BioSample').join(metadata, how='left')
    results = results.reset_index()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved results -> {output_path}")
    
    # Print sample results
    print("\nSample results (first 10 rows):")
    print(results.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Cluster bacterial samples using PCA + K-Means")
    parser.add_argument("--matrix", default="data/genotype_matrix.csv", help="Path to genotype matrix")
    parser.add_argument("--metadata", default="data/metadata.csv", help="Path to metadata")
    parser.add_argument("--n-clusters", type=int, default=6, help="Number of clusters (clans)")
    parser.add_argument("--out-plot", default="results/outbreak_clans_3d.png", help="Output plot path")
    parser.add_argument("--out-results", default="results/cluster_results.csv", help="Output results CSV")
    args = parser.parse_args()
    
    print("=" * 70)
    print("OUTBREAK DETECTIVE: Bacterial Clan Clustering")
    print("=" * 70)
    
    # Load data
    X, meta = load_data(args.matrix, args.metadata)
    
    # Apply PCA
    X_pca, pca = apply_pca(X, n_components=3)
    
    # Cluster
    labels = cluster_samples(X_pca, n_clusters=args.n_clusters)
    
    # Visualize
    visualize_3d(X_pca, labels, meta, output_path=args.out_plot)
    
    # Save results
    save_results(X, labels, X_pca, meta, output_path=args.out_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
