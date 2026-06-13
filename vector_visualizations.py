#!/usr/bin/env python3
"""
Vector Concepts Visualization
Educational visualizations for vector search concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Tuple
import json
import boto3
from config import Config
from qdrant_client import QdrantClient
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class VectorVisualizer:
    """Visualize vector search concepts"""

    def __init__(self):
        """Initialize visualizer"""
        self.fig_size = (12, 8)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.fig_size

    def plot_basic_vectors(self):
        """Visualize basic vector concepts: magnitude, direction"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Basic Vector Concepts', fontsize=16, fontweight='bold')

        # 1. Vector Magnitude
        ax = axes[0, 0]
        vectors = np.array([
            [3, 4],
            [1, 1],
            [4, 2],
            [2, 5]
        ])

        for vec in vectors:
            ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.2,
                    fc='blue', ec='blue', alpha=0.6, width=0.05)
            magnitude = np.linalg.norm(vec)
            ax.text(vec[0]*0.5, vec[1]*0.5, f'|v|={magnitude:.2f}',
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat'))

        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vector Magnitude')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # 2. Vector Direction (Unit Vectors)
        ax = axes[0, 1]
        original = np.array([3, 4])
        unit_vec = original / np.linalg.norm(original)

        ax.arrow(0, 0, original[0], original[1], head_width=0.2, head_length=0.2,
                fc='red', ec='red', alpha=0.6, width=0.05, label='Original')
        ax.arrow(0, 0, unit_vec[0], unit_vec[1], head_width=0.1, head_length=0.1,
                fc='green', ec='green', alpha=0.8, width=0.03, label='Unit Vector')

        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vector Direction (Normalization)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # 3. Dot Product
        ax = axes[1, 0]
        v1 = np.array([3, 2])
        v2 = np.array([1, 4])

        ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
                fc='blue', ec='blue', alpha=0.6, width=0.05, label='Vector A')
        ax.arrow(0, 0, v2[0], v2[1], head_width=0.2, head_length=0.2,
                fc='red', ec='red', alpha=0.6, width=0.05, label='Vector B')

        dot_product = np.dot(v1, v2)
        ax.text(2, 4, f'Dot Product = {dot_product}', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightblue'))

        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Dot Product')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # 4. Vector Addition
        ax = axes[1, 1]
        v1 = np.array([2, 1])
        v2 = np.array([1, 3])
        v_sum = v1 + v2

        ax.arrow(0, 0, v1[0], v1[1], head_width=0.2, head_length=0.2,
                fc='blue', ec='blue', alpha=0.6, width=0.05, label='A')
        ax.arrow(v1[0], v1[1], v2[0], v2[1], head_width=0.2, head_length=0.2,
                fc='red', ec='red', alpha=0.6, width=0.05, label='B')
        ax.arrow(0, 0, v_sum[0], v_sum[1], head_width=0.2, head_length=0.2,
                fc='green', ec='green', alpha=0.8, width=0.08, label='A + B')

        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vector Addition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        plt.tight_layout()
        plt.savefig('visualizations/01_basic_vectors.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/01_basic_vectors.png")
        plt.close()

    def plot_similarity_metrics(self):
        """Visualize different similarity metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Vector Similarity Metrics', fontsize=16, fontweight='bold')

        # Generate sample vectors
        query = np.array([1, 1])
        vectors = np.array([
            [0.9, 0.9],   # Very similar
            [0.5, 1.5],   # Moderate
            [-1, 1],      # Different direction
            [2, 0.2]      # Different direction
        ])

        # 1. Cosine Similarity
        ax = axes[0, 0]
        ax.arrow(0, 0, query[0], query[1], head_width=0.1, head_length=0.1,
                fc='red', ec='red', width=0.05, label='Query', linewidth=2)

        for i, vec in enumerate(vectors):
            # Calculate cosine similarity
            cos_sim = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))

            # Color based on similarity
            color = plt.cm.RdYlGn(cos_sim)

            ax.arrow(0, 0, vec[0], vec[1], head_width=0.08, head_length=0.08,
                    fc=color, ec=color, alpha=0.7, width=0.03)
            ax.text(vec[0]*1.2, vec[1]*1.2, f'{cos_sim:.2f}',
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-0.5, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Cosine Similarity\n(measures angle between vectors)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # 2. Euclidean Distance
        ax = axes[0, 1]
        ax.arrow(0, 0, query[0], query[1], head_width=0.1, head_length=0.1,
                fc='red', ec='red', width=0.05, label='Query', linewidth=2)

        for i, vec in enumerate(vectors):
            # Calculate Euclidean distance
            distance = np.linalg.norm(query - vec)

            # Color based on distance (inverted - closer is greener)
            color = plt.cm.RdYlGn(1 - min(distance / 3, 1))

            ax.arrow(0, 0, vec[0], vec[1], head_width=0.08, head_length=0.08,
                    fc=color, ec=color, alpha=0.7, width=0.03)

            # Draw distance line
            ax.plot([query[0], vec[0]], [query[1], vec[1]],
                   'k--', alpha=0.3, linewidth=1)

            ax.text(vec[0]*1.2, vec[1]*1.2, f'd={distance:.2f}',
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-0.5, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Euclidean Distance (L2)\n(measures straight-line distance)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # 3. Manhattan Distance
        ax = axes[1, 0]
        ax.arrow(0, 0, query[0], query[1], head_width=0.1, head_length=0.1,
                fc='red', ec='red', width=0.05, label='Query', linewidth=2)

        for i, vec in enumerate(vectors):
            # Calculate Manhattan distance
            distance = np.sum(np.abs(query - vec))

            color = plt.cm.RdYlGn(1 - min(distance / 4, 1))

            ax.arrow(0, 0, vec[0], vec[1], head_width=0.08, head_length=0.08,
                    fc=color, ec=color, alpha=0.7, width=0.03)

            # Draw Manhattan path
            ax.plot([query[0], vec[0], vec[0]], [query[1], query[1], vec[1]],
                   'b--', alpha=0.3, linewidth=1)

            ax.text(vec[0]*1.2, vec[1]*1.2, f'd={distance:.2f}',
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-0.5, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Manhattan Distance (L1)\n(measures grid-based distance)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # 4. Comparison table
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate all metrics
        metrics_data = []
        for i, vec in enumerate(vectors):
            cos_sim = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            l2_dist = np.linalg.norm(query - vec)
            l1_dist = np.sum(np.abs(query - vec))

            metrics_data.append([
                f"Vec {i+1}",
                f"{cos_sim:.3f}",
                f"{l2_dist:.3f}",
                f"{l1_dist:.3f}"
            ])

        # Create table
        table = ax.table(cellText=metrics_data,
                        colLabels=['Vector', 'Cosine', 'L2 Dist', 'L1 Dist'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.3, 1, 0.6])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Similarity Metrics Comparison', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('visualizations/02_similarity_metrics.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/02_similarity_metrics.png")
        plt.close()

    def plot_knn_search(self):
        """Visualize k-NN search"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('k-Nearest Neighbors (k-NN) Search', fontsize=16, fontweight='bold')

        # Generate dataset
        np.random.seed(42)
        n_points = 100
        data_points = np.random.randn(n_points, 2) * 2

        # Query point
        query = np.array([1.5, 1.5])

        # Calculate distances
        distances = np.linalg.norm(data_points - query, axis=1)

        # 1. Exact k-NN
        ax = axes[0]
        k = 5

        # Find k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]

        # Plot all points
        ax.scatter(data_points[:, 0], data_points[:, 1],
                  c='lightblue', s=50, alpha=0.6, label='Data Points')

        # Plot k-nearest neighbors
        ax.scatter(data_points[nearest_indices, 0],
                  data_points[nearest_indices, 1],
                  c='green', s=100, marker='^', label=f'{k}-Nearest', zorder=5)

        # Plot query
        ax.scatter(query[0], query[1], c='red', s=200, marker='*',
                  label='Query', edgecolors='black', linewidth=2, zorder=10)

        # Draw circles to k-nearest neighbors
        for idx in nearest_indices:
            ax.plot([query[0], data_points[idx, 0]],
                   [query[1], data_points[idx, 1]],
                   'g--', alpha=0.5, linewidth=1)

        # Draw search radius
        max_dist = distances[nearest_indices[-1]]
        circle = plt.Circle(query, max_dist, fill=False, edgecolor='orange',
                           linestyle='--', linewidth=2, label=f'Search Radius')
        ax.add_patch(circle)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Exact k-NN Search (k={k})\nSearches all {n_points} points')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

        # 2. Approximate k-NN (HNSW concept)
        ax = axes[1]

        # Simulate HNSW layers
        layer_colors = ['lightblue', 'lightgreen', 'lightyellow']
        layer_sizes = [20, 50, 100]

        # Plot layers
        for i, (size, color) in enumerate(zip(layer_sizes, layer_colors)):
            indices = np.random.choice(n_points, size, replace=False)
            ax.scatter(data_points[indices, 0], data_points[indices, 1],
                      c=color, s=50, alpha=0.5, label=f'Layer {3-i}')

        # Plot approximate k-nearest (may not be exact)
        approx_search_points = 30  # Only search subset
        search_indices = np.argsort(distances)[:approx_search_points]
        approx_nearest = search_indices[:k]

        ax.scatter(data_points[approx_nearest, 0],
                  data_points[approx_nearest, 1],
                  c='green', s=100, marker='^', label=f'Approx {k}-Nearest', zorder=5)

        # Plot query
        ax.scatter(query[0], query[1], c='red', s=200, marker='*',
                  label='Query', edgecolors='black', linewidth=2, zorder=10)

        # Draw connections
        for idx in approx_nearest:
            ax.plot([query[0], data_points[idx, 0]],
                   [query[1], data_points[idx, 1]],
                   'g--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Approximate k-NN (HNSW-style)\nSearches ~{approx_search_points} points (~{approx_search_points/n_points*100:.0f}%)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)

        plt.tight_layout()
        plt.savefig('visualizations/03_knn_search.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/03_knn_search.png")
        plt.close()

    def plot_hnsw_structure(self):
        """Visualize HNSW (Hierarchical Navigable Small World) structure"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('HNSW Index Structure', fontsize=16, fontweight='bold')

        # Create 3 subplots for different layers
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Generate points
        np.random.seed(42)
        n_points = 50
        points = np.random.randn(n_points, 2) * 3

        # Layer 0 (bottom - all points)
        ax0 = fig.add_subplot(gs[0, :])
        ax0.scatter(points[:, 0], points[:, 1], c='lightblue', s=100, alpha=0.8)

        # Draw edges (dense connections)
        for i in range(n_points):
            neighbors = np.argsort(np.linalg.norm(points - points[i], axis=1))[1:7]
            for j in neighbors:
                ax0.plot([points[i, 0], points[j, 0]],
                        [points[i, 1], points[j, 1]],
                        'gray', alpha=0.2, linewidth=0.5)

        ax0.set_title('Layer 0 (Base Layer): All Points (50), Dense Connections', fontweight='bold')
        ax0.set_xlabel('Dimension 1')
        ax0.set_ylabel('Dimension 2')
        ax0.grid(True, alpha=0.3)

        # Layer 1 (middle - subset of points)
        ax1 = fig.add_subplot(gs[1, :])
        layer1_indices = np.random.choice(n_points, 20, replace=False)
        layer1_points = points[layer1_indices]

        ax1.scatter(points[:, 0], points[:, 1], c='lightgray', s=50, alpha=0.3)
        ax1.scatter(layer1_points[:, 0], layer1_points[:, 1],
                   c='lightgreen', s=150, alpha=0.8, edgecolors='green', linewidth=2)

        # Draw edges (medium connections)
        for i in range(len(layer1_points)):
            neighbors = np.argsort(np.linalg.norm(layer1_points - layer1_points[i], axis=1))[1:5]
            for j in neighbors:
                ax1.plot([layer1_points[i, 0], layer1_points[j, 0]],
                        [layer1_points[i, 1], layer1_points[j, 1]],
                        'green', alpha=0.4, linewidth=1.5)

        ax1.set_title('Layer 1 (Middle Layer): Subset (20 points), Medium Connections', fontweight='bold')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.grid(True, alpha=0.3)

        # Layer 2 (top - small subset)
        ax2 = fig.add_subplot(gs[2, :])
        layer2_indices = np.random.choice(layer1_indices, 5, replace=False)
        layer2_points = points[layer2_indices]

        ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=50, alpha=0.3)
        ax2.scatter(layer2_points[:, 0], layer2_points[:, 1],
                   c='orange', s=200, alpha=0.8, edgecolors='darkorange', linewidth=2)

        # Draw edges (sparse, long-range connections)
        for i in range(len(layer2_points)):
            for j in range(i+1, len(layer2_points)):
                ax2.plot([layer2_points[i, 0], layer2_points[j, 0]],
                        [layer2_points[i, 1], layer2_points[j, 1]],
                        'orange', alpha=0.6, linewidth=2)

        ax2.set_title('Layer 2 (Top Layer): Entry Points (5), Long-Range Connections', fontweight='bold')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.grid(True, alpha=0.3)

        plt.savefig('visualizations/04_hnsw_structure.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/04_hnsw_structure.png")
        plt.close()

    def plot_dimensionality_reduction(self):
        """Visualize high-dimensional vectors in 2D/3D"""
        print("\n🎨 Generating dimensionality reduction visualization...")

        # Generate high-dimensional data (simulate embeddings)
        np.random.seed(42)
        n_samples = 100
        n_dims = 128  # Simulating embedding dimensions

        # Create clusters
        centers = [
            np.random.randn(n_dims) * 5,
            np.random.randn(n_dims) * 5 + 10,
            np.random.randn(n_dims) * 5 - 10
        ]

        data = []
        labels = []
        for i, center in enumerate(centers):
            cluster_data = center + np.random.randn(n_samples // 3, n_dims)
            data.append(cluster_data)
            labels.extend([f'Cluster {i+1}'] * (n_samples // 3))

        data = np.vstack(data)

        # PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        data_tsne = tsne.fit_transform(data)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Dimensionality Reduction: {n_dims}D → 2D', fontsize=16, fontweight='bold')

        # PCA plot
        scatter = axes[0].scatter(data_pca[:, 0], data_pca[:, 1],
                                 c=[0, 1, 2] * (n_samples // 3),
                                 cmap='viridis', s=50, alpha=0.7)
        axes[0].set_title(f'PCA (Principal Component Analysis)\nVariance Explained: {pca.explained_variance_ratio_.sum():.2%}')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].grid(True, alpha=0.3)

        # t-SNE plot
        scatter = axes[1].scatter(data_tsne[:, 0], data_tsne[:, 1],
                                 c=[0, 1, 2] * (n_samples // 3),
                                 cmap='viridis', s=50, alpha=0.7)
        axes[1].set_title('t-SNE (t-Distributed Stochastic Neighbor Embedding)\nPreserves Local Structure')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/05_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/05_dimensionality_reduction.png")
        plt.close()

    def plot_search_performance(self):
        """Visualize search performance: exact vs approximate"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Search Performance: Exact vs Approximate', fontsize=16, fontweight='bold')

        # 1. Accuracy vs Speed trade-off
        ax = axes[0, 0]
        ef_values = np.array([10, 20, 50, 100, 200, 500])
        accuracy = np.array([0.70, 0.85, 0.92, 0.96, 0.98, 0.99])
        speed = np.array([0.5, 1, 2.5, 5, 10, 25])  # ms

        ax.plot(ef_values, accuracy * 100, 'o-', linewidth=2, markersize=8,
               color='green', label='Accuracy')
        ax2 = ax.twinx()
        ax2.plot(ef_values, speed, 's-', linewidth=2, markersize=8,
                color='red', label='Latency')

        ax.set_xlabel('ef_search Parameter')
        ax.set_ylabel('Accuracy (%)', color='green')
        ax2.set_ylabel('Latency (ms)', color='red')
        ax.set_title('Accuracy vs Speed Trade-off (HNSW)')
        ax.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # 2. Dataset size vs search time
        ax = axes[0, 1]
        sizes = np.array([1000, 10000, 100000, 1000000, 10000000])
        exact_time = sizes * 0.001  # Linear with dataset size
        approx_time = np.log(sizes) * 2  # Logarithmic

        ax.loglog(sizes, exact_time, 'o-', linewidth=2, label='Exact k-NN', color='red')
        ax.loglog(sizes, approx_time, 's-', linewidth=2, label='Approximate k-NN (HNSW)', color='green')

        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Search Time (ms)')
        ax.set_title('Scalability: Dataset Size vs Search Time')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # 3. Recall vs Database Size
        ax = axes[1, 0]
        db_sizes = [10000, 100000, 1000000, 10000000]
        recall_hnsw = [0.98, 0.96, 0.94, 0.92]
        recall_ivf = [0.95, 0.92, 0.88, 0.85]
        recall_exact = [1.0, 1.0, 1.0, 1.0]

        x = np.arange(len(db_sizes))
        width = 0.25

        ax.bar(x - width, recall_exact, width, label='Exact', color='gold')
        ax.bar(x, recall_hnsw, width, label='HNSW', color='green')
        ax.bar(x + width, recall_ivf, width, label='IVF', color='blue')

        ax.set_xlabel('Database Size')
        ax.set_ylabel('Recall@10')
        ax.set_title('Recall vs Database Size')
        ax.set_xticks(x)
        ax.set_xticklabels(['10K', '100K', '1M', '10M'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.8, 1.05])

        # 4. Memory vs Accuracy
        ax = axes[1, 1]
        algorithms = ['Exact\nk-NN', 'HNSW\n(M=16)', 'HNSW\n(M=32)', 'IVF', 'Product\nQuantization']
        memory = [100, 120, 150, 80, 30]  # Relative memory usage
        accuracy = [100, 96, 98, 92, 85]  # Accuracy %

        colors = ['gold', 'green', 'darkgreen', 'blue', 'purple']
        bars = ax.bar(algorithms, memory, color=colors, alpha=0.7)

        # Add accuracy as text on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracy)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc}%',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Relative Memory Usage')
        ax.set_title('Memory Usage vs Accuracy\n(accuracy shown on bars)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('visualizations/06_search_performance.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/06_search_performance.png")
        plt.close()

    def create_all_visualizations(self):
        """Generate all visualizations"""
        import os
        os.makedirs('visualizations', exist_ok=True)

        print("=" * 70)
        print("🎨 Generating Vector Visualizations")
        print("=" * 70)

        print("\n1. Basic Vector Concepts...")
        self.plot_basic_vectors()

        print("\n2. Similarity Metrics...")
        self.plot_similarity_metrics()

        print("\n3. k-NN Search...")
        self.plot_knn_search()

        print("\n4. HNSW Structure...")
        self.plot_hnsw_structure()

        print("\n5. Dimensionality Reduction...")
        self.plot_dimensionality_reduction()

        print("\n6. Search Performance...")
        self.plot_search_performance()

        print("\n" + "=" * 70)
        print("✅ All Visualizations Created!")
        print("=" * 70)
        print("\nSaved in: visualizations/")
        print("  01_basic_vectors.png")
        print("  02_similarity_metrics.png")
        print("  03_knn_search.png")
        print("  04_hnsw_structure.png")
        print("  05_dimensionality_reduction.png")
        print("  06_search_performance.png")
        print("\nOpen these files to see the visualizations!")


def main():
    """Main visualization workflow"""
    visualizer = VectorVisualizer()
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()
