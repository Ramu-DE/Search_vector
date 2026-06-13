"""
Visualization utilities for understanding vector search concepts
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def visualize_vector_basics():
    """Visualize basic vector concepts"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2D vector visualization
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('2D Vector Representation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    # Plot vectors
    vectors = {
        'House 1': ([5, 4], 'blue'),
        'House 2': ([4, 3], 'green'),
        'Query': ([4.5, 3.5], 'red')
    }

    for name, (vec, color) in vectors.items():
        ax1.arrow(0, 0, vec[0], vec[1],
                 head_width=0.3, head_length=0.3,
                 fc=color, ec=color, linewidth=2,
                 label=name)
        ax1.text(vec[0] + 0.3, vec[1] + 0.3, name,
                fontsize=10, fontweight='bold')

    ax1.legend(loc='upper left')

    # Distance metrics visualization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Distance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')

    # Two points
    point_a = np.array([2, 2])
    point_b = np.array([7, 6])

    ax2.scatter(*point_a, s=200, c='blue', marker='o', label='Point A', zorder=3)
    ax2.scatter(*point_b, s=200, c='red', marker='o', label='Point B', zorder=3)

    # Euclidean distance (straight line)
    ax2.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]],
            'g--', linewidth=2, label='Euclidean Distance')

    # Manhattan distance
    ax2.plot([point_a[0], point_b[0]], [point_a[1], point_a[1]],
            'orange', linewidth=2, alpha=0.7)
    ax2.plot([point_b[0], point_b[0]], [point_a[1], point_b[1]],
            'orange', linewidth=2, alpha=0.7, label='Manhattan Distance')

    # Calculate distances
    euclidean = np.linalg.norm(point_b - point_a)
    manhattan = np.sum(np.abs(point_b - point_a))

    ax2.text(4.5, 3, f'Euclidean: {euclidean:.2f}',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax2.text(4.5, 1.5, f'Manhattan: {manhattan:.2f}',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('vector_basics.png', dpi=300, bbox_inches='tight')
    print("Saved: vector_basics.png")
    plt.close()


def visualize_similarity_metrics():
    """Visualize cosine similarity vs euclidean distance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Cosine similarity
    ax1 = axes[0]
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Cosine Similarity (Angle-based)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    # Vectors
    vec1 = np.array([4, 2])
    vec2 = np.array([3, 4])
    vec3 = np.array([5, 2.5])  # Similar direction to vec1

    # Draw vectors
    ax1.arrow(0, 0, vec1[0], vec1[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='Vector 1')
    ax1.arrow(0, 0, vec2[0], vec2[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, label='Vector 2 (Different angle)')
    ax1.arrow(0, 0, vec3[0], vec3[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=2, label='Vector 3 (Similar angle)')

    # Calculate cosine similarities
    cos_sim_12 = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_sim_13 = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))

    # Add angle arcs
    angle1 = np.degrees(np.arccos(cos_sim_12))
    angle2 = np.degrees(np.arccos(cos_sim_13))

    ax1.text(2, 4.5, f'Angle(V1, V2): {angle1:.1f}°\nSimilarity: {cos_sim_12:.3f}',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    ax1.text(2, 1, f'Angle(V1, V3): {angle2:.1f}°\nSimilarity: {cos_sim_13:.3f}',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax1.legend(loc='upper left', fontsize=8)

    # Euclidean distance
    ax2 = axes[1]
    ax2.set_xlim(-1, 6)
    ax2.set_ylim(-1, 6)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Euclidean Distance (Length-based)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')

    # Same vectors
    ax2.arrow(0, 0, vec1[0], vec1[1], head_width=0.2, head_length=0.2,
             fc='blue', ec='blue', linewidth=2, label='Vector 1')
    ax2.arrow(0, 0, vec2[0], vec2[1], head_width=0.2, head_length=0.2,
             fc='red', ec='red', linewidth=2, label='Vector 2')
    ax2.arrow(0, 0, vec3[0], vec3[1], head_width=0.2, head_length=0.2,
             fc='green', ec='green', linewidth=2, label='Vector 3')

    # Draw distance lines
    ax2.plot([vec1[0], vec2[0]], [vec1[1], vec2[1]],
            'r--', linewidth=2, alpha=0.5)
    ax2.plot([vec1[0], vec3[0]], [vec1[1], vec3[1]],
            'g--', linewidth=2, alpha=0.5)

    # Calculate Euclidean distances
    dist_12 = np.linalg.norm(vec1 - vec2)
    dist_13 = np.linalg.norm(vec1 - vec3)

    ax2.text(2, 4.5, f'Distance(V1, V2): {dist_12:.2f}',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    ax2.text(4, 1, f'Distance(V1, V3): {dist_13:.2f}',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('similarity_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: similarity_metrics.png")
    plt.close()


def visualize_knn_process():
    """Visualize k-NN search process"""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Generate random points
    n_points = 50
    points = np.random.rand(n_points, 2) * 10

    # Query point
    query = np.array([5, 5])

    # Calculate distances
    distances = np.linalg.norm(points - query, axis=1)
    k = 5
    nearest_indices = np.argsort(distances)[:k]

    # Exact k-NN
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.set_title('Exact k-NN (k=5)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')

    # Plot all points
    ax1.scatter(points[:, 0], points[:, 1], c='lightgray', s=50, alpha=0.5, label='Other points')

    # Plot nearest neighbors
    ax1.scatter(points[nearest_indices, 0], points[nearest_indices, 1],
               c='green', s=100, alpha=0.7, label=f'Top {k} neighbors')

    # Plot query
    ax1.scatter(query[0], query[1], c='red', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Query', zorder=5)

    # Draw circles to nearest neighbors
    for idx in nearest_indices:
        ax1.plot([query[0], points[idx, 0]], [query[1], points[idx, 1]],
                'g--', alpha=0.3, linewidth=1)

    # Draw search radius
    max_dist = distances[nearest_indices[-1]]
    circle = plt.Circle(query, max_dist, fill=False, edgecolor='red',
                       linestyle='--', linewidth=2, alpha=0.5, label=f'Search radius')
    ax1.add_patch(circle)

    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # HNSW visualization (simplified)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Approximate k-NN (HNSW)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')

    # Create hierarchical structure (simplified)
    # Layer 0: All points
    ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=30, alpha=0.3)

    # Layer 1: Subset of points
    layer1_indices = np.random.choice(n_points, size=15, replace=False)
    ax2.scatter(points[layer1_indices, 0], points[layer1_indices, 1],
               c='lightblue', s=60, alpha=0.5, label='Layer 1')

    # Layer 2: Even smaller subset
    layer2_indices = np.random.choice(layer1_indices, size=5, replace=False)
    ax2.scatter(points[layer2_indices, 0], points[layer2_indices, 1],
               c='blue', s=100, alpha=0.7, label='Layer 2 (Entry)')

    # Query
    ax2.scatter(query[0], query[1], c='red', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Query', zorder=5)

    # Show navigation path (simplified)
    entry_point = points[layer2_indices[0]]
    ax2.annotate('', xy=entry_point, xytext=query,
                arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
    ax2.text(query[0] + 0.3, query[1] + 0.3, '1. Start at entry',
            fontsize=8, color='red')

    # Show greedy navigation
    intermediate = points[layer1_indices[3]]
    ax2.annotate('', xy=intermediate, xytext=entry_point,
                arrowprops=dict(arrowstyle='->', color='orange', lw=2, alpha=0.7))
    ax2.text(entry_point[0] + 0.3, entry_point[1] - 0.5, '2. Greedy descent',
            fontsize=8, color='orange')

    # Final neighbors
    final_neighbors = points[nearest_indices[:3]]
    for neighbor in final_neighbors:
        ax2.scatter(neighbor[0], neighbor[1], c='green', s=100, alpha=0.9, zorder=4)

    ax2.text(1, 9, 'Fast: O(log n)\nApproximate: ~95-99% recall',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('knn_process.png', dpi=300, bbox_inches='tight')
    print("Saved: knn_process.png")
    plt.close()


def visualize_search_evolution():
    """Visualize the evolution of search methods"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    search_methods = [
        {
            'name': 'Keyword Search (BM25)',
            'query': 'apple headphones',
            'results': [
                ('Apple AirPods Pro', 0.85, 'green'),
                ('Apple Store Locations', 0.75, 'orange'),
                ('Red Apple Fruit', 0.45, 'red')
            ]
        },
        {
            'name': 'Semantic Search (k-NN)',
            'query': 'best wireless earbuds',
            'results': [
                ('Sony WH-1000XM4', 0.92, 'green'),
                ('Apple AirPods Pro', 0.88, 'green'),
                ('Bose QuietComfort', 0.85, 'green')
            ]
        },
        {
            'name': 'Hybrid Search',
            'query': 'apple wireless headphones',
            'results': [
                ('Apple AirPods Pro', 0.95, 'darkgreen'),
                ('Sony WH-1000XM4', 0.82, 'green'),
                ('Apple AirPods Max', 0.80, 'green')
            ]
        },
        {
            'name': 'Agentic Search',
            'query': 'find me the best noise cancelling headphones under $300',
            'results': [
                ('Sony WH-1000XM4 ($280)', 0.96, 'darkgreen'),
                ('Bose QuietComfort ($299)', 0.94, 'darkgreen'),
                ('Sennheiser HD 450BT ($199)', 0.90, 'green')
            ]
        }
    ]

    for idx, (ax, method) in enumerate(zip(axes.flat, search_methods)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(method['results']) + 1)
        ax.set_title(method['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Relevance Score')
        ax.set_yticks(range(1, len(method['results']) + 1))
        ax.set_yticklabels([r[0] for r in method['results']])
        ax.grid(True, axis='x', alpha=0.3)

        # Add query text
        ax.text(5, len(method['results']) + 0.5, f"Query: '{method['query']}'",
               ha='center', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Plot bars
        for i, (name, score, color) in enumerate(method['results'], 1):
            ax.barh(i, score * 10, height=0.6, color=color, alpha=0.7)
            ax.text(score * 10 + 0.2, i, f'{score:.2f}',
                   va='center', fontsize=9, fontweight='bold')

        # Add method description
        descriptions = [
            'Exact term matching\nFast but literal',
            'Meaning-based\nUnderstands context',
            'Best of both worlds\nBalanced results',
            'AI agent plans search\nNatural language'
        ]
        ax.text(5, 0.2, descriptions[idx],
               ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('search_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: search_evolution.png")
    plt.close()


def visualize_performance_tradeoffs():
    """Visualize cost-latency-recall tradeoff"""
    fig = plt.figure(figsize=(12, 10))

    # 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')

    # Different configurations
    configs = {
        'Exact k-NN\n(In-memory)': ([100, 95, 100], 'red', 's', 200),
        'HNSW\n(In-memory)': ([40, 20, 98], 'orange', 'o', 150),
        'HNSW + FP16': ([20, 20, 98], 'yellow', 'o', 150),
        'HNSW + Binary': ([5, 22, 96], 'lightgreen', 'o', 150),
        'Disk-based': ([10, 50, 97], 'green', '^', 150),
        'Sparse Encoding': ([8, 10, 94], 'cyan', 'd', 150),
        'S3 Vectors': ([2, 200, 95], 'blue', 'v', 150)
    }

    for name, (coords, color, marker, size) in configs.items():
        cost, latency, recall = coords
        ax.scatter(cost, latency, recall, c=color, marker=marker,
                  s=size, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.text(cost, latency, recall + 1, name, fontsize=8, ha='center')

    ax.set_xlabel('Cost (Relative)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Recall (%)', fontsize=11, fontweight='bold')
    ax.set_title('Performance Tradeoffs: Cost vs Latency vs Recall',
                fontsize=14, fontweight='bold', pad=20)

    # Set limits
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 220)
    ax.set_zlim(90, 102)

    # Add "sweet spot" annotation
    ax.text(20, 20, 90, 'Sweet Spot:\nHNSW + Quantization',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'),
           ha='center')

    plt.tight_layout()
    plt.savefig('performance_tradeoffs.png', dpi=300, bbox_inches='tight')
    print("Saved: performance_tradeoffs.png")
    plt.close()


def visualize_embedding_space():
    """Visualize how embeddings cluster in vector space"""
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate clusters for different topics
    clusters = {
        'Action Movies': (np.random.randn(15, 2) * 0.5 + [2, 8], 'red'),
        'Romantic Movies': (np.random.randn(15, 2) * 0.5 + [8, 8], 'pink'),
        'Sci-Fi Movies': (np.random.randn(15, 2) * 0.5 + [5, 5], 'blue'),
        'Comedy Movies': (np.random.randn(15, 2) * 0.5 + [8, 2], 'orange'),
        'Horror Movies': (np.random.randn(15, 2) * 0.5 + [2, 2], 'purple'),
    }

    # Plot clusters
    for name, (points, color) in clusters.items():
        ax.scatter(points[:, 0], points[:, 1], c=color, s=100,
                  alpha=0.6, edgecolors='black', linewidth=0.5, label=name)

        # Add cluster center label
        center = points.mean(axis=0)
        ax.text(center[0], center[1], name, fontsize=10, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add example queries
    queries = [
        ('action thriller', [2.5, 7.5], 'darkred'),
        ('romantic comedy', [8, 5], 'deeppink'),
        ('space adventure', [5, 6], 'darkblue')
    ]

    for query_text, pos, color in queries:
        ax.scatter(pos[0], pos[1], c=color, s=300, marker='*',
                  edgecolors='black', linewidth=2, zorder=5)
        ax.text(pos[0] + 0.5, pos[1] + 0.5, f'Query:\n"{query_text}"',
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Embedding Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Embedding Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Embedding Space (2D Projection)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add explanation
    ax.text(5, -0.5, 'Similar concepts cluster together in vector space\n'
                     'Queries find nearest neighbors based on semantic similarity',
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('embedding_space.png', dpi=300, bbox_inches='tight')
    print("Saved: embedding_space.png")
    plt.close()


if __name__ == "__main__":
    print("Generating visualizations...")
    print("=" * 60)

    visualize_vector_basics()
    visualize_similarity_metrics()
    visualize_knn_process()
    visualize_search_evolution()
    visualize_performance_tradeoffs()
    visualize_embedding_space()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("Files saved:")
    print("  - vector_basics.png")
    print("  - similarity_metrics.png")
    print("  - knn_process.png")
    print("  - search_evolution.png")
    print("  - performance_tradeoffs.png")
    print("  - embedding_space.png")
