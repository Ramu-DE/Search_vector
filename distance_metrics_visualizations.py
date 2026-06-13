#!/usr/bin/env python3
"""
Distance Metrics Visualizations
Visual explanations of Euclidean, Cosine Similarity, and Dot Product
Based on AWS slide page 37
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Arc
from mpl_toolkits.mplot3d import Axes3D
import os


class DistanceMetricsVisualizer:
    """Visualize KNN distance metrics"""

    def __init__(self):
        """Initialize visualizer"""
        self.fig_size = (14, 10)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.fig_size
        os.makedirs('visualizations', exist_ok=True)

    def plot_three_metrics_comparison(self):
        """Visualize all three metrics from AWS slide"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KNN Distance Metrics - AWS Slide Page 37',
                     fontsize=16, fontweight='bold')

        # Define two vectors for demonstration
        vec_a = np.array([2, 5])
        vec_b = np.array([6, 3])

        # 1. Euclidean Distance - Visualization
        ax = axes[0, 0]
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw vectors
        ax.arrow(0, 0, vec_a[0], vec_a[1], head_width=0.3, head_length=0.3,
                fc='blue', ec='blue', linewidth=2, label='Vector a')
        ax.arrow(0, 0, vec_b[0], vec_b[1], head_width=0.3, head_length=0.3,
                fc='red', ec='red', linewidth=2, label='Vector b')

        # Draw distance line
        ax.plot([vec_a[0], vec_b[0]], [vec_a[1], vec_b[1]],
               'g--', linewidth=3, label='Distance')

        # Calculate distance
        dist = np.sqrt((vec_b[0] - vec_a[0])**2 + (vec_b[1] - vec_a[1])**2)

        ax.text(vec_a[0] - 0.5, vec_a[1] + 0.5, 'a', fontsize=14, fontweight='bold')
        ax.text(vec_b[0] + 0.3, vec_b[1] + 0.3, 'b', fontsize=14, fontweight='bold')
        ax.text(4, 4.5, f'd = {dist:.2f}', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.set_title('Euclidean Distance (L2)', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left')

        # 2. Euclidean Distance - Formula
        ax = axes[0, 1]
        ax.axis('off')

        formula_text = r'''
Euclidean Distance (L2)

Formula:
$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n}(q_i - p_i)^2}$

Properties:
  • Straight-line distance
  • Sensitive to magnitude
  • Range: [0, ∞)
  • Lower is more similar

Use Cases:
  ✓ Counts / Measurements
  ✓ Recommendation Systems
  ✓ Physical distances
  ✓ When scale matters

Example:
  a = [2, 5]
  b = [6, 3]
  d = √[(6-2)² + (3-5)²]
    = √[16 + 4]
    = √20 = 4.47
'''

        ax.text(0.1, 0.5, formula_text, fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 3. Cosine Similarity - Visualization
        ax = axes[0, 2]
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw vectors
        ax.arrow(0, 0, vec_a[0], vec_a[1], head_width=0.3, head_length=0.3,
                fc='blue', ec='blue', linewidth=2, label='Vector a')
        ax.arrow(0, 0, vec_b[0], vec_b[1], head_width=0.3, head_length=0.3,
                fc='red', ec='red', linewidth=2, label='Vector b')

        # Draw angle arc
        angle_a = np.arctan2(vec_a[1], vec_a[0])
        angle_b = np.arctan2(vec_b[1], vec_b[0])

        arc = Arc((0, 0), 2, 2, angle=0,
                 theta1=np.degrees(angle_b), theta2=np.degrees(angle_a),
                 color='green', linewidth=2)
        ax.add_patch(arc)

        # Calculate cosine similarity
        cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        angle_deg = np.degrees(np.arccos(cos_sim))

        ax.text(vec_a[0] - 0.5, vec_a[1] + 0.5, 'a', fontsize=14, fontweight='bold')
        ax.text(vec_b[0] + 0.3, vec_b[1] + 0.3, 'b', fontsize=14, fontweight='bold')
        ax.text(1.5, 2, f'θ = {angle_deg:.1f}°', fontsize=11, color='green', fontweight='bold')
        ax.text(3, 6, f'cos(θ) = {cos_sim:.3f}', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.set_title('Cosine Similarity', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left')

        # 4. Cosine Similarity - Formula
        ax = axes[1, 0]
        ax.axis('off')

        formula_text = r'''
Cosine Similarity

Formula:
$sim(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$

Properties:
  • Measures angle, not distance
  • Invariant to magnitude
  • Range: [-1, 1]
  • 1 = same direction

Use Cases:
  ✓ Semantic search
  ✓ Document classification
  ✓ Text similarity
  ✓ When scale irrelevant

Example:
  a = [2, 5]
  b = [6, 3]
  sim = (2×6 + 5×3) / (√29 × √45)
      = 27 / 36.12 = 0.747
'''

        ax.text(0.1, 0.5, formula_text, fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        # 5. Dot Product - Visualization
        ax = axes[1, 1]
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw vectors
        ax.arrow(0, 0, vec_a[0], vec_a[1], head_width=0.3, head_length=0.3,
                fc='blue', ec='blue', linewidth=2, label='Vector a')
        ax.arrow(0, 0, vec_b[0], vec_b[1], head_width=0.3, head_length=0.3,
                fc='red', ec='red', linewidth=2, label='Vector b')

        # Project b onto a
        a_unit = vec_a / np.linalg.norm(vec_a)
        projection_length = np.dot(vec_b, a_unit)
        projection = projection_length * a_unit

        ax.plot([vec_b[0], projection[0]], [vec_b[1], projection[1]],
               'g--', linewidth=2, alpha=0.7, label='Projection')
        ax.plot([0, projection[0]], [0, projection[1]],
               'g-', linewidth=3, alpha=0.7)

        # Calculate dot product
        dot_prod = np.dot(vec_a, vec_b)

        ax.text(vec_a[0] - 0.5, vec_a[1] + 0.5, 'a', fontsize=14, fontweight='bold')
        ax.text(vec_b[0] + 0.3, vec_b[1] + 0.3, 'b', fontsize=14, fontweight='bold')
        ax.text(3, 6, f'a·b = {dot_prod:.1f}', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.set_title('Dot Product', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper left')

        # 6. Dot Product - Formula
        ax = axes[1, 2]
        ax.axis('off')

        formula_text = r'''
Dot Product

Formula:
$\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}|| ||\mathbf{b}|| \cos\alpha$

Alternative:
$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \times b_i$

Properties:
  • Combines angle & magnitude
  • Sensitive to scale
  • Range: (-∞, ∞)
  • Higher = more similar

Use Cases:
  ✓ Collaborative filtering
  ✓ Recommendations
  ✓ When both matter

Example:
  a = [2, 5]
  b = [6, 3]
  dot = 2×6 + 5×3
      = 12 + 15 = 27
'''

        ax.text(0.1, 0.5, formula_text, fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        plt.tight_layout()
        plt.savefig('visualizations/17_distance_metrics_overview.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/17_distance_metrics_overview.png")
        plt.close()

    def plot_metric_comparisons(self):
        """Compare behavior of different metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Distance Metrics Behavior Comparison',
                     fontsize=16, fontweight='bold')

        # Generate test vectors
        np.random.seed(42)
        n_points = 50
        query = np.array([1, 1])

        # 1. Distance vs angle visualization
        ax = axes[0, 0]

        # Generate points at different angles and distances
        angles = np.linspace(0, 2*np.pi, n_points)
        points = np.array([[np.cos(a), np.sin(a)] for a in angles]) * np.random.uniform(0.5, 3, (n_points, 1))

        # Calculate metrics
        euclidean_dists = np.array([np.linalg.norm(query - p) for p in points])
        cosine_sims = np.array([np.dot(query, p) / (np.linalg.norm(query) * np.linalg.norm(p)) for p in points])

        # Plot
        scatter = ax.scatter(points[:, 0], points[:, 1],
                           c=euclidean_dists, cmap='RdYlGn_r', s=100, alpha=0.7,
                           edgecolors='black', linewidth=1)
        ax.arrow(0, 0, query[0], query[1], head_width=0.2, head_length=0.2,
                fc='red', ec='red', linewidth=3, label='Query')

        plt.colorbar(scatter, ax=ax, label='Euclidean Distance')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title('Euclidean: Circular Distance Contours')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Cosine similarity visualization
        ax = axes[0, 1]

        scatter = ax.scatter(points[:, 0], points[:, 1],
                           c=cosine_sims, cmap='RdYlGn', s=100, alpha=0.7,
                           edgecolors='black', linewidth=1, vmin=-1, vmax=1)
        ax.arrow(0, 0, query[0], query[1], head_width=0.2, head_length=0.2,
                fc='red', ec='red', linewidth=3, label='Query')

        # Draw angle lines
        for angle in [0, 45, 90]:
            rad = np.radians(angle)
            ax.plot([0, 3*np.cos(rad)], [0, 3*np.sin(rad)],
                   'k--', alpha=0.3, linewidth=1)
            ax.text(3.3*np.cos(rad), 3.3*np.sin(rad), f'{angle}°',
                   fontsize=9, ha='center')

        plt.colorbar(scatter, ax=ax, label='Cosine Similarity')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title('Cosine: Angle-Based (Radial Contours)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3. Scale sensitivity comparison
        ax = axes[1, 0]

        # Original vector and scaled versions
        base = np.array([1, 1])
        scales = [0.5, 1, 2, 3, 4]
        colors = ['red', 'orange', 'green', 'blue', 'purple']

        euclidean_vals = []
        cosine_vals = []
        dot_vals = []

        for scale, color in zip(scales, colors):
            vec = base * scale
            ax.arrow(0, 0, vec[0], vec[1], head_width=0.15, head_length=0.15,
                    fc=color, ec=color, linewidth=2, alpha=0.7)
            ax.text(vec[0] + 0.2, vec[1] + 0.2, f'{scale}x', fontsize=10,
                   color=color, fontweight='bold')

            # Calculate metrics
            euclidean_vals.append(np.linalg.norm(query - vec))
            cosine_vals.append(np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec)))
            dot_vals.append(np.dot(query, vec))

        ax.arrow(0, 0, query[0], query[1], head_width=0.15, head_length=0.15,
                fc='black', ec='black', linewidth=3, label='Query')

        ax.set_xlim(-0.5, 5)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect('equal')
        ax.set_title('Scale Sensitivity Test\n(All vectors same direction)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 4. Metric values vs scale
        ax = axes[1, 1]

        ax.plot(scales, euclidean_vals, 'o-', linewidth=2, markersize=8,
               label='Euclidean', color='red')
        ax2 = ax.twinx()
        ax2.plot(scales, cosine_vals, 's-', linewidth=2, markersize=8,
                label='Cosine', color='green')
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(scales, dot_vals, '^-', linewidth=2, markersize=8,
                label='Dot Product', color='blue')

        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Euclidean Distance', color='red')
        ax2.set_ylabel('Cosine Similarity', color='green')
        ax3.set_ylabel('Dot Product', color='blue')

        ax.tick_params(axis='y', labelcolor='red')
        ax2.tick_params(axis='y', labelcolor='green')
        ax3.tick_params(axis='y', labelcolor='blue')

        ax.set_title('Metric Response to Scaling')
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.text(3, 2.5, '← Euclidean increases\n(farther from query)',
               fontsize=9, color='red')
        ax2.text(3, 1.0, '← Cosine constant\n(same direction)',
                fontsize=9, color='green')
        ax3.text(3, 14, '← Dot Product increases\n(magnitude matters)',
                fontsize=9, color='blue')

        plt.tight_layout()
        plt.savefig('visualizations/18_metric_comparisons.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/18_metric_comparisons.png")
        plt.close()

    def plot_use_case_guide(self):
        """Visual guide for choosing the right metric"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Choosing the Right Distance Metric', fontsize=16, fontweight='bold')

        # 1. Decision tree
        ax = axes[0, 0]
        ax.axis('off')

        decision_text = '''
╔════════════════════════════════════════════╗
║    METRIC SELECTION DECISION TREE          ║
╚════════════════════════════════════════════╝

Does scale/magnitude matter?
    │
    ├─ YES → Does direction matter too?
    │         │
    │         ├─ YES → DOT PRODUCT
    │         │         (e.g., collaborative filtering)
    │         │
    │         └─ NO  → EUCLIDEAN (L2)
    │                  (e.g., measurements, features)
    │
    └─ NO  → COSINE SIMILARITY
             (e.g., text, semantic search)


Quick Rules:
  • Similar features/counts → Euclidean
  • Similar proportions → Cosine
  • Both matter → Dot Product
'''

        ax.text(0.1, 0.5, decision_text, fontsize=10, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.set_title('Decision Tree', fontweight='bold', pad=20)

        # 2. Use case matrix
        ax = axes[0, 1]
        ax.axis('off')

        use_cases = [
            ['Use Case', 'Euclidean', 'Cosine', 'Dot Product'],
            ['', '', '', ''],
            ['E-commerce', '✓✓✓', '✓', '✓✓'],
            ['Image features', '✓✓✓', '✓✓', '✓'],
            ['Document similarity', '✓', '✓✓✓', '✓✓'],
            ['User ratings', '✓', '✓', '✓✓✓'],
            ['Semantic search', '✓', '✓✓✓', '✓✓'],
            ['Recommendation', '✓✓', '✓', '✓✓✓'],
            ['Clustering', '✓✓✓', '✓✓', '✓'],
            ['Text classification', '✓', '✓✓✓', '✓✓'],
        ]

        table = ax.table(cellText=use_cases, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows alternately
        for i in range(2, len(use_cases)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        ax.set_title('Use Case Matrix\n✓✓✓ = Excellent, ✓✓ = Good, ✓ = OK',
                    fontweight='bold', pad=20)

        # 3. Performance characteristics
        ax = axes[1, 0]

        metrics = ['Euclidean\n(L2)', 'Cosine\nSimilarity', 'Dot\nProduct',
                  'Manhattan\n(L1)', 'Chebyshev\n(L∞)']

        # Ratings out of 10
        speed = [9, 8, 10, 9, 10]  # Computation speed
        interpretability = [8, 9, 7, 9, 6]  # How easy to understand
        robustness = [6, 9, 7, 8, 5]  # Robustness to outliers

        x = np.arange(len(metrics))
        width = 0.25

        bars1 = ax.bar(x - width, speed, width, label='Speed', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, interpretability, width, label='Interpretability',
                      color='#2ecc71', alpha=0.8)
        bars3 = ax.bar(x + width, robustness, width, label='Robustness',
                      color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Rating (out of 10)')
        ax.set_title('Performance Characteristics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 11])

        # 4. Real-world example scenarios
        ax = axes[1, 1]
        ax.axis('off')

        examples_text = '''
╔════════════════════════════════════════════╗
║        REAL-WORLD EXAMPLES                 ║
╚════════════════════════════════════════════╝

1. E-COMMERCE PRODUCT SEARCH
   Query: [price: 500, rating: 4.5, reviews: 100]
   → Use EUCLIDEAN
   Why: Feature values matter

2. NEWS ARTICLE CLASSIFICATION
   Query: [sports: 80%, tech: 15%, other: 5%]
   → Use COSINE
   Why: Proportions matter, not absolute counts

3. MOVIE RECOMMENDATIONS
   Query: User ratings [4, 5, 2, 0, 5]
   → Use DOT PRODUCT
   Why: Both rating values and agreement matter

4. HOUSE PRICE PREDICTION
   Query: [beds: 3, baths: 2, sqft: 2000]
   → Use EUCLIDEAN
   Why: Physical measurements

5. SEARCH QUERY SIMILARITY
   Query: "machine learning tutorial"
   → Use COSINE
   Why: Word frequency proportions

6. COLLABORATIVE FILTERING
   Query: User purchase history
   → Use DOT PRODUCT
   Why: Purchase frequency AND overlap matter
'''

        ax.text(0.1, 0.5, examples_text, fontsize=9, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        ax.set_title('Real-World Examples', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('visualizations/19_metric_selection_guide.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/19_metric_selection_guide.png")
        plt.close()

    def create_all_distance_visualizations(self):
        """Generate all distance metric visualizations"""
        print("=" * 70)
        print("🎨 Generating Distance Metrics Visualizations")
        print("=" * 70)

        print("\n1. Three Metrics Overview (AWS Slide)...")
        self.plot_three_metrics_comparison()

        print("\n2. Metric Comparisons...")
        self.plot_metric_comparisons()

        print("\n3. Metric Selection Guide...")
        self.plot_use_case_guide()

        print("\n" + "=" * 70)
        print("✅ All Distance Metrics Visualizations Created!")
        print("=" * 70)
        print("\nNew visualizations:")
        print("  17_distance_metrics_overview.png")
        print("  18_metric_comparisons.png")
        print("  19_metric_selection_guide.png")


def main():
    """Main visualization workflow"""
    visualizer = DistanceMetricsVisualizer()
    visualizer.create_all_distance_visualizations()


if __name__ == "__main__":
    main()
