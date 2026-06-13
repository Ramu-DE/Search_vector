#!/usr/bin/env python3
"""
Hybrid Search Visualizations
Educational visualizations for hybrid search concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os


class HybridSearchVisualizer:
    """Visualize hybrid search concepts"""

    def __init__(self):
        """Initialize visualizer"""
        self.fig_size = (14, 10)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.fig_size
        os.makedirs('visualizations', exist_ok=True)

    def plot_hybrid_architecture(self):
        """Visualize hybrid search architecture"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Hybrid Search Architecture - Best of Both Worlds',
                     fontsize=16, fontweight='bold')

        # Main architecture diagram
        ax_main = plt.subplot(2, 2, (1, 2))
        ax_main.axis('off')

        arch_text = '''
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                              │
│            "expensive apple products"                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ KEYWORD │ │ SPARSE  │ │ DENSE   │
  │ (BM25)  │ │ (TF-IDF)│ │(BERT)   │
  └────┬────┘ └────┬────┘ └────┬────┘
       │           │           │
       │ Exact     │ Term      │ Semantic
       │ Match     │ Weighted  │ Context
       │           │           │
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │  0.85   │ │  0.72   │ │  0.91   │
  │ Score   │ │ Score   │ │ Score   │
  └────┬────┘ └────┬────┘ └────┬────┘
       │           │           │
       └─────────┬─┴───────────┘
                 ▼
       ┌─────────────────────┐
       │ SCORE NORMALIZATION │
       │  & COMBINATION      │
       │                     │
       │ 0.3×K + 0.3×S + 0.4×D│
       └─────────┬───────────┘
                 ▼
       ┌─────────────────────┐
       │ UNIFIED RANKED      │
       │ RESULTS             │
       │                     │
       │ 1. Doc A (0.89)     │
       │ 2. Doc B (0.76)     │
       │ 3. Doc C (0.68)     │
       └─────────────────────┘
'''
        ax_main.text(0.1, 0.5, arch_text, fontsize=9, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Score normalization methods
        ax_norm = plt.subplot(2, 2, 3)
        methods = ['Min-Max', 'Z-Score', 'L2 Norm']
        before = [3.5, 0.05, 0.85]
        after = [0.95, 0.85, 0.92]

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax_norm.bar(x - width/2, before, width, label='Before',
                           color='lightcoral', alpha=0.8)
        bars2 = ax_norm.bar(x + width/2, after, width, label='After',
                           color='lightgreen', alpha=0.8)

        ax_norm.set_ylabel('Score')
        ax_norm.set_title('Score Normalization\n(Different scales → [0,1])')
        ax_norm.set_xticks(x)
        ax_norm.set_xticklabels(methods)
        ax_norm.legend()
        ax_norm.grid(True, alpha=0.3, axis='y')
        ax_norm.set_ylim([0, 1.1])

        # Combination methods comparison
        ax_comb = plt.subplot(2, 2, 4)

        # Simulate different combination results
        docs = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4']
        arith = [0.75, 0.68, 0.82, 0.71]
        weighted = [0.79, 0.64, 0.85, 0.69]
        harmonic = [0.71, 0.59, 0.80, 0.66]

        x = np.arange(len(docs))
        width = 0.25

        ax_comb.bar(x - width, arith, width, label='Arithmetic', color='#3498db', alpha=0.8)
        ax_comb.bar(x, weighted, width, label='Weighted (0.3+0.3+0.4)',
                   color='#e74c3c', alpha=0.8)
        ax_comb.bar(x + width, harmonic, width, label='Harmonic',
                   color='#2ecc71', alpha=0.8)

        ax_comb.set_ylabel('Combined Score')
        ax_comb.set_title('Combination Methods Comparison')
        ax_comb.set_xticks(x)
        ax_comb.set_xticklabels(docs)
        ax_comb.legend(fontsize=8)
        ax_comb.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('visualizations/12_hybrid_architecture.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/12_hybrid_architecture.png")
        plt.close()

    def plot_combination_methods(self):
        """Visualize different score combination methods"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Score Combination Methods', fontsize=16, fontweight='bold')

        # Sample scores for 5 documents from 3 methods
        keyword_scores = np.array([0.9, 0.3, 0.7, 0.1, 0.6])
        sparse_scores = np.array([0.6, 0.8, 0.5, 0.2, 0.7])
        dense_scores = np.array([0.7, 0.7, 0.8, 0.9, 0.5])

        docs = ['Doc A', 'Doc B', 'Doc C', 'Doc D', 'Doc E']
        x = np.arange(len(docs))

        # 1. Individual scores
        ax = axes[0, 0]
        width = 0.25
        ax.bar(x - width, keyword_scores, width, label='Keyword', color='#3498db', alpha=0.8)
        ax.bar(x, sparse_scores, width, label='Sparse', color='#e67e22', alpha=0.8)
        ax.bar(x + width, dense_scores, width, label='Dense', color='#2ecc71', alpha=0.8)
        ax.set_title('Individual Method Scores')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(docs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # 2. Arithmetic Mean (equal weight)
        ax = axes[0, 1]
        arith_mean = (keyword_scores + sparse_scores + dense_scores) / 3
        bars = ax.bar(docs, arith_mean, color='#9b59b6', alpha=0.8)
        ax.set_title('Arithmetic Mean\n(K + S + D) / 3')
        ax.set_ylabel('Combined Score')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        # Add values on bars
        for bar, val in zip(bars, arith_mean):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 3. Weighted Sum (0.3, 0.3, 0.4)
        ax = axes[0, 2]
        weighted = 0.3 * keyword_scores + 0.3 * sparse_scores + 0.4 * dense_scores
        bars = ax.bar(docs, weighted, color='#e74c3c', alpha=0.8)
        ax.set_title('Weighted Sum\n0.3K + 0.3S + 0.4D')
        ax.set_ylabel('Combined Score')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        for bar, val in zip(bars, weighted):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 4. Harmonic Mean
        ax = axes[1, 0]
        # Avoid division by zero
        harmonic = 3 / (1/(keyword_scores + 1e-10) + 1/(sparse_scores + 1e-10) +
                       1/(dense_scores + 1e-10))
        bars = ax.bar(docs, harmonic, color='#27ae60', alpha=0.8)
        ax.set_title('Harmonic Mean\n3 / (1/K + 1/S + 1/D)\nPenalizes discrepancies')
        ax.set_ylabel('Combined Score')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        for bar, val in zip(bars, harmonic):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 5. Geometric Mean
        ax = axes[1, 1]
        geometric = np.power(keyword_scores * sparse_scores * dense_scores, 1/3)
        bars = ax.bar(docs, geometric, color='#f39c12', alpha=0.8)
        ax.set_title('Geometric Mean\n(K × S × D)^(1/3)')
        ax.set_ylabel('Combined Score')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        for bar, val in zip(bars, geometric):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 6. Comparison table
        ax = axes[1, 2]
        ax.axis('off')

        # Calculate all methods
        methods_scores = {
            'Arithmetic': arith_mean,
            'Weighted': weighted,
            'Harmonic': harmonic,
            'Geometric': geometric
        }

        # Find winner for each document
        winners = []
        for i in range(len(docs)):
            doc_scores = {method: scores[i] for method, scores in methods_scores.items()}
            winner = max(doc_scores, key=doc_scores.get)
            winners.append(winner)

        # Create summary table
        table_data = [['Method', 'When to Use']]
        table_data.append(['Arithmetic', 'Equal importance'])
        table_data.append(['Weighted', 'Tunable priorities'])
        table_data.append(['Harmonic', 'All must be good'])
        table_data.append(['Geometric', 'Multiplicative effect'])
        table_data.append(['', ''])
        table_data.append(['Best for:', ''])
        for doc, winner in zip(docs, winners):
            table_data.append([doc, winner])

        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Color header
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Method Selection Guide', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('visualizations/13_combination_methods.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/13_combination_methods.png")
        plt.close()

    def plot_score_normalization(self):
        """Visualize score normalization process"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Score Normalization - Making Scores Comparable',
                     fontsize=16, fontweight='bold')

        # Raw scores from different methods (different scales!)
        docs = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5']
        keyword_raw = np.array([3.5, 1.2, 2.8, 0.5, 2.1])  # TF-IDF scores
        sparse_raw = np.array([0.65, 0.82, 0.54, 0.23, 0.71])  # Cosine sim
        dense_raw = np.array([0.91, 0.75, 0.88, 0.95, 0.68])  # Cosine sim

        # 1. Raw scores (different scales)
        ax = axes[0, 0]
        x = np.arange(len(docs))
        width = 0.25

        ax.bar(x - width, keyword_raw, width, label='Keyword (0-5)',
               color='#e74c3c', alpha=0.8)
        ax.bar(x, sparse_raw, width, label='Sparse (0-1)',
               color='#3498db', alpha=0.8)
        ax.bar(x + width, dense_raw, width, label='Dense (0-1)',
               color='#2ecc71', alpha=0.8)

        ax.set_title('Problem: Different Scales\nKeyword uses 0-5, others use 0-1')
        ax.set_ylabel('Raw Score')
        ax.set_xticks(x)
        ax.set_xticklabels(docs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0.5, 1.1, 'Scale mismatch!', color='red', fontweight='bold')

        # 2. Min-Max normalization
        ax = axes[0, 1]

        def minmax_norm(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s == min_s:
                return np.ones_like(scores)
            return (scores - min_s) / (max_s - min_s)

        keyword_norm = minmax_norm(keyword_raw)
        sparse_norm = minmax_norm(sparse_raw)
        dense_norm = minmax_norm(dense_raw)

        ax.bar(x - width, keyword_norm, width, label='Keyword',
               color='#e74c3c', alpha=0.8)
        ax.bar(x, sparse_norm, width, label='Sparse',
               color='#3498db', alpha=0.8)
        ax.bar(x + width, dense_norm, width, label='Dense',
               color='#2ecc71', alpha=0.8)

        ax.set_title('Min-Max Normalization\n(x - min) / (max - min)')
        ax.set_ylabel('Normalized Score [0, 1]')
        ax.set_xticks(x)
        ax.set_xticklabels(docs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        # 3. Combined scores comparison
        ax = axes[1, 0]

        # Without normalization (broken!)
        combined_raw = (keyword_raw + sparse_raw + dense_raw) / 3

        # With normalization (correct!)
        combined_norm = (keyword_norm + sparse_norm + dense_norm) / 3

        x_groups = np.arange(len(docs))
        width = 0.35

        bars1 = ax.bar(x_groups - width/2, combined_raw, width,
                       label='Without Norm (WRONG)', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x_groups + width/2, combined_norm, width,
                       label='With Norm (CORRECT)', color='#27ae60', alpha=0.8)

        ax.set_title('Impact on Combined Scores')
        ax.set_ylabel('Combined Score')
        ax.set_xticks(x_groups)
        ax.set_xticklabels(docs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight the difference
        ax.text(0.5, 1.8, '← Keyword dominates!\n(larger scale)',
               color='red', fontsize=9)
        ax.text(3.5, 0.8, '← Balanced! ✓\n(equal scales)',
               color='green', fontsize=9)

        # 4. Normalization methods comparison
        ax = axes[1, 1]
        ax.axis('off')

        norm_text = '''
Normalization Methods:

1. MIN-MAX (Most Common)
   Formula: (x - min) / (max - min)
   Result: [0, 1] range
   Pros: Simple, preserves distribution
   Cons: Sensitive to outliers

2. Z-SCORE (Standard)
   Formula: (x - mean) / std
   Result: Mean=0, Std=1
   Pros: Handles outliers well
   Cons: Can produce negative values

3. L2 NORMALIZATION
   Formula: x / ||x||₂
   Result: Unit vector
   Pros: Good for cosine similarity
   Cons: Doesn't bound to [0,1]

Why Normalize?
  ✓ Makes scores comparable
  ✓ Prevents one method from dominating
  ✓ Fair weight combination
  ✓ Better hybrid results

Example:
  Keyword: 3.5 → 0.95  (scale down)
  Sparse:  0.65 → 0.71 (slight adjust)
  Dense:   0.91 → 0.93 (slight adjust)

  Now they're comparable!
'''

        ax.text(0.1, 0.5, norm_text, fontsize=9, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig('visualizations/14_score_normalization.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/14_score_normalization.png")
        plt.close()

    def plot_method_strengths(self):
        """Visualize when each method excels"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('When Each Search Method Excels', fontsize=16, fontweight='bold')

        # 1. Query type performance
        ax = axes[0, 0]

        query_types = ['Exact\nMatch', 'Synonym', 'Concept', 'Rare\nTerms', 'Typo\nTolerant']
        keyword_perf = [0.95, 0.45, 0.35, 0.85, 0.25]
        sparse_perf = [0.88, 0.52, 0.42, 0.78, 0.35]
        dense_perf = [0.75, 0.92, 0.88, 0.55, 0.82]
        hybrid_perf = [0.93, 0.89, 0.85, 0.82, 0.71]

        x = np.arange(len(query_types))
        width = 0.2

        ax.bar(x - 1.5*width, keyword_perf, width, label='Keyword', color='#3498db', alpha=0.8)
        ax.bar(x - 0.5*width, sparse_perf, width, label='Sparse', color='#e67e22', alpha=0.8)
        ax.bar(x + 0.5*width, dense_perf, width, label='Dense', color='#2ecc71', alpha=0.8)
        ax.bar(x + 1.5*width, hybrid_perf, width, label='Hybrid', color='#9b59b6', alpha=0.8)

        ax.set_ylabel('Performance (NDCG)')
        ax.set_title('Performance by Query Type')
        ax.set_xticks(x)
        ax.set_xticklabels(query_types, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # Mark hybrid wins
        for i, (h, k, s, d) in enumerate(zip(hybrid_perf, keyword_perf, sparse_perf, dense_perf)):
            if h == max(h, k, s, d):
                ax.plot(i + 1.5*width, h + 0.03, 'v', color='gold', markersize=10)

        # 2. Dataset size impact
        ax = axes[0, 1]

        sizes = ['1K', '10K', '100K', '1M', '10M']
        sizes_x = np.arange(len(sizes))

        keyword_latency = [0.5, 2, 15, 120, 1200]
        sparse_latency = [0.8, 3, 18, 140, 1400]
        dense_latency = [2, 8, 45, 350, 3500]
        hybrid_latency = [2.5, 10, 50, 400, 4000]

        ax.plot(sizes_x, keyword_latency, 'o-', linewidth=2, markersize=8,
               label='Keyword', color='#3498db')
        ax.plot(sizes_x, sparse_latency, 's-', linewidth=2, markersize=8,
               label='Sparse', color='#e67e22')
        ax.plot(sizes_x, dense_latency, '^-', linewidth=2, markersize=8,
               label='Dense', color='#2ecc71')
        ax.plot(sizes_x, hybrid_latency, 'D-', linewidth=2, markersize=8,
               label='Hybrid', color='#9b59b6')

        ax.set_yscale('log')
        ax.set_ylabel('Latency (ms, log scale)')
        ax.set_xlabel('Dataset Size')
        ax.set_title('Scalability: Latency vs Dataset Size')
        ax.set_xticks(sizes_x)
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # 3. Quality vs Speed trade-off
        ax = axes[1, 0]

        methods = ['Keyword\nOnly', 'Sparse\nOnly', 'Dense\nOnly',
                  'K+S\nHybrid', 'K+D\nHybrid', 'S+D\nHybrid',
                  'K+S+D\nHybrid']

        quality = [0.72, 0.74, 0.85, 0.78, 0.83, 0.87, 0.91]  # NDCG@10
        speed = [2, 5, 45, 6, 35, 40, 42]  # Latency (ms)

        colors = ['#3498db', '#e67e22', '#2ecc71',
                 '#f39c12', '#9b59b6', '#e74c3c', '#1abc9c']

        scatter = ax.scatter(speed, quality, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

        # Add labels
        for i, method in enumerate(methods):
            ax.annotate(method, (speed[i], quality[i]),
                       fontsize=8, ha='center', va='center')

        # Add ideal zone
        from matplotlib.patches import Rectangle
        ideal = Rectangle((0, 0.85), 10, 0.15, alpha=0.1, facecolor='green')
        ax.add_patch(ideal)
        ax.text(5, 0.96, 'Ideal Zone', fontsize=10, ha='center',
               color='green', fontweight='bold')

        ax.set_xlabel('Latency (ms) - Lower is better →')
        ax.set_ylabel('Quality (NDCG@10) - Higher is better →')
        ax.set_title('Quality vs Speed Trade-off')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 50])
        ax.set_ylim([0.70, 1.0])

        # 4. Use case recommendations
        ax = axes[1, 1]
        ax.axis('off')

        rec_text = '''
╔═══════════════════════════════════════════════╗
║      WHEN TO USE EACH METHOD                  ║
╚═══════════════════════════════════════════════╝

KEYWORD ONLY (BM25):
  ✓ Product codes, SKUs, IDs
  ✓ Technical error codes
  ✓ Exact term critical
  ✓ Speed is paramount
  ✗ Poor with synonyms

SPARSE ONLY (TF-IDF):
  ✓ Similar to keyword but faster
  ✓ Interpretable results needed
  ✓ Domain terminology
  ✓ Low memory environments
  ✗ Limited semantic understanding

DENSE ONLY (Embeddings):
  ✓ Conceptual searches
  ✓ Synonym-heavy queries
  ✓ Cross-lingual search
  ✓ Conversational queries
  ✗ Slow for large scale
  ✗ Not interpretable

HYBRID (K+S+D) - RECOMMENDED:
  ✓ Production applications
  ✓ Best overall quality
  ✓ Covers all query types
  ✓ Robust to different inputs
  ⚠ Higher latency
  ⚠ More complex setup

Typical Weights:
  • Equal (0.33, 0.33, 0.33) - Safe default
  • Semantic (0.2, 0.3, 0.5) - Focus on meaning
  • Exact (0.5, 0.3, 0.2) - Focus on terms
  • Speed (0.4, 0.4, 0.2) - Faster, good enough
'''

        ax.text(0.1, 0.5, rec_text, fontsize=8, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.savefig('visualizations/15_method_strengths.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/15_method_strengths.png")
        plt.close()

    def plot_weight_tuning(self):
        """Visualize weight tuning effects"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weight Tuning - Finding the Right Balance',
                     fontsize=16, fontweight='bold')

        # Simulate quality at different weight combinations
        # weights for (keyword, sparse, dense)
        weight_configs = [
            (1.0, 0.0, 0.0, "Keyword\nOnly"),
            (0.0, 1.0, 0.0, "Sparse\nOnly"),
            (0.0, 0.0, 1.0, "Dense\nOnly"),
            (0.33, 0.33, 0.33, "Equal\n(0.33,0.33,0.33)"),
            (0.5, 0.3, 0.2, "Keyword\nFocus"),
            (0.2, 0.3, 0.5, "Semantic\nFocus"),
            (0.3, 0.3, 0.4, "Balanced\n(Recommended)"),
        ]

        quality_scores = [0.72, 0.74, 0.85, 0.82, 0.76, 0.88, 0.91]
        latency_ms = [2, 5, 45, 15, 8, 30, 25]

        # 1. Weight configurations comparison
        ax = axes[0, 0]

        configs = [w[3] for w in weight_configs]
        x = np.arange(len(configs))

        bars = ax.bar(x, quality_scores, color='#3498db', alpha=0.7)

        # Color the recommended one
        bars[6].set_color('#27ae60')
        bars[6].set_edgecolor('black')
        bars[6].set_linewidth(2)

        ax.set_ylabel('Quality (NDCG@10)')
        ax.set_title('Quality by Weight Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.65, 1.0])

        # Add values
        for bar, val in zip(bars, quality_scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # Mark recommended
        ax.text(6, 0.94, '★ BEST', ha='center', fontsize=10,
               color='green', fontweight='bold')

        # 2. Weight triangle visualization
        ax = axes[0, 1]
        ax.axis('equal')

        # Plot ternary-ish plot (simplified)
        # Each config as a point colored by quality
        keyword_weights = [w[0] for w in weight_configs]
        dense_weights = [w[2] for w in weight_configs]

        scatter = ax.scatter(keyword_weights, dense_weights,
                           s=500, c=quality_scores, cmap='RdYlGn',
                           vmin=0.65, vmax=1.0, alpha=0.8,
                           edgecolors='black', linewidth=2)

        # Add labels
        for i, (k, d, label, q) in enumerate(zip(keyword_weights, dense_weights,
                                                  configs, quality_scores)):
            ax.annotate(f'{label}\n{q:.2f}', (k, d),
                       fontsize=7, ha='center', va='center')

        ax.set_xlabel('Keyword Weight →')
        ax.set_ylabel('Dense Weight →')
        ax.set_title('Weight Space Exploration\n(Sparse = 1 - K - D)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Quality', rotation=270, labelpad=20)

        # 3. Sensitivity analysis
        ax = axes[1, 0]

        # Fix sparse=0.3, vary keyword and dense
        keyword_range = np.linspace(0, 0.7, 50)
        dense_range = 0.7 - keyword_range  # sparse=0.3, k+d=0.7

        # Simulate quality (peak at 0.3 keyword, 0.4 dense)
        quality_curve = 0.6 + 0.3 * np.exp(-((keyword_range - 0.3)**2 + (dense_range - 0.4)**2) / 0.05)

        ax.plot(keyword_range, quality_curve, linewidth=3, color='#e74c3c')
        ax.axvline(x=0.3, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(y=0.91, color='green', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Keyword Weight (Sparse=0.3, Dense=0.7-K)')
        ax.set_ylabel('Quality (NDCG@10)')
        ax.set_title('Sensitivity Analysis\nHow quality changes with keyword weight')
        ax.grid(True, alpha=0.3)
        ax.text(0.3, 0.93, '← Optimal: 0.3', fontsize=10, color='green', fontweight='bold')

        # Shade optimal region
        optimal_mask = (keyword_range >= 0.25) & (keyword_range <= 0.35)
        ax.fill_between(keyword_range, 0.6, quality_curve,
                        where=optimal_mask, alpha=0.2, color='green')

        # 4. Tuning guide
        ax = axes[1, 1]
        ax.axis('off')

        tuning_text = '''
╔═══════════════════════════════════════════════╗
║       WEIGHT TUNING GUIDE                     ║
╚═══════════════════════════════════════════════╝

Starting Point (Safe Default):
  weights = (0.33, 0.33, 0.33)
  All methods equal importance

Tuning Process:

1. MEASURE BASELINE
   Run eval on test queries
   Record: NDCG@10, latency, user feedback

2. ADJUST FOR YOUR USE CASE
   • E-commerce: (0.4, 0.3, 0.3)
     More keyword for SKUs

   • Q&A/Support: (0.2, 0.3, 0.5)
     More semantic for concepts

   • Technical docs: (0.5, 0.3, 0.2)
     Exact terms matter most

3. VALIDATE
   A/B test with real users
   Monitor: CTR, time-to-answer, satisfaction

4. ITERATE
   Adjust in 0.05 increments
   Re-measure after each change

Rules of Thumb:
  • Sum must equal 1.0
  • Keep all weights > 0.1
    (don't zero out a method)
  • Dense weight usually 0.3-0.5
    (semantic understanding helps)
  • Keyword weight 0.2-0.5
    (exact matching matters)

Recommended:
  weights = (0.3, 0.3, 0.4) ✓
  Balances all strengths
'''

        ax.text(0.1, 0.5, tuning_text, fontsize=8, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig('visualizations/16_weight_tuning.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/16_weight_tuning.png")
        plt.close()

    def create_all_hybrid_visualizations(self):
        """Generate all hybrid search visualizations"""
        print("=" * 70)
        print("🎨 Generating Hybrid Search Visualizations")
        print("=" * 70)

        print("\n1. Hybrid Architecture...")
        self.plot_hybrid_architecture()

        print("\n2. Combination Methods...")
        self.plot_combination_methods()

        print("\n3. Score Normalization...")
        self.plot_score_normalization()

        print("\n4. Method Strengths...")
        self.plot_method_strengths()

        print("\n5. Weight Tuning...")
        self.plot_weight_tuning()

        print("\n" + "=" * 70)
        print("✅ All Hybrid Visualizations Created!")
        print("=" * 70)
        print("\nNew visualizations:")
        print("  12_hybrid_architecture.png")
        print("  13_combination_methods.png")
        print("  14_score_normalization.png")
        print("  15_method_strengths.png")
        print("  16_weight_tuning.png")


def main():
    """Main visualization workflow"""
    visualizer = HybridSearchVisualizer()
    visualizer.create_all_hybrid_visualizations()


if __name__ == "__main__":
    main()
