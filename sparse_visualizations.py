#!/usr/bin/env python3
"""
Sparse Encoding Visualizations
Educational visualizations comparing dense vs sparse vector representations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
from sparse_encoding import SparseEncoder, LearnedSparseEncoder


class SparseVectorVisualizer:
    """Visualize sparse encoding concepts and comparisons"""

    def __init__(self):
        """Initialize visualizer"""
        self.fig_size = (14, 10)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.fig_size
        os.makedirs('visualizations', exist_ok=True)

    def plot_dense_vs_sparse(self):
        """Compare dense and sparse vector representations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dense vs Sparse Vector Encoding', fontsize=16, fontweight='bold')

        # 1. Dense Vector Representation
        ax = axes[0, 0]
        dense_dim = 50
        dense_vector = np.random.randn(dense_dim) * 0.3  # All positions have values

        x = np.arange(dense_dim)
        colors = ['red' if v < 0 else 'blue' for v in dense_vector]
        ax.bar(x, dense_vector, color=colors, alpha=0.7, width=1.0)

        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Value')
        ax.set_title('Dense Vector (e.g., BERT embeddings)\n384-768 dimensions, ALL non-zero')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linewidth=0.5)

        # Add statistics
        non_zero = np.sum(np.abs(dense_vector) > 0.01)
        ax.text(0.02, 0.98, f'Non-zero: {non_zero}/{dense_dim} ({non_zero/dense_dim*100:.0f}%)',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. Sparse Vector Representation
        ax = axes[0, 1]
        sparse_dim = 50
        sparse_vector = np.zeros(sparse_dim)
        # Only a few positions have values
        active_indices = [5, 12, 18, 23, 31, 42, 47]
        for idx in active_indices:
            sparse_vector[idx] = np.random.uniform(0.3, 1.0)

        colors = ['green' if v > 0 else 'lightgray' for v in sparse_vector]
        ax.bar(x, sparse_vector, color=colors, alpha=0.7, width=1.0)

        ax.set_xlabel('Dimension Index (Vocabulary Position)')
        ax.set_ylabel('Weight (TF-IDF score)')
        ax.set_title('Sparse Vector (TF-IDF)\n10,000-30,000 dimensions, ~1-5% non-zero')
        ax.grid(True, alpha=0.3, axis='y')

        non_zero_sparse = len(active_indices)
        ax.text(0.02, 0.98, f'Non-zero: {non_zero_sparse}/{sparse_dim} ({non_zero_sparse/sparse_dim*100:.0f}%)',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # 3. Memory Usage Comparison
        ax = axes[1, 0]
        methods = ['Dense\n(768-dim)', 'Sparse\n(10k vocab,\n1% active)', 'Sparse\n(30k vocab,\n1% active)']
        memory_mb = [3.1, 0.4, 1.2]  # Approximate memory per vector in KB
        colors_mem = ['#e74c3c', '#3498db', '#2ecc71']

        bars = ax.bar(methods, memory_mb, color=colors_mem, alpha=0.7)
        ax.set_ylabel('Memory per Vector (KB)')
        ax.set_title('Memory Usage Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, memory_mb):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val} KB',
                   ha='center', va='bottom', fontweight='bold')

        # 4. Interpretability Comparison
        ax = axes[1, 1]
        ax.axis('off')

        # Create comparison table
        comparison_data = [
            ['Property', 'Dense Vectors', 'Sparse Vectors'],
            ['Dimensions', '384-1536', '10K-30K'],
            ['Non-zero %', '~100%', '~1-5%'],
            ['Memory/vector', '1.5-6 KB', '0.4-1.2 KB'],
            ['Interpretable', '❌ No', '✅ Yes'],
            ['Exact match', '❌ Poor', '✅ Excellent'],
            ['Semantic', '✅ Excellent', '⚠️ Moderate'],
            ['Speed', '⚡ Fast (ANN)', '⚡⚡ Very Fast'],
            ['RAM at query', 'High', 'Zero increase']
        ]

        table = ax.table(cellText=comparison_data, cellLoc='left', loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows alternately
        for i in range(1, len(comparison_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        plt.tight_layout()
        plt.savefig('visualizations/07_dense_vs_sparse.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/07_dense_vs_sparse.png")
        plt.close()

    def plot_sparse_encoding_process(self):
        """Visualize the sparse encoding process step by step"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Sparse Encoding Process: Text → Sparse Vector', fontsize=16, fontweight='bold')

        # Example text
        text = "Apple products are expensive"

        # Step 1: Tokenization
        ax1 = plt.subplot(3, 2, 1)
        ax1.axis('off')
        ax1.text(0.5, 0.8, 'Step 1: Tokenization', ha='center', fontsize=14, fontweight='bold')
        ax1.text(0.5, 0.6, f'Original Text:\n"{text}"', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax1.text(0.5, 0.3, 'Tokens:\n["apple", "products", "expensive"]', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Step 2: Vocabulary Lookup
        ax2 = plt.subplot(3, 2, 2)
        ax2.axis('off')
        ax2.text(0.5, 0.9, 'Step 2: Vocabulary Mapping', ha='center', fontsize=14, fontweight='bold')

        vocab_text = '''Vocabulary (30,522 terms):

Position 1024: "apple"
Position 7823: "products"
Position 4567: "expensive"
Position 9234: "gadget"
...'''
        ax2.text(0.5, 0.4, vocab_text, ha='center', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Step 3: TF-IDF Calculation
        ax3 = plt.subplot(3, 2, 3)
        terms = ['apple', 'products', 'expensive']
        tf_values = [0.58, 0.58, 0.58]  # Term frequency
        idf_values = [2.3, 1.8, 2.1]  # Inverse document frequency
        tfidf_values = [tf * idf for tf, idf in zip(tf_values, idf_values)]

        x = np.arange(len(terms))
        width = 0.25

        ax3.bar(x - width, tf_values, width, label='TF', color='lightblue', alpha=0.8)
        ax3.bar(x, idf_values, width, label='IDF', color='lightcoral', alpha=0.8)
        ax3.bar(x + width, tfidf_values, width, label='TF-IDF', color='lightgreen', alpha=0.8)

        ax3.set_ylabel('Score')
        ax3.set_title('Step 3: TF-IDF Weight Calculation')
        ax3.set_xticks(x)
        ax3.set_xticklabels(terms)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Step 4: Sparse Vector Creation
        ax4 = plt.subplot(3, 2, 4)
        vocab_size = 30
        sparse_vec = np.zeros(vocab_size)
        # Map our terms to positions
        positions = [5, 12, 18]  # apple, products, expensive
        for pos, val in zip(positions, tfidf_values):
            sparse_vec[pos] = val

        colors = ['green' if v > 0 else 'lightgray' for v in sparse_vec]
        ax4.bar(range(vocab_size), sparse_vec, color=colors, alpha=0.7)
        ax4.set_xlabel('Vocabulary Position')
        ax4.set_ylabel('Weight')
        ax4.set_title('Step 4: Sparse Vector (30K dims, 3 active)')
        ax4.grid(True, alpha=0.3, axis='y')

        # Highlight active positions
        for pos, term, val in zip(positions, terms, tfidf_values):
            ax4.annotate(f'{term}\n{val:.2f}',
                        xy=(pos, val), xytext=(pos, val + 0.3),
                        ha='center', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1))

        # Step 5: Dictionary Representation
        ax5 = plt.subplot(3, 2, 5)
        ax5.axis('off')
        ax5.text(0.5, 0.9, 'Step 5: Dictionary Format (Human-Readable)', ha='center',
                fontsize=14, fontweight='bold')

        dict_text = '''{
  "apple": 1.33,
  "products": 1.04,
  "expensive": 1.22,

  [30,519 other terms = 0]
}

Sparsity: 99.99%'''

        ax5.text(0.5, 0.4, dict_text, ha='center', fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Step 6: Comparison with Dense
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        ax6.text(0.5, 0.9, 'Comparison: Same Text, Dense Encoding', ha='center',
                fontsize=14, fontweight='bold')

        dense_text = '''Dense (BERT):
[0.712, 0.049, 0.914, 0.930,
 0.224, 0.913, 0.578, 0.364,
 0.123, 0.456, ..., 0.789]

768 values, ALL non-zero
Not interpretable ❌
Better semantic understanding ✅'''

        ax6.text(0.5, 0.4, dense_text, ha='center', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

        plt.tight_layout()
        plt.savefig('visualizations/08_sparse_encoding_process.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/08_sparse_encoding_process.png")
        plt.close()

    def plot_sparse_similarity(self):
        """Visualize sparse similarity scoring"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sparse Vector Similarity Scoring', fontsize=16, fontweight='bold')

        # Documents and query
        doc1_terms = {'apple': 7.45, 'products': 4.32, 'expensive': 3.21, 'gadget': 2.10}
        doc2_terms = {'apple': 5.56, 'day': 2.34, 'doctor': 2.12, 'health': 1.89}
        query_terms = {'apple': 7.89, 'headphones': 6.54, 'wireless': 3.21}

        # 1. Term overlap visualization
        ax = axes[0, 0]
        ax.axis('off')

        overlap_text = '''Query: "apple headphones"
{
  "apple": 7.89,
  "headphones": 6.54,
  "wireless": 3.21
}

Doc 1: "Apple products are expensive"
{
  "apple": 7.45,        ← MATCH!
  "products": 4.32,
  "expensive": 3.21,
  "gadget": 2.10
}

Score = 7.89 × 7.45 = 58.78

Only overlapping terms contribute!'''

        ax.text(0.1, 0.5, overlap_text, fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               verticalalignment='center')
        ax.set_title('Term Overlap Scoring', fontweight='bold', pad=20)

        # 2. Scoring formula visualization
        ax = axes[0, 1]
        terms = list(query_terms.keys())
        query_weights = [query_terms[t] for t in terms]
        doc1_weights = [doc1_terms.get(t, 0) for t in terms]

        x = np.arange(len(terms))
        width = 0.35

        bars1 = ax.bar(x - width/2, query_weights, width, label='Query', color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, doc1_weights, width, label='Doc 1', color='lightcoral', alpha=0.8)

        ax.set_xlabel('Terms')
        ax.set_ylabel('Weight')
        ax.set_title('Query vs Document Term Weights')
        ax.set_xticks(x)
        ax.set_xticklabels(terms)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight matching term
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Match')
        ax.text(0, max(query_weights) * 0.9, '✓ MATCH', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # 3. Multiple documents comparison
        ax = axes[1, 0]

        docs = ['Doc 1\n(Products)', 'Doc 2\n(Health)', 'Doc 3\n(Tech)']
        scores = [58.78, 43.87, 72.45]
        overlap_counts = [1, 1, 2]

        colors_bars = ['#e74c3c', '#f39c12', '#27ae60']
        bars = ax.bar(docs, scores, color=colors_bars, alpha=0.7)

        ax.set_ylabel('Similarity Score')
        ax.set_title('Sparse Similarity Scores for Query "apple headphones"')
        ax.grid(True, alpha=0.3, axis='y')

        # Add annotations
        for bar, score, overlap in zip(bars, scores, overlap_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}\n({overlap} terms)',
                   ha='center', va='bottom', fontweight='bold')

        # 4. Sparsity benefit visualization
        ax = axes[1, 1]

        # Comparison table
        ax.axis('off')

        table_data = [
            ['Metric', 'Dense', 'Sparse'],
            ['Non-zero terms', '768', '3-50'],
            ['Computation', 'ALL 768 dims', 'Only overlaps'],
            ['Example calc', '768 multiplies', '1 multiply'],
            ['', '768 additions', '1 addition'],
            ['Speed', '1.0x', '10-50x faster'],
            ['', '', ''],
            ['Storage (1M docs)', '3 GB', '300 MB'],
            ['Index type', 'HNSW', 'Inverted index']
        ]

        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        bbox=[0.1, 0, 0.8, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Highlight key rows
        table[(5, 2)].set_facecolor('#27ae60')
        table[(5, 2)].set_text_props(weight='bold', color='white')

        ax.set_title('Computation Efficiency: Dense vs Sparse', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('visualizations/09_sparse_similarity.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/09_sparse_similarity.png")
        plt.close()

    def plot_learned_sparse_expansion(self):
        """Visualize learned sparse encoding with term expansion"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learned Sparse Encoding (SPLADE-style)', fontsize=16, fontweight='bold')

        # 1. Base vs Expanded terms
        ax = axes[0, 0]

        base_terms = ['expensive', 'product', 'quality']
        base_weights = [0.85, 0.72, 0.68]

        expanded_terms = base_terms + ['costly', 'pricey', 'item', 'gadget']
        expanded_weights = base_weights + [0.45, 0.42, 0.38, 0.35]

        y_base = np.arange(len(base_terms))
        y_expanded = np.arange(len(expanded_terms))

        ax.barh(y_base, base_weights, color='lightblue', alpha=0.8, label='Base terms')
        ax.barh(y_expanded[len(base_terms):], expanded_weights[len(base_terms):],
               color='lightgreen', alpha=0.8, label='Expanded terms')

        ax.set_yticks(y_expanded)
        ax.set_yticklabels(expanded_terms)
        ax.set_xlabel('Weight')
        ax.set_title('Base TF-IDF vs Expanded Terms')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Add markers
        for i in range(len(base_terms), len(expanded_terms)):
            ax.text(expanded_weights[i] + 0.02, i, '🆕', fontsize=12)

        # 2. Expansion process
        ax = axes[0, 1]
        ax.axis('off')

        expansion_text = '''Text: "This product is expensive"

Step 1: Base TF-IDF
{
  "expensive": 0.85,
  "product": 0.72,
  "this": 0.45
}

Step 2: Neural Expansion
• "expensive" similar to:
  - "costly" (0.92 sim)
  - "pricey" (0.89 sim)
  - "high-price" (0.85 sim)

Step 3: Add Expanded Terms
{
  "expensive": 0.85,
  "product": 0.72,
  "costly": 0.45,     ← NEW
  "pricey": 0.42,     ← NEW
  "this": 0.45
}'''

        ax.text(0.1, 0.5, expansion_text, fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               verticalalignment='center')
        ax.set_title('Term Expansion Process', fontweight='bold', pad=20)

        # 3. Retrieval improvement
        ax = axes[1, 0]

        methods = ['BM25\n(keyword)', 'Dense\n(semantic)', 'Sparse\n(base)', 'Learned Sparse\n(expanded)']

        # Query: "costly gadgets" looking for "expensive products"
        recall_at_10 = [0.4, 0.85, 0.45, 0.92]
        precision_at_10 = [0.6, 0.75, 0.65, 0.88]

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width/2, recall_at_10, width, label='Recall@10', color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, precision_at_10, width, label='Precision@10', color='lightcoral', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Retrieval Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # Highlight best
        best_idx = 3
        ax.axvline(x=best_idx, color='green', linestyle='--', linewidth=2, alpha=0.3)

        # 4. Benefits summary
        ax = axes[1, 1]
        ax.axis('off')

        benefits_text = '''Benefits of Learned Sparse:

✅ Semantic Understanding
   "expensive" matches "costly"
   without exact word match

✅ Interpretability
   Can see WHY docs match
   (expanded term overlap)

✅ Efficiency
   Still sparse (~1-5% non-zero)
   Fast inverted index lookup

✅ Best of Both Worlds
   Semantic: Like dense vectors
   Speed: Like sparse TF-IDF

⚡ 10x faster than dense
🎯 95%+ of dense accuracy'''

        ax.text(0.5, 0.5, benefits_text, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               verticalalignment='center', horizontalalignment='center')
        ax.set_title('Key Benefits', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('visualizations/10_learned_sparse_expansion.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/10_learned_sparse_expansion.png")
        plt.close()

    def plot_hybrid_sparse_dense(self):
        """Visualize hybrid search combining sparse and dense"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hybrid Search: Sparse + Dense', fontsize=16, fontweight='bold')

        # 1. Architecture
        ax = axes[0, 0]
        ax.axis('off')

        arch_text = '''Hybrid Search Architecture:

Query: "Apple headphones wireless"
    │
    ├─→ Dense Encoder (BERT)
    │   [0.712, 0.049, 0.914, ...]
    │   ↓
    │   Dense Search (k-NN)
    │   Results: [Doc3, Doc1, Doc5, ...]
    │
    └─→ Sparse Encoder (TF-IDF)
        {"apple": 0.85, "headphones": 0.78}
        ↓
        Sparse Search (Inverted Index)
        Results: [Doc1, Doc3, Doc7, ...]

    ↓  Combine & Rerank ↓

Final Results: [Doc1, Doc3, Doc5, ...]
(Best of both methods)'''

        ax.text(0.1, 0.5, arch_text, fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
               verticalalignment='center')
        ax.set_title('Hybrid Architecture', fontweight='bold', pad=20)

        # 2. Score combination
        ax = axes[0, 1]

        docs = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5']
        sparse_scores = [0.85, 0.45, 0.92, 0.23, 0.67]
        dense_scores = [0.78, 0.91, 0.65, 0.88, 0.54]

        # Weighted combination (0.6 dense + 0.4 sparse)
        hybrid_scores = [0.6*d + 0.4*s for d, s in zip(dense_scores, sparse_scores)]

        x = np.arange(len(docs))
        width = 0.25

        ax.bar(x - width, sparse_scores, width, label='Sparse', color='lightblue', alpha=0.8)
        ax.bar(x, dense_scores, width, label='Dense', color='lightcoral', alpha=0.8)
        ax.bar(x + width, hybrid_scores, width, label='Hybrid', color='lightgreen', alpha=0.8)

        ax.set_xlabel('Documents')
        ax.set_ylabel('Score')
        ax.set_title('Score Combination (0.6×Dense + 0.4×Sparse)')
        ax.set_xticks(x)
        ax.set_xticklabels(docs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. When each method wins
        ax = axes[1, 0]

        scenarios = ['Exact\nMatch', 'Synonym', 'Conceptual', 'Rare\nTerms', 'Out of\nDomain']
        sparse_performance = [0.95, 0.45, 0.35, 0.85, 0.40]
        dense_performance = [0.75, 0.92, 0.88, 0.55, 0.75]

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax.bar(x - width/2, sparse_performance, width, label='Sparse', color='lightblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, dense_performance, width, label='Dense', color='lightcoral', alpha=0.8)

        ax.set_ylabel('Performance')
        ax.set_title('When Each Method Excels')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        # Highlight winners
        for i, (sparse, dense) in enumerate(zip(sparse_performance, dense_performance)):
            if sparse > dense:
                bars1[i].set_edgecolor('green')
                bars1[i].set_linewidth(3)
            else:
                bars2[i].set_edgecolor('green')
                bars2[i].set_linewidth(3)

        # 4. Real-world results
        ax = axes[1, 1]

        methods_comp = ['Dense\nOnly', 'Sparse\nOnly', 'Hybrid\n(Simple)', 'Hybrid\n(Learned)']
        ndcg_at_10 = [0.78, 0.72, 0.85, 0.91]
        latency_ms = [45, 12, 35, 38]

        # Create dual axis
        ax_ndcg = ax
        ax_latency = ax.twinx()

        x = np.arange(len(methods_comp))
        width = 0.35

        bars1 = ax_ndcg.bar(x - width/2, ndcg_at_10, width, label='NDCG@10', color='lightgreen', alpha=0.8)
        bars2 = ax_latency.bar(x + width/2, latency_ms, width, label='Latency', color='lightcoral', alpha=0.8)

        ax_ndcg.set_xlabel('Method')
        ax_ndcg.set_ylabel('NDCG@10 (Quality)', color='green')
        ax_latency.set_ylabel('Latency (ms)', color='red')
        ax_ndcg.set_title('Quality vs Speed Trade-off')
        ax_ndcg.set_xticks(x)
        ax_ndcg.set_xticklabels(methods_comp, fontsize=9)
        ax_ndcg.tick_params(axis='y', labelcolor='green')
        ax_latency.tick_params(axis='y', labelcolor='red')
        ax_ndcg.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, ndcg_at_10)):
            height = bar.get_height()
            ax_ndcg.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig('visualizations/11_hybrid_sparse_dense.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/11_hybrid_sparse_dense.png")
        plt.close()

    def create_all_sparse_visualizations(self):
        """Generate all sparse encoding visualizations"""
        print("=" * 70)
        print("🎨 Generating Sparse Encoding Visualizations")
        print("=" * 70)

        print("\n1. Dense vs Sparse Comparison...")
        self.plot_dense_vs_sparse()

        print("\n2. Sparse Encoding Process...")
        self.plot_sparse_encoding_process()

        print("\n3. Sparse Similarity Scoring...")
        self.plot_sparse_similarity()

        print("\n4. Learned Sparse Expansion...")
        self.plot_learned_sparse_expansion()

        print("\n5. Hybrid Sparse + Dense...")
        self.plot_hybrid_sparse_dense()

        print("\n" + "=" * 70)
        print("✅ All Sparse Visualizations Created!")
        print("=" * 70)
        print("\nNew visualizations:")
        print("  07_dense_vs_sparse.png")
        print("  08_sparse_encoding_process.png")
        print("  09_sparse_similarity.png")
        print("  10_learned_sparse_expansion.png")
        print("  11_hybrid_sparse_dense.png")


def main():
    """Main visualization workflow"""
    visualizer = SparseVectorVisualizer()
    visualizer.create_all_sparse_visualizations()


if __name__ == "__main__":
    main()
