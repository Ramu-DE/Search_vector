#!/usr/bin/env python3
"""
Sparse Encoding CLI Tool
Quick command-line interface for sparse encoding operations
"""

import argparse
import sys
from pathlib import Path
from sparse_encoding import SparseEncoder, LearnedSparseEncoder


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Sparse Encoding CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic demo
  python sparse_cli.py demo

  # Generate all visualizations
  python sparse_cli.py visualize

  # Encode a text
  python sparse_cli.py encode "apple products expensive"

  # Compare two texts
  python sparse_cli.py compare "apple products" "fruit store"

  # Show statistics
  python sparse_cli.py stats
        """
    )

    parser.add_argument(
        'command',
        choices=['demo', 'visualize', 'encode', 'compare', 'stats', 'help'],
        help='Command to execute'
    )

    parser.add_argument(
        'args',
        nargs='*',
        help='Arguments for the command'
    )

    parser.add_argument(
        '--max-features',
        type=int,
        default=100,
        help='Maximum vocabulary size (default: 100)'
    )

    args = parser.parse_args()

    # Execute command
    if args.command == 'demo':
        run_demo()
    elif args.command == 'visualize':
        run_visualize()
    elif args.command == 'encode':
        if not args.args:
            print("Error: 'encode' requires text argument")
            print("Example: python sparse_cli.py encode \"your text here\"")
            sys.exit(1)
        run_encode(' '.join(args.args), args.max_features)
    elif args.command == 'compare':
        if len(args.args) < 2:
            print("Error: 'compare' requires two text arguments")
            print("Example: python sparse_cli.py compare \"text 1\" \"text 2\"")
            sys.exit(1)
        text1 = args.args[0]
        text2 = ' '.join(args.args[1:])
        run_compare(text1, text2, args.max_features)
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'help':
        show_help()


def run_demo():
    """Run the complete demo"""
    print("🚀 Running Sparse Search Demo...\n")
    from demo_sparse_search import main as demo_main
    demo_main()


def run_visualize():
    """Generate all visualizations"""
    print("🎨 Generating Sparse Encoding Visualizations...\n")
    from sparse_visualizations import main as viz_main
    viz_main()


def run_encode(text, max_features):
    """Encode a single text"""
    print("=" * 70)
    print("Sparse Encoding")
    print("=" * 70)
    print(f"Text: '{text}'")
    print("-" * 70)

    # Sample corpus for fitting
    corpus = [
        "Apple products are expensive but high quality",
        "An apple a day keeps the doctor away",
        "The new iPhone is very expensive",
        "Healthy fruits include apples and oranges",
        "Technology gadgets can be costly"
    ]

    # Initialize encoder
    encoder = SparseEncoder(max_features=max_features)
    encoder.fit(corpus)

    # Encode
    sparse_dict, _ = encoder.encode(text)

    # Display results
    print(f"\nVocabulary size: {len(encoder.vocabulary)}")
    print(f"Active terms: {len(sparse_dict)}")
    print(f"Sparsity: {encoder.get_sparsity(sparse_dict):.1f}%")

    print(f"\nTerm weights:")
    for term, weight in list(sparse_dict.items())[:10]:
        print(f"  '{term}': {weight:.4f}")

    if len(sparse_dict) > 10:
        print(f"  ... and {len(sparse_dict) - 10} more terms")


def run_compare(text1, text2, max_features):
    """Compare two texts"""
    print("=" * 70)
    print("Text Comparison")
    print("=" * 70)
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print("-" * 70)

    # Sample corpus
    corpus = [
        "Apple products are expensive but high quality",
        "An apple a day keeps the doctor away",
        "The new iPhone is very expensive",
        text1,
        text2
    ]

    # Initialize encoder
    encoder = SparseEncoder(max_features=max_features)
    encoder.fit(corpus)

    # Compare
    comparison = encoder.compare_texts(text1, text2)

    # Display results
    print(f"\nSimilarity: {comparison['similarity']:.4f}")
    print(f"Text 1 terms: {comparison['text1_terms']}")
    print(f"Text 2 terms: {comparison['text2_terms']}")
    print(f"Overlapping terms: {comparison['overlap_terms']} ({comparison['overlap_ratio']:.1%})")

    if comparison['overlapping_terms']:
        print(f"\nShared terms:")
        for term, weights in list(comparison['overlapping_terms'].items())[:10]:
            print(f"  '{term}': {weights['weight1']:.4f} vs {weights['weight2']:.4f}")


def show_stats():
    """Show statistics about implementations"""
    print("=" * 70)
    print("Sparse Encoding Implementation Statistics")
    print("=" * 70)

    # Check files
    files = {
        'sparse_encoding.py': 'Core implementation',
        'sparse_visualizations.py': 'Visualization generation',
        'demo_sparse_search.py': 'Interactive demo',
        'SPARSE_ENCODING_GUIDE.md': 'User guide',
        'SPARSE_IMPLEMENTATION_SUMMARY.md': 'Implementation summary'
    }

    print("\n📁 Files:")
    for filename, description in files.items():
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size
            lines = len(path.read_text().splitlines())
            print(f"  ✅ {filename}")
            print(f"     {description}")
            print(f"     {size:,} bytes, {lines:,} lines")
        else:
            print(f"  ❌ {filename} (not found)")

    # Check visualizations
    viz_dir = Path('visualizations')
    if viz_dir.exists():
        sparse_viz = list(viz_dir.glob('0[7-9]_*.png')) + list(viz_dir.glob('1[0-1]_*.png'))
        print(f"\n🎨 Visualizations: {len(sparse_viz)} sparse encoding images")
        for viz in sorted(sparse_viz):
            size_kb = viz.stat().st_size / 1024
            print(f"  ✓ {viz.name} ({size_kb:.0f} KB)")
    else:
        print("\n🎨 Visualizations directory not found")

    print("\n" + "=" * 70)


def show_help():
    """Show detailed help"""
    help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                   SPARSE ENCODING CLI TOOL                           ║
╚══════════════════════════════════════════════════════════════════════╝

OVERVIEW:
  This CLI provides quick access to sparse encoding functionality,
  including demos, visualizations, and encoding operations.

COMMANDS:

  demo
      Run the complete interactive demo
      Shows: basic encoding, term expansion, comparisons, interpretability

      Example: python sparse_cli.py demo

  visualize
      Generate all 5 sparse encoding visualizations
      Creates: 07-11.png in visualizations/ directory

      Example: python sparse_cli.py visualize

  encode <text>
      Encode a text string and show sparse representation
      Shows: active terms, weights, sparsity percentage

      Example: python sparse_cli.py encode "expensive apple products"

  compare <text1> <text2>
      Compare two texts and show similarity
      Shows: similarity score, overlapping terms, weights

      Example: python sparse_cli.py compare "apple products" "fruit store"

  stats
      Show implementation statistics
      Shows: files, sizes, visualizations created

      Example: python sparse_cli.py stats

  help
      Show this help message

      Example: python sparse_cli.py help

OPTIONS:

  --max-features N
      Set vocabulary size (default: 100)
      Larger = more terms but more memory

      Example: python sparse_cli.py encode "text" --max-features 1000

QUICK START:

  1. Run the demo to see all features:
     $ python sparse_cli.py demo

  2. Generate visualizations:
     $ python sparse_cli.py visualize

  3. Try encoding your own text:
     $ python sparse_cli.py encode "your text here"

FILES:

  sparse_encoding.py           - Core implementation
  sparse_visualizations.py     - Visualization code
  demo_sparse_search.py        - Interactive demo
  SPARSE_ENCODING_GUIDE.md     - Complete user guide

VISUALIZATIONS:

  07_dense_vs_sparse.png           - Dense vs Sparse comparison
  08_sparse_encoding_process.png   - Encoding process steps
  09_sparse_similarity.png         - Similarity calculation
  10_learned_sparse_expansion.png  - Term expansion
  11_hybrid_sparse_dense.png       - Hybrid approach

For more information, read SPARSE_ENCODING_GUIDE.md

╚══════════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


if __name__ == "__main__":
    main()
