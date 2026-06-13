#!/usr/bin/env python3
"""
Sparse Encoding Implementation
Implements sparse vector representations using TF-IDF and learned sparse embeddings
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import json


class SparseEncoder:
    """
    Sparse vector encoder using TF-IDF and term expansion

    Unlike dense embeddings which produce vectors like:
        [0.712, 0.049, 0.914, ...] (all positions filled)

    Sparse encodings produce vectors like:
        {"apple": 0.85, "products": 0.72, "expensive": 0.68} (mostly zeros)
    """

    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize sparse encoder

        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range (1,1) for unigrams, (1,2) for unigrams+bigrams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=1,
            max_df=0.95
        )

        self.is_fitted = False
        self.vocabulary = None
        self.idf_values = None

    def fit(self, corpus: List[str]):
        """
        Fit the sparse encoder on a corpus

        Args:
            corpus: List of text documents
        """
        print(f"Fitting sparse encoder on {len(corpus)} documents...")
        self.vectorizer.fit(corpus)
        self.is_fitted = True

        # Store vocabulary and IDF values
        self.vocabulary = self.vectorizer.vocabulary_
        self.idf_values = dict(zip(
            self.vectorizer.get_feature_names_out(),
            self.vectorizer.idf_
        ))

        print(f"✓ Vocabulary size: {len(self.vocabulary)}")
        print(f"✓ N-gram range: {self.ngram_range}")

    def encode(self, text: str) -> Tuple[Dict[str, float], csr_matrix]:
        """
        Encode text into sparse representation

        Args:
            text: Input text

        Returns:
            Tuple of (sparse_dict, sparse_matrix)
            - sparse_dict: {term: weight} dictionary (human-readable)
            - sparse_matrix: scipy sparse matrix (for computation)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding")

        # Get TF-IDF sparse matrix
        sparse_matrix = self.vectorizer.transform([text])

        # Convert to dictionary (only non-zero values)
        sparse_dict = self._matrix_to_dict(sparse_matrix)

        return sparse_dict, sparse_matrix

    def encode_batch(self, texts: List[str]) -> Tuple[List[Dict[str, float]], csr_matrix]:
        """
        Encode multiple texts

        Args:
            texts: List of input texts

        Returns:
            Tuple of (list of sparse_dicts, sparse_matrix)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before encoding")

        # Get TF-IDF sparse matrices
        sparse_matrices = self.vectorizer.transform(texts)

        # Convert each to dictionary
        sparse_dicts = [
            self._matrix_to_dict(sparse_matrices[i:i+1])
            for i in range(len(texts))
        ]

        return sparse_dicts, sparse_matrices

    def _matrix_to_dict(self, sparse_matrix: csr_matrix) -> Dict[str, float]:
        """Convert sparse matrix to dictionary of {term: weight}"""
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Get non-zero indices and values
        cx = sparse_matrix.tocoo()
        sparse_dict = {}

        for i, j, v in zip(cx.row, cx.col, cx.data):
            term = feature_names[j]
            sparse_dict[term] = float(v)

        # Sort by weight (descending)
        sparse_dict = dict(sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True))

        return sparse_dict

    def get_sparsity(self, sparse_dict: Dict[str, float]) -> float:
        """
        Calculate sparsity percentage

        Returns:
            Percentage of zero values (0-100)
        """
        non_zero = len(sparse_dict)
        total = self.max_features
        sparsity = (1 - non_zero / total) * 100
        return sparsity

    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two texts using sparse encoding

        Args:
            text1: First text
            text2: Second text

        Returns:
            Comparison results with overlap analysis
        """
        dict1, matrix1 = self.encode(text1)
        dict2, matrix2 = self.encode(text2)

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(matrix1, matrix2)[0][0]

        # Find overlapping terms
        terms1 = set(dict1.keys())
        terms2 = set(dict2.keys())
        overlap = terms1 & terms2

        return {
            'similarity': float(similarity),
            'text1_terms': len(terms1),
            'text2_terms': len(terms2),
            'overlap_terms': len(overlap),
            'overlap_ratio': len(overlap) / max(len(terms1), len(terms2)),
            'overlapping_terms': {
                term: {
                    'weight1': dict1.get(term, 0),
                    'weight2': dict2.get(term, 0)
                }
                for term in overlap
            }
        }

    def explain_encoding(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Explain the sparse encoding for a text

        Args:
            text: Input text
            top_k: Number of top terms to show

        Returns:
            Detailed explanation of the encoding
        """
        sparse_dict, sparse_matrix = self.encode(text)

        # Get top terms
        top_terms = list(sparse_dict.items())[:top_k]

        # Calculate statistics
        sparsity = self.get_sparsity(sparse_dict)

        return {
            'original_text': text,
            'vocabulary_size': len(self.vocabulary),
            'active_terms': len(sparse_dict),
            'sparsity_percent': sparsity,
            'top_terms': top_terms,
            'full_sparse_dict': sparse_dict
        }


class LearnedSparseEncoder:
    """
    Learned sparse encoder with term expansion
    Simulates models like SPLADE (Sparse Lexical and Expansion Model)

    This adds semantic expansion to sparse encoding:
    - "apple" → includes weights for "fruit", "iphone", "computer"
    - "expensive" → includes "costly", "pricey", "high-price"
    """

    def __init__(self, base_encoder: SparseEncoder, expansion_factor: float = 0.3):
        """
        Initialize learned sparse encoder

        Args:
            base_encoder: Base TF-IDF sparse encoder
            expansion_factor: Weight multiplier for expanded terms (0-1)
        """
        self.base_encoder = base_encoder
        self.expansion_factor = expansion_factor

        # Synonym/expansion dictionary (simplified - in practice, use word embeddings)
        self.expansions = {
            'expensive': ['costly', 'pricey', 'high'],
            'cheap': ['inexpensive', 'affordable', 'budget'],
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'fast': ['quick', 'rapid', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'compact'],
            'happy': ['joyful', 'pleased', 'delighted'],
            'sad': ['unhappy', 'depressed', 'melancholy'],
        }

    def encode_with_expansion(self, text: str) -> Dict[str, float]:
        """
        Encode text with learned term expansion

        Args:
            text: Input text

        Returns:
            Expanded sparse dictionary
        """
        # Get base sparse encoding
        sparse_dict, _ = self.base_encoder.encode(text)

        # Create expanded dictionary
        expanded_dict = sparse_dict.copy()

        # Add expanded terms
        for term, weight in sparse_dict.items():
            if term in self.expansions:
                for expanded_term in self.expansions[term]:
                    # Check if expanded term is in vocabulary
                    if expanded_term in self.base_encoder.vocabulary:
                        # Add with reduced weight
                        expansion_weight = weight * self.expansion_factor
                        if expanded_term in expanded_dict:
                            expanded_dict[expanded_term] = max(
                                expanded_dict[expanded_term],
                                expansion_weight
                            )
                        else:
                            expanded_dict[expanded_term] = expansion_weight

        # Sort by weight
        expanded_dict = dict(sorted(expanded_dict.items(), key=lambda x: x[1], reverse=True))

        return expanded_dict


def demo_sparse_encoding():
    """Demonstrate sparse encoding with examples"""
    print("=" * 70)
    print("Sparse Encoding Demo")
    print("=" * 70)

    # Sample corpus
    corpus = [
        "Apple products are expensive but high quality",
        "An apple a day keeps the doctor away",
        "The new iPhone is very expensive",
        "Healthy fruits include apples and oranges",
        "Technology gadgets can be costly"
    ]

    # Initialize and fit encoder
    encoder = SparseEncoder(max_features=50, ngram_range=(1, 1))
    encoder.fit(corpus)

    # Test queries
    queries = [
        "apple headphones",
        "expensive technology",
        "healthy food"
    ]

    print("\n" + "=" * 70)
    print("Query Analysis")
    print("=" * 70)

    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 70)

        explanation = encoder.explain_encoding(query)

        print(f"Active terms: {explanation['active_terms']} / {explanation['vocabulary_size']}")
        print(f"Sparsity: {explanation['sparsity_percent']:.1f}%")
        print(f"\nTop weighted terms:")
        for term, weight in explanation['top_terms']:
            print(f"  '{term}': {weight:.3f}")

    # Compare texts
    print("\n" + "=" * 70)
    print("Text Comparison")
    print("=" * 70)

    text1 = "Apple products are expensive"
    text2 = "An apple a day"

    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print("-" * 70)

    comparison = encoder.compare_texts(text1, text2)
    print(f"Similarity: {comparison['similarity']:.3f}")
    print(f"Overlapping terms: {comparison['overlap_terms']} ({comparison['overlap_ratio']:.1%})")
    print(f"\nShared terms:")
    for term, weights in list(comparison['overlapping_terms'].items())[:5]:
        print(f"  '{term}': {weights['weight1']:.3f} vs {weights['weight2']:.3f}")

    # Demonstrate learned expansion
    print("\n" + "=" * 70)
    print("Learned Sparse Encoding (with expansion)")
    print("=" * 70)

    learned_encoder = LearnedSparseEncoder(encoder, expansion_factor=0.5)

    test_text = "This product is expensive"
    print(f"\nOriginal text: '{test_text}'")
    print("-" * 70)

    # Base encoding
    base_dict, _ = encoder.encode(test_text)
    print(f"\nBase sparse encoding ({len(base_dict)} terms):")
    for term, weight in list(base_dict.items())[:10]:
        print(f"  '{term}': {weight:.3f}")

    # Expanded encoding
    expanded_dict = learned_encoder.encode_with_expansion(test_text)
    print(f"\nExpanded sparse encoding ({len(expanded_dict)} terms):")
    for term, weight in list(expanded_dict.items())[:10]:
        marker = "🆕" if term not in base_dict else "  "
        print(f"  {marker} '{term}': {weight:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_sparse_encoding()
