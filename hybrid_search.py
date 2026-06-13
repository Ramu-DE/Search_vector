#!/usr/bin/env python3
"""
Hybrid Search Implementation
Combines keyword (BM25), sparse (TF-IDF), and dense (embeddings) search
with multiple score normalization and combination strategies
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class CombinationMethod(Enum):
    """Score combination methods"""
    ARITHMETIC_MEAN = "arithmetic_mean"  # Equal weight
    WEIGHTED_SUM = "weighted_sum"        # Tunable weights
    HARMONIC_MEAN = "harmonic_mean"      # Penalizes discrepancies
    GEOMETRIC_MEAN = "geometric_mean"    # Multiplicative
    MAX = "max"                          # Take maximum
    MIN = "min"                          # Take minimum


@dataclass
class SearchResult:
    """Single search result with scores"""
    doc_id: int
    title: str
    text: str
    keyword_score: float = 0.0
    sparse_score: float = 0.0
    dense_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""
    keyword_weight: float = 0.3
    sparse_weight: float = 0.3
    dense_weight: float = 0.4
    combination_method: CombinationMethod = CombinationMethod.WEIGHTED_SUM
    normalize_scores: bool = True
    k: int = 10  # Top-K results


class ScoreCombiner:
    """Combine scores from different search methods"""

    @staticmethod
    def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize scores to [0, 1] range

        Args:
            scores: Array of scores
            method: "minmax" or "zscore"

        Returns:
            Normalized scores
        """
        if len(scores) == 0 or np.all(scores == 0):
            return scores

        if method == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score == min_score:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)

        elif method == "zscore":
            # Z-score normalization: (x - mean) / std
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                return np.zeros_like(scores)
            normalized = (scores - mean) / std
            # Clip to [0, 1]
            return np.clip(normalized, 0, 1)

        return scores

    @staticmethod
    def arithmetic_mean(scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Arithmetic mean: (a + b + c) / 3
        Equal weight to all methods
        """
        scores_list = [s for s in scores_dict.values() if len(s) > 0]
        if not scores_list:
            return np.array([])
        return np.mean(scores_list, axis=0)

    @staticmethod
    def weighted_sum(
        scores_dict: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Weighted sum: w1*a + w2*b + w3*c
        Tunable weights for each method
        """
        if not scores_dict:
            return np.array([])

        combined = np.zeros_like(list(scores_dict.values())[0])
        for name, scores in scores_dict.items():
            weight = weights.get(name, 0.0)
            combined += weight * scores

        return combined

    @staticmethod
    def harmonic_mean(scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Harmonic mean: n / (1/a + 1/b + 1/c)
        Penalizes large discrepancies (all scores must be good)
        """
        scores_list = [s for s in scores_dict.values() if len(s) > 0]
        if not scores_list:
            return np.array([])

        # Avoid division by zero
        n = len(scores_list)
        reciprocals = np.zeros_like(scores_list[0])

        for scores in scores_list:
            # Add small epsilon to avoid division by zero
            reciprocals += 1.0 / (scores + 1e-10)

        return n / (reciprocals + 1e-10)

    @staticmethod
    def geometric_mean(scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Geometric mean: (a * b * c)^(1/3)
        Multiplicative combination
        """
        scores_list = [s for s in scores_dict.values() if len(s) > 0]
        if not scores_list:
            return np.array([])

        n = len(scores_list)
        product = np.ones_like(scores_list[0])

        for scores in scores_list:
            # Add small value to avoid zero
            product *= (scores + 1e-10)

        return np.power(product, 1.0 / n)

    @staticmethod
    def max_score(scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Take maximum score from any method"""
        scores_list = [s for s in scores_dict.values() if len(s) > 0]
        if not scores_list:
            return np.array([])
        return np.max(scores_list, axis=0)

    @staticmethod
    def min_score(scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Take minimum score from any method (conservative)"""
        scores_list = [s for s in scores_dict.values() if len(s) > 0]
        if not scores_list:
            return np.array([])
        return np.min(scores_list, axis=0)


class HybridSearchEngine:
    """
    Hybrid search engine combining multiple search methods

    Supports:
    - Keyword search (BM25-style TF-IDF)
    - Sparse semantic search (learned sparse)
    - Dense semantic search (embeddings)
    - Multiple score combination strategies
    """

    def __init__(self, config: Optional[HybridSearchConfig] = None):
        """Initialize hybrid search engine"""
        self.config = config or HybridSearchConfig()
        self.combiner = ScoreCombiner()

        # Search components
        self.keyword_vectorizer = None
        self.sparse_vectorizer = None
        self.dense_encoder = None

        # Indexed data
        self.documents = []
        self.keyword_vectors = None
        self.sparse_vectors = None
        self.dense_vectors = None

        self.is_fitted = False

    def fit(
        self,
        documents: List[Dict[str, Any]],
        max_features: int = 5000
    ):
        """
        Fit the hybrid search engine on a corpus

        Args:
            documents: List of documents with 'text' and 'title' fields
            max_features: Maximum vocabulary size
        """
        print(f"Fitting hybrid search on {len(documents)} documents...")

        self.documents = documents
        texts = [doc['text'] for doc in documents]

        # 1. Keyword search (BM25-style TF-IDF)
        print("  Building keyword index...")
        self.keyword_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,  # Use log(tf) for BM25-like behavior
            norm='l2'
        )
        self.keyword_vectors = self.keyword_vectorizer.fit_transform(texts)

        # 2. Sparse semantic search (TF-IDF with potential expansion)
        print("  Building sparse semantic index...")
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            stop_words='english',
            norm='l2'
        )
        self.sparse_vectors = self.sparse_vectorizer.fit_transform(texts)

        # 3. Dense semantic search (simulated - in practice use real embeddings)
        print("  Building dense semantic index...")
        # For demo, use TF-IDF as proxy (in production, use actual embeddings)
        self.dense_vectorizer = TfidfVectorizer(
            max_features=min(384, max_features),  # Simulate embedding dimension
            ngram_range=(1, 1),
            stop_words='english',
            norm='l2'
        )
        self.dense_vectors = self.dense_vectorizer.fit_transform(texts).toarray()

        self.is_fitted = True
        print(f"✓ Hybrid search fitted")
        print(f"  Keyword vocab: {len(self.keyword_vectorizer.vocabulary_)}")
        print(f"  Sparse vocab: {len(self.sparse_vectorizer.vocabulary_)}")
        print(f"  Dense dims: {self.dense_vectors.shape[1]}")

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        method: Optional[CombinationMethod] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search

        Args:
            query: Search query
            k: Number of results (default: from config)
            method: Combination method (default: from config)
            weights: Custom weights for weighted_sum

        Returns:
            List of SearchResult objects
        """
        if not self.is_fitted:
            raise ValueError("Must fit() the search engine first")

        k = k or self.config.k
        method = method or self.config.combination_method

        # Get scores from each search method
        keyword_scores = self._keyword_search(query)
        sparse_scores = self._sparse_search(query)
        dense_scores = self._dense_search(query)

        # Normalize scores if configured
        if self.config.normalize_scores:
            keyword_scores = self.combiner.normalize_scores(keyword_scores)
            sparse_scores = self.combiner.normalize_scores(sparse_scores)
            dense_scores = self.combiner.normalize_scores(dense_scores)

        # Combine scores
        scores_dict = {
            'keyword': keyword_scores,
            'sparse': sparse_scores,
            'dense': dense_scores
        }

        if method == CombinationMethod.ARITHMETIC_MEAN:
            combined_scores = self.combiner.arithmetic_mean(scores_dict)

        elif method == CombinationMethod.WEIGHTED_SUM:
            if weights is None:
                weights = {
                    'keyword': self.config.keyword_weight,
                    'sparse': self.config.sparse_weight,
                    'dense': self.config.dense_weight
                }
            combined_scores = self.combiner.weighted_sum(scores_dict, weights)

        elif method == CombinationMethod.HARMONIC_MEAN:
            combined_scores = self.combiner.harmonic_mean(scores_dict)

        elif method == CombinationMethod.GEOMETRIC_MEAN:
            combined_scores = self.combiner.geometric_mean(scores_dict)

        elif method == CombinationMethod.MAX:
            combined_scores = self.combiner.max_score(scores_dict)

        elif method == CombinationMethod.MIN:
            combined_scores = self.combiner.min_score(scores_dict)

        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Create results
        results = []
        for idx in range(len(self.documents)):
            result = SearchResult(
                doc_id=idx,
                title=self.documents[idx].get('title', ''),
                text=self.documents[idx].get('text', ''),
                keyword_score=float(keyword_scores[idx]),
                sparse_score=float(sparse_scores[idx]),
                dense_score=float(dense_scores[idx]),
                combined_score=float(combined_scores[idx])
            )
            results.append(result)

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        # Add ranks
        for rank, result in enumerate(results[:k], 1):
            result.rank = rank

        return results[:k]

    def _keyword_search(self, query: str) -> np.ndarray:
        """Keyword search using BM25-style TF-IDF"""
        query_vec = self.keyword_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.keyword_vectors)[0]
        return scores

    def _sparse_search(self, query: str) -> np.ndarray:
        """Sparse semantic search"""
        query_vec = self.sparse_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.sparse_vectors)[0]
        return scores

    def _dense_search(self, query: str) -> np.ndarray:
        """Dense semantic search (simulated)"""
        # In production, encode query with actual embedding model
        # Use the same vectorizer that created dense_vectors to ensure vocabulary alignment
        query_vec = self.dense_vectorizer.transform([query]).toarray()

        scores = cosine_similarity(query_vec, self.dense_vectors)[0]
        return scores

    def compare_methods(
        self,
        query: str,
        k: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """
        Compare all combination methods

        Args:
            query: Search query
            k: Number of results per method

        Returns:
            Dict mapping method name to results
        """
        results = {}

        for method in CombinationMethod:
            method_results = self.search(query, k=k, method=method)
            results[method.value] = method_results

        return results

    def explain_result(
        self,
        query: str,
        doc_id: int
    ) -> Dict[str, Any]:
        """
        Explain why a document matched

        Args:
            query: Search query
            doc_id: Document ID

        Returns:
            Explanation with score breakdown
        """
        keyword_scores = self._keyword_search(query)
        sparse_scores = self._sparse_search(query)
        dense_scores = self._dense_search(query)

        # Normalize
        keyword_norm = self.combiner.normalize_scores(keyword_scores)
        sparse_norm = self.combiner.normalize_scores(sparse_scores)
        dense_norm = self.combiner.normalize_scores(dense_scores)

        return {
            'document': self.documents[doc_id],
            'scores': {
                'keyword_raw': float(keyword_scores[doc_id]),
                'keyword_normalized': float(keyword_norm[doc_id]),
                'sparse_raw': float(sparse_scores[doc_id]),
                'sparse_normalized': float(sparse_norm[doc_id]),
                'dense_raw': float(dense_scores[doc_id]),
                'dense_normalized': float(dense_norm[doc_id])
            },
            'weights': {
                'keyword': self.config.keyword_weight,
                'sparse': self.config.sparse_weight,
                'dense': self.config.dense_weight
            }
        }


def demo_hybrid_search():
    """Demonstrate hybrid search capabilities"""
    print("=" * 70)
    print("Hybrid Search Demo")
    print("=" * 70)

    # Sample documents
    documents = [
        {
            'id': 1,
            'title': 'iPhone 15 Pro Review',
            'text': 'The new iPhone 15 Pro features an expensive titanium design with excellent camera quality and fast performance'
        },
        {
            'id': 2,
            'title': 'Healthy Apple Recipes',
            'text': 'Eating an apple a day keeps the doctor away. Fresh apples are healthy and nutritious fruits'
        },
        {
            'id': 3,
            'title': 'MacBook Pro 2024',
            'text': 'Apple MacBook Pro is a powerful laptop computer for professionals with high performance and premium build quality'
        },
        {
            'id': 4,
            'title': 'Budget Smartphone Guide',
            'text': 'Affordable phones that are cheap but good quality. Best budget-friendly mobile devices under 500 dollars'
        },
        {
            'id': 5,
            'title': 'Wireless Headphones Comparison',
            'text': 'Top wireless headphones and earbuds for music. High-quality audio devices with noise cancellation technology'
        }
    ]

    # Initialize and fit
    config = HybridSearchConfig(
        keyword_weight=0.3,
        sparse_weight=0.3,
        dense_weight=0.4,
        combination_method=CombinationMethod.WEIGHTED_SUM
    )

    engine = HybridSearchEngine(config)
    engine.fit(documents, max_features=100)

    # Test query
    query = "expensive apple products"

    print(f"\n📝 Query: '{query}'")
    print("=" * 70)

    # Compare all methods
    print("\nComparing combination methods:")
    print("-" * 70)

    all_results = engine.compare_methods(query, k=3)

    for method_name, results in all_results.items():
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        for result in results:
            print(f"  {result.rank}. [{result.combined_score:.3f}] {result.title}")
            print(f"     K:{result.keyword_score:.2f} S:{result.sparse_score:.2f} D:{result.dense_score:.2f}")

    # Explain top result
    print("\n" + "=" * 70)
    print("Explaining Top Result")
    print("-" * 70)

    results = engine.search(query, k=1)
    if results:
        explanation = engine.explain_result(query, results[0].doc_id)
        print(f"\nDocument: {explanation['document']['title']}")
        print(f"\nScore Breakdown:")
        for score_type, value in explanation['scores'].items():
            print(f"  {score_type}: {value:.4f}")
        print(f"\nWeights Used:")
        for weight_type, value in explanation['weights'].items():
            print(f"  {weight_type}: {value}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_hybrid_search()
