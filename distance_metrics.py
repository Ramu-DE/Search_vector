#!/usr/bin/env python3
"""
KNN Distance Metrics Implementation
Comprehensive implementation of Euclidean, Cosine Similarity, and Dot Product
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DistanceMetric(Enum):
    """Available distance/similarity metrics"""
    EUCLIDEAN = "euclidean"  # L2 distance
    COSINE = "cosine"  # Cosine similarity
    DOT_PRODUCT = "dot_product"  # Dot product
    MANHATTAN = "manhattan"  # L1 distance
    CHEBYSHEV = "chebyshev"  # L-infinity distance


@dataclass
class DistanceResult:
    """Result of distance calculation"""
    metric: str
    value: float
    interpretation: str
    normalized: float = 0.0


class DistanceCalculator:
    """
    Calculate distances and similarities between vectors

    Implements three main metrics from AWS slide:
    1. Euclidean (L2) - For counts/measurements, recommendation systems
    2. Cosine Similarity - For semantic search, document classification
    3. Dot Product - For collaborative filtering
    """

    @staticmethod
    def euclidean(p: np.ndarray, q: np.ndarray) -> float:
        """
        Euclidean Distance (L2)

        Formula: d(p,q) = √(Σ(qi - pi)²)

        Properties:
        - Measures straight-line distance
        - Sensitive to magnitude
        - Range: [0, ∞), lower is more similar

        Use cases:
        - Recommendation systems
        - Physical measurements
        - Count-based data

        Args:
            p: First vector
            q: Second vector

        Returns:
            Euclidean distance (0 = identical, larger = more different)
        """
        if len(p) != len(q):
            raise ValueError(f"Vectors must have same length: {len(p)} vs {len(q)}")

        # Calculate squared differences
        squared_diff = (q - p) ** 2

        # Sum and take square root
        distance = np.sqrt(np.sum(squared_diff))

        return float(distance)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine Similarity

        Formula: sim(a,b) = (a·b) / (||a|| · ||b||)

        Properties:
        - Measures angle between vectors
        - Invariant to magnitude (scale)
        - Range: [-1, 1], 1 = identical direction

        Use cases:
        - Semantic search
        - Document classification
        - Text similarity
        - When scale doesn't matter

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (-1 = opposite, 0 = orthogonal, 1 = same)
        """
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")

        # Calculate dot product
        dot_product = np.dot(a, b)

        # Calculate magnitudes
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (norm_a * norm_b)

        # Clip to [-1, 1] to handle numerical errors
        return float(np.clip(similarity, -1.0, 1.0))

    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """
        Dot Product

        Formula: a·b = ||a|| ||b|| cos(α) = Σ(ai × bi)

        Properties:
        - Combines magnitude and direction
        - Sensitive to scale
        - Range: (-∞, ∞), higher is more similar

        Use cases:
        - Collaborative filtering
        - Recommendation systems
        - When both magnitude and direction matter

        Args:
            a: First vector
            b: Second vector

        Returns:
            Dot product (higher = more similar in direction and magnitude)
        """
        if len(a) != len(b):
            raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")

        # Calculate dot product
        product = np.dot(a, b)

        return float(product)

    @staticmethod
    def manhattan(p: np.ndarray, q: np.ndarray) -> float:
        """
        Manhattan Distance (L1)

        Formula: d(p,q) = Σ|qi - pi|

        Properties:
        - Sum of absolute differences
        - Less sensitive to outliers than Euclidean
        - Range: [0, ∞)

        Args:
            p: First vector
            q: Second vector

        Returns:
            Manhattan distance
        """
        if len(p) != len(q):
            raise ValueError(f"Vectors must have same length: {len(p)} vs {len(q)}")

        return float(np.sum(np.abs(q - p)))

    @staticmethod
    def chebyshev(p: np.ndarray, q: np.ndarray) -> float:
        """
        Chebyshev Distance (L∞)

        Formula: d(p,q) = max|qi - pi|

        Properties:
        - Maximum absolute difference
        - Useful for grid-based problems
        - Range: [0, ∞)

        Args:
            p: First vector
            q: Second vector

        Returns:
            Chebyshev distance
        """
        if len(p) != len(q):
            raise ValueError(f"Vectors must have same length: {len(p)} vs {len(q)}")

        return float(np.max(np.abs(q - p)))

    @classmethod
    def calculate_all(cls, p: np.ndarray, q: np.ndarray) -> Dict[str, DistanceResult]:
        """
        Calculate all distance metrics

        Args:
            p: First vector
            q: Second vector

        Returns:
            Dictionary of all distance/similarity results
        """
        results = {}

        # Euclidean
        eucl = cls.euclidean(p, q)
        results['euclidean'] = DistanceResult(
            metric='Euclidean (L2)',
            value=eucl,
            interpretation='Lower is more similar',
            normalized=1 / (1 + eucl)  # Normalize to [0, 1]
        )

        # Cosine
        cos = cls.cosine_similarity(p, q)
        results['cosine'] = DistanceResult(
            metric='Cosine Similarity',
            value=cos,
            interpretation='1 = same direction, -1 = opposite',
            normalized=(cos + 1) / 2  # Normalize to [0, 1]
        )

        # Dot Product
        dot = cls.dot_product(p, q)
        results['dot_product'] = DistanceResult(
            metric='Dot Product',
            value=dot,
            interpretation='Higher is more similar',
            normalized=dot  # Already meaningful
        )

        # Manhattan
        manh = cls.manhattan(p, q)
        results['manhattan'] = DistanceResult(
            metric='Manhattan (L1)',
            value=manh,
            interpretation='Lower is more similar',
            normalized=1 / (1 + manh)
        )

        # Chebyshev
        cheb = cls.chebyshev(p, q)
        results['chebyshev'] = DistanceResult(
            metric='Chebyshev (L∞)',
            value=cheb,
            interpretation='Lower is more similar',
            normalized=1 / (1 + cheb)
        )

        return results


class KNNSearchEngine:
    """
    K-Nearest Neighbors search with different distance metrics
    """

    def __init__(self, metric: DistanceMetric = DistanceMetric.EUCLIDEAN):
        """Initialize KNN search with specified metric"""
        self.metric = metric
        self.calculator = DistanceCalculator()
        self.vectors = None
        self.labels = None

    def fit(self, vectors: np.ndarray, labels: List[Any]):
        """
        Fit the search engine with vectors and labels

        Args:
            vectors: Array of shape (n_samples, n_features)
            labels: List of labels for each vector
        """
        self.vectors = np.array(vectors)
        self.labels = labels

        if len(self.vectors) != len(self.labels):
            raise ValueError("Number of vectors must match number of labels")

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Find k nearest neighbors

        Args:
            query: Query vector
            k: Number of neighbors to return

        Returns:
            List of (label, distance/similarity) tuples
        """
        if self.vectors is None:
            raise ValueError("Must call fit() before search()")

        # Calculate distances/similarities
        scores = []

        for i, vec in enumerate(self.vectors):
            if self.metric == DistanceMetric.EUCLIDEAN:
                score = self.calculator.euclidean(query, vec)
                # Lower is better
                scores.append((self.labels[i], score))

            elif self.metric == DistanceMetric.COSINE:
                score = self.calculator.cosine_similarity(query, vec)
                # Higher is better, so negate for sorting
                scores.append((self.labels[i], -score))

            elif self.metric == DistanceMetric.DOT_PRODUCT:
                score = self.calculator.dot_product(query, vec)
                # Higher is better, so negate for sorting
                scores.append((self.labels[i], -score))

            elif self.metric == DistanceMetric.MANHATTAN:
                score = self.calculator.manhattan(query, vec)
                # Lower is better
                scores.append((self.labels[i], score))

            elif self.metric == DistanceMetric.CHEBYSHEV:
                score = self.calculator.chebyshev(query, vec)
                # Lower is better
                scores.append((self.labels[i], score))

        # Sort by score
        scores.sort(key=lambda x: x[1])

        # Return top k
        results = []
        for label, score in scores[:k]:
            # Convert back to original scale if needed
            if self.metric in [DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT]:
                score = -score
            results.append((label, score))

        return results


def demo_distance_metrics():
    """Demonstrate distance metrics with examples"""
    print("=" * 70)
    print("KNN Distance Metrics Demo")
    print("=" * 70)

    # Example 1: Euclidean Distance (Physical measurements)
    print("\n1. EUCLIDEAN DISTANCE (L2)")
    print("-" * 70)
    print("Use case: Recommendation systems, physical measurements")
    print()

    # Houses with [bedrooms, bathrooms, sqm/100, price/100k]
    house1 = np.array([3, 2, 8, 5])  # 3 bed, 2 bath, 800 sqm, $500k
    house2 = np.array([3, 2, 9, 5.5])  # Similar house
    house3 = np.array([5, 4, 15, 12])  # Luxury house

    calc = DistanceCalculator()

    dist_12 = calc.euclidean(house1, house2)
    dist_13 = calc.euclidean(house1, house3)

    print(f"House 1: {house1} (3 bed, 2 bath, 800sqm, $500k)")
    print(f"House 2: {house2} (3 bed, 2 bath, 900sqm, $550k)")
    print(f"House 3: {house3} (5 bed, 4 bath, 1500sqm, $1.2M)")
    print()
    print(f"Distance(House1, House2): {dist_12:.2f} ← Similar houses")
    print(f"Distance(House1, House3): {dist_13:.2f} ← Very different")
    print()
    print("💡 Smaller distance = more similar")

    # Example 2: Cosine Similarity (Document similarity)
    print("\n2. COSINE SIMILARITY")
    print("-" * 70)
    print("Use case: Semantic search, document classification")
    print()

    # Documents represented by term frequencies
    # [sports, politics, tech, entertainment]
    doc1 = np.array([10, 2, 1, 0])  # Sports article
    doc2 = np.array([8, 1, 2, 1])   # Also sports
    doc3 = np.array([1, 10, 2, 1])  # Politics article

    sim_12 = calc.cosine_similarity(doc1, doc2)
    sim_13 = calc.cosine_similarity(doc1, doc3)

    print(f"Doc 1: {doc1} (Sports: 10, Politics: 2, Tech: 1, Entertainment: 0)")
    print(f"Doc 2: {doc2} (Sports: 8, Politics: 1, Tech: 2, Entertainment: 1)")
    print(f"Doc 3: {doc3} (Sports: 1, Politics: 10, Tech: 2, Entertainment: 1)")
    print()
    print(f"Similarity(Doc1, Doc2): {sim_12:.4f} ← Both about sports")
    print(f"Similarity(Doc1, Doc3): {sim_13:.4f} ← Different topics")
    print()
    print("💡 Value close to 1 = similar direction (same topic)")
    print("💡 Ignores magnitude (article length doesn't matter)")

    # Example 3: Dot Product (Collaborative filtering)
    print("\n3. DOT PRODUCT")
    print("-" * 70)
    print("Use case: Collaborative filtering, recommendations")
    print()

    # User ratings [movie1, movie2, movie3, movie4]
    user1 = np.array([5, 4, 1, 0])  # Likes movies 1 & 2
    user2 = np.array([5, 5, 0, 1])  # Also likes movies 1 & 2
    user3 = np.array([1, 0, 5, 5])  # Likes different movies

    dot_12 = calc.dot_product(user1, user2)
    dot_13 = calc.dot_product(user1, user3)

    print(f"User 1 ratings: {user1} (Movie1: 5★, Movie2: 4★, Movie3: 1★, Movie4: 0★)")
    print(f"User 2 ratings: {user2} (Movie1: 5★, Movie2: 5★, Movie3: 0★, Movie4: 1★)")
    print(f"User 3 ratings: {user3} (Movie1: 1★, Movie2: 0★, Movie3: 5★, Movie4: 5★)")
    print()
    print(f"Dot Product(User1, User2): {dot_12:.0f} ← Similar tastes")
    print(f"Dot Product(User1, User3): {dot_13:.0f} ← Different tastes")
    print()
    print("💡 Higher value = more similar preferences")
    print("💡 Considers both rating values and agreement")

    # Comparison of all metrics
    print("\n" + "=" * 70)
    print("COMPARISON: All Metrics on Same Data")
    print("-" * 70)

    vec1 = np.array([1, 2, 3])
    vec2 = np.array([2, 4, 6])  # 2x vec1 (same direction)
    vec3 = np.array([3, 2, 1])  # Different

    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2} (2× Vector 1, same direction)")
    print(f"Vector 3: {vec3} (different direction)")
    print()

    results_12 = calc.calculate_all(vec1, vec2)
    results_13 = calc.calculate_all(vec1, vec3)

    print("Comparing Vec1 vs Vec2 (same direction):")
    for name, result in results_12.items():
        print(f"  {result.metric:20s}: {result.value:8.4f} - {result.interpretation}")

    print("\nComparing Vec1 vs Vec3 (different):")
    for name, result in results_13.items():
        print(f"  {result.metric:20s}: {result.value:8.4f} - {result.interpretation}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print("• Euclidean: Sensitive to magnitude (Vec2 is 2x, so larger distance)")
    print("• Cosine: Perfect match (1.0) - same direction, ignores scale")
    print("• Dot Product: Higher for Vec2 - rewards both direction AND magnitude")
    print("=" * 70)


def demo_knn_search():
    """Demonstrate KNN search with different metrics"""
    print("\n" + "=" * 70)
    print("KNN Search Demo - Finding Similar Items")
    print("=" * 70)

    # Movie dataset: [action, comedy, drama, romance, sci-fi]
    movies = {
        'Die Hard': np.array([9, 2, 4, 0, 2]),
        'The Matrix': np.array([10, 1, 5, 0, 9]),
        'Terminator': np.array([9, 1, 3, 0, 8]),
        'The Hangover': np.array([2, 10, 3, 5, 0]),
        'Superbad': np.array([1, 9, 2, 4, 0]),
        'Titanic': np.array([3, 2, 9, 10, 1]),
        'The Notebook': np.array([1, 2, 8, 10, 0]),
    }

    vectors = np.array(list(movies.values()))
    labels = list(movies.keys())

    query_movie = 'The Matrix'
    query_vector = movies[query_movie]

    print(f"Query movie: '{query_movie}'")
    print(f"Features: {query_vector} (Action: 10, Comedy: 1, Drama: 5, Romance: 0, Sci-Fi: 9)")
    print()

    # Try different metrics
    metrics = [
        (DistanceMetric.EUCLIDEAN, "Euclidean (L2)"),
        (DistanceMetric.COSINE, "Cosine Similarity"),
        (DistanceMetric.DOT_PRODUCT, "Dot Product")
    ]

    for metric, name in metrics:
        print(f"\n{name}:")
        print("-" * 70)

        engine = KNNSearchEngine(metric=metric)
        engine.fit(vectors, labels)

        results = engine.search(query_vector, k=4)

        for i, (movie, score) in enumerate(results[1:], 1):  # Skip self-match
            print(f"  {i}. {movie:20s} (score: {score:.4f})")

    print("\n" + "=" * 70)
    print("OBSERVATIONS:")
    print("-" * 70)
    print("• Euclidean: Finds movies with similar feature values")
    print("• Cosine: Finds movies with similar proportions (direction)")
    print("• Dot Product: Finds movies with similar AND strong features")
    print("=" * 70)


if __name__ == "__main__":
    demo_distance_metrics()
    demo_knn_search()
