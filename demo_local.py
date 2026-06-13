#!/usr/bin/env python3
"""
Local Demo - AI-Powered Search (No AWS Required)
Demonstrates vector search concepts using local embeddings and in-memory storage
"""

import numpy as np
from typing import List, Dict, Any
import json

print("=" * 70)
print("  AI-POWERED SEARCH - LOCAL DEMO")
print("  Demonstrating Vector Search Concepts")
print("=" * 70)

# ============================================================================
# STEP 1: Sample Movie Database
# ============================================================================

SAMPLE_MOVIES = [
    {
        "id": 1,
        "title": "The Shawshank Redemption",
        "year": 1994,
        "rating": 9.3,
        "genre": ["Drama"],
        "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "keywords": ["prison", "friendship", "hope", "redemption", "freedom"]
    },
    {
        "id": 2,
        "title": "The Godfather",
        "year": 1972,
        "rating": 9.2,
        "genre": ["Crime", "Drama"],
        "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "keywords": ["mafia", "family", "crime", "power", "loyalty"]
    },
    {
        "id": 3,
        "title": "The Dark Knight",
        "year": 2008,
        "rating": 9.0,
        "genre": ["Action", "Crime", "Drama"],
        "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological tests.",
        "keywords": ["superhero", "action", "villain", "batman", "chaos"]
    },
    {
        "id": 4,
        "title": "Forrest Gump",
        "year": 1994,
        "rating": 8.8,
        "genre": ["Drama", "Romance"],
        "plot": "The presidencies of Kennedy and Johnson, the Vietnam War, and other historical events unfold from the perspective of an Alabama man.",
        "keywords": ["history", "life", "journey", "love", "inspiration"]
    },
    {
        "id": 5,
        "title": "Inception",
        "year": 2010,
        "rating": 8.8,
        "genre": ["Action", "Sci-Fi", "Thriller"],
        "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.",
        "keywords": ["dreams", "mind", "heist", "reality", "thriller"]
    },
    {
        "id": 6,
        "title": "The Matrix",
        "year": 1999,
        "rating": 8.7,
        "genre": ["Action", "Sci-Fi"],
        "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "keywords": ["virtual reality", "hacker", "ai", "philosophy", "action"]
    },
    {
        "id": 7,
        "title": "Pulp Fiction",
        "year": 1994,
        "rating": 8.9,
        "genre": ["Crime", "Drama"],
        "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
        "keywords": ["crime", "nonlinear", "violence", "dialogue", "cool"]
    },
    {
        "id": 8,
        "title": "The Lord of the Rings: The Return of the King",
        "year": 2003,
        "rating": 9.0,
        "genre": ["Action", "Adventure", "Drama"],
        "plot": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom.",
        "keywords": ["fantasy", "epic", "war", "quest", "heroism"]
    },
    {
        "id": 9,
        "title": "Fight Club",
        "year": 1999,
        "rating": 8.8,
        "genre": ["Drama"],
        "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
        "keywords": ["rebellion", "identity", "violence", "philosophy", "twist"]
    },
    {
        "id": 10,
        "title": "Goodfellas",
        "year": 1990,
        "rating": 8.7,
        "genre": ["Crime", "Drama"],
        "plot": "The story of Henry Hill and his life in the mob, covering his relationship with his wife and his mob partners.",
        "keywords": ["mafia", "crime", "biography", "violence", "betrayal"]
    }
]

print(f"\n✓ Loaded {len(SAMPLE_MOVIES)} movies into database")

# ============================================================================
# STEP 2: Simple TF-IDF Vectorization (Simulating Embeddings)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: Creating Vector Representations (Simulated Embeddings)")
print("=" * 70)

class SimpleTFIDFVectorizer:
    """Simple TF-IDF vectorizer to simulate embeddings"""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.documents = []

    def fit(self, documents):
        """Build vocabulary and compute IDF"""
        # Build vocabulary
        for doc in documents:
            words = doc.lower().split()
            for word in words:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)

        # Compute IDF
        n_docs = len(documents)
        for word, idx in self.vocabulary.items():
            doc_count = sum(1 for doc in documents if word in doc.lower())
            self.idf[word] = np.log(n_docs / (1 + doc_count))

        self.documents = documents
        print(f"  → Vocabulary size: {len(self.vocabulary)} words")

    def transform(self, text):
        """Convert text to TF-IDF vector"""
        vector = np.zeros(len(self.vocabulary))
        words = text.lower().split()
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / len(words)
                vector[idx] = tf * self.idf.get(word, 0)

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

# Create vectorizer
vectorizer = SimpleTFIDFVectorizer()

# Prepare documents (combine title + plot)
documents = [f"{m['title']} {m['plot']}" for m in SAMPLE_MOVIES]
vectorizer.fit(documents)

# Generate vectors for all movies
movie_vectors = {}
for movie in SAMPLE_MOVIES:
    text = f"{movie['title']} {movie['plot']}"
    vector = vectorizer.transform(text)
    movie_vectors[movie['id']] = vector
    print(f"  → {movie['title']}: Vector shape {vector.shape}, non-zero: {np.count_nonzero(vector)}")

print(f"\n✓ Generated vectors for {len(movie_vectors)} movies")

# ============================================================================
# STEP 3: Similarity Search Functions
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Implementing Search Functions")
print("=" * 70)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(vec1 - vec2)

def semantic_search(query, k=5, min_rating=0.0, metric='cosine'):
    """
    Perform semantic search

    Args:
        query: Search query text
        k: Number of results to return
        min_rating: Minimum movie rating filter
        metric: 'cosine' or 'euclidean'
    """
    # Generate query vector
    query_vector = vectorizer.transform(query)

    # Calculate similarities
    results = []
    for movie in SAMPLE_MOVIES:
        # Apply filters
        if movie['rating'] < min_rating:
            continue

        movie_vector = movie_vectors[movie['id']]

        if metric == 'cosine':
            similarity = cosine_similarity(query_vector, movie_vector)
            score = similarity
        else:  # euclidean
            distance = euclidean_distance(query_vector, movie_vector)
            score = 1 / (1 + distance)  # Convert to similarity

        results.append({
            'movie': movie,
            'score': score
        })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:k]

def keyword_search(query, k=5, min_rating=0.0):
    """
    Perform keyword search (exact word matching)
    """
    query_words = set(query.lower().split())

    results = []
    for movie in SAMPLE_MOVIES:
        # Apply filters
        if movie['rating'] < min_rating:
            continue

        # Count matching words
        movie_text = f"{movie['title']} {movie['plot']}".lower()
        movie_words = set(movie_text.split())

        matches = len(query_words.intersection(movie_words))
        score = matches / len(query_words) if query_words else 0

        if score > 0:
            results.append({
                'movie': movie,
                'score': score
            })

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:k]

def hybrid_search(query, k=5, min_rating=0.0, semantic_weight=0.6):
    """
    Hybrid search combining keyword and semantic
    """
    keyword_results = keyword_search(query, k=k*2, min_rating=min_rating)
    semantic_results = semantic_search(query, k=k*2, min_rating=min_rating)

    # Combine scores
    combined_scores = {}

    for result in keyword_results:
        movie_id = result['movie']['id']
        combined_scores[movie_id] = {
            'movie': result['movie'],
            'keyword_score': result['score'],
            'semantic_score': 0.0
        }

    for result in semantic_results:
        movie_id = result['movie']['id']
        if movie_id not in combined_scores:
            combined_scores[movie_id] = {
                'movie': result['movie'],
                'keyword_score': 0.0,
                'semantic_score': result['score']
            }
        else:
            combined_scores[movie_id]['semantic_score'] = result['score']

    # Calculate weighted score
    results = []
    keyword_weight = 1 - semantic_weight

    for movie_id, scores in combined_scores.items():
        final_score = (keyword_weight * scores['keyword_score'] +
                      semantic_weight * scores['semantic_score'])
        results.append({
            'movie': scores['movie'],
            'score': final_score,
            'keyword_score': scores['keyword_score'],
            'semantic_score': scores['semantic_score']
        })

    # Sort by final score
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:k]

print("✓ Search functions implemented:")
print("  → Semantic search (cosine similarity)")
print("  → Keyword search (exact matching)")
print("  → Hybrid search (combined)")

# ============================================================================
# STEP 4: Run Demo Searches
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Running Demo Searches")
print("=" * 70)

def print_results(results, method_name):
    """Pretty print search results"""
    print(f"\n{method_name}:")
    print("-" * 70)

    if not results:
        print("  No results found")
        return

    for i, result in enumerate(results, 1):
        movie = result['movie']
        score = result['score']

        print(f"\n  {i}. {movie['title']} ({movie['year']}) - Rating: {movie['rating']}")
        print(f"     Score: {score:.4f}")
        print(f"     Genre: {', '.join(movie['genre'])}")
        print(f"     Plot: {movie['plot'][:100]}...")

# Test Query 1: Semantic understanding
print("\n" + "=" * 70)
print("QUERY 1: 'movie about friendship and hope'")
print("=" * 70)

query1 = "movie about friendship and hope"

print_results(
    keyword_search(query1, k=3),
    "Keyword Search"
)

print_results(
    semantic_search(query1, k=3),
    "Semantic Search"
)

print_results(
    hybrid_search(query1, k=3, semantic_weight=0.7),
    "Hybrid Search (70% semantic)"
)

# Test Query 2: Action movies
print("\n" + "=" * 70)
print("QUERY 2: 'action thriller with great fighting'")
print("=" * 70)

query2 = "action thriller with great fighting"

print_results(
    keyword_search(query2, k=3),
    "Keyword Search"
)

print_results(
    semantic_search(query2, k=3),
    "Semantic Search"
)

print_results(
    hybrid_search(query2, k=3, semantic_weight=0.6),
    "Hybrid Search (60% semantic)"
)

# Test Query 3: With filters
print("\n" + "=" * 70)
print("QUERY 3: 'crime movie' (rating >= 8.8)")
print("=" * 70)

query3 = "crime movie"

print_results(
    semantic_search(query3, k=5, min_rating=8.8),
    "Semantic Search with Filter"
)

# ============================================================================
# STEP 5: Visualize Vector Space
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Vector Space Analysis")
print("=" * 70)

# Calculate pairwise similarities
print("\nPairwise Similarities (top 3 most similar movies):")
print("-" * 70)

for movie in SAMPLE_MOVIES[:5]:  # First 5 movies
    vec1 = movie_vectors[movie['id']]
    similarities = []

    for other_movie in SAMPLE_MOVIES:
        if other_movie['id'] != movie['id']:
            vec2 = movie_vectors[other_movie['id']]
            sim = cosine_similarity(vec1, vec2)
            similarities.append((other_movie['title'], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{movie['title']}:")
    for title, sim in similarities[:3]:
        print(f"  → {title}: {sim:.4f}")

# ============================================================================
# STEP 6: Performance Metrics
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Performance Metrics")
print("=" * 70)

import time

def benchmark_search(query, num_runs=100):
    """Benchmark search performance"""

    # Keyword search
    start = time.time()
    for _ in range(num_runs):
        keyword_search(query, k=5)
    keyword_time = (time.time() - start) / num_runs * 1000

    # Semantic search
    start = time.time()
    for _ in range(num_runs):
        semantic_search(query, k=5)
    semantic_time = (time.time() - start) / num_runs * 1000

    # Hybrid search
    start = time.time()
    for _ in range(num_runs):
        hybrid_search(query, k=5)
    hybrid_time = (time.time() - start) / num_runs * 1000

    return {
        'keyword': keyword_time,
        'semantic': semantic_time,
        'hybrid': hybrid_time
    }

test_query = "action movie with hero"
print(f"\nBenchmarking query: '{test_query}'")
print(f"Running {100} iterations...\n")

times = benchmark_search(test_query)

print("Average Search Times:")
print(f"  Keyword Search:  {times['keyword']:.2f} ms")
print(f"  Semantic Search: {times['semantic']:.2f} ms")
print(f"  Hybrid Search:   {times['hybrid']:.2f} ms")

# ============================================================================
# STEP 7: Summary
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE - Summary")
print("=" * 70)

print("""
✓ Demonstrated core concepts:
  1. Vector representation of text (TF-IDF simulation)
  2. Cosine similarity for semantic matching
  3. Three search methods: Keyword, Semantic, Hybrid
  4. Filtering by metadata (rating)
  5. Performance benchmarking

✓ Key Takeaways:
  • Semantic search understands meaning, not just words
  • Keyword search is fast but literal
  • Hybrid search combines strengths of both
  • Vector similarity enables finding related content

✓ Next Steps:
  1. Replace TF-IDF with real embeddings (sentence-transformers)
  2. Connect to OpenSearch for production scale
  3. Add more sophisticated ranking
  4. Implement caching and optimization

For production implementation with real embeddings and OpenSearch,
see: embeddings.py, indexer.py, search.py, app.py
""")

print("=" * 70)
print("  Demo completed successfully!")
print("=" * 70)
