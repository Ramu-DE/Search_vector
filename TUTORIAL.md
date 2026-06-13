# Complete Tutorial: Building AI-Powered Search from Scratch

This tutorial will guide you through building a production-ready AI-powered search application step by step.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Understanding the Basics](#understanding-the-basics)
4. [Building the Index](#building-the-index)
5. [Implementing Search](#implementing-search)
6. [Optimization](#optimization)
7. [Testing and Monitoring](#testing-and-monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of REST APIs
- Familiarity with AWS services (basic)

### Required Accounts
- AWS Account with access to:
  - Amazon OpenSearch Serverless
  - Amazon Bedrock (optional, for Titan embeddings)
  - Amazon EC2 (for hosting)

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Internet connection

---

## Environment Setup

### Step 1: Clone or Download the Repository

```bash
# Create project directory
mkdir ai-powered-search
cd ai-powered-search

# Copy all project files here
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Set Up AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-west-2
```

### Step 5: Set Up OpenSearch Serverless

#### Create Collection via AWS Console

1. Navigate to Amazon OpenSearch Service
2. Click "Serverless" → "Collections"
3. Click "Create collection"
4. Configure:
   - **Name**: `movies-search`
   - **Type**: Search
   - **Encryption**: AWS owned key (or your own KMS key)
   - **Network**: Public access (for development)
5. Set up access policies:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR-ACCOUNT-ID:user/YOUR-USER"
      },
      "Action": "aoss:*",
      "Resource": "collection/movies-search"
    }
  ]
}
```

6. Note your collection endpoint (e.g., `https://xyz.us-west-2.aoss.amazonaws.com`)

#### Set Environment Variable

```bash
export AOSS_VECTORSEARCH_ENDPOINT=https://xyz.us-west-2.aoss.amazonaws.com
```

---

## Understanding the Basics

### What is a Vector?

A vector is simply a list of numbers representing something:

```python
# Text as numbers
text = "Hello, world!"
vector = [0.25, -0.18, 0.92, ..., 0.14]  # 384 numbers

# Each dimension captures some aspect of meaning
```

### How Embeddings Work

```
Input Text → Embedding Model → Vector
─────────────────────────────────────
"action movie"  →  [0.8, 0.1, ..., 0.3]
"thriller film" →  [0.7, 0.2, ..., 0.4]
"recipe book"   →  [0.1, 0.9, ..., 0.2]
```

Similar texts have similar vectors (measured by distance/angle).

### Search Process Flow

```
1. Index Time:
   Document → Generate Embedding → Store in OpenSearch
   
2. Query Time:
   Query → Generate Embedding → Find Similar Vectors → Return Results
```

---

## Building the Index

### Step 1: Test Embedding Generation

```bash
# Test the embedding module
python embeddings.py
```

**Expected Output:**
```
=== Testing Local Embeddings ===
Loading local model: all-MiniLM-L6-v2
Loaded model with dimension: 384

Text: This is a test sentence for embedding generation
Embedding dimension: 384
First 10 values: [0.048, -0.023, 0.156, ...]

Batch embeddings generated: 3
```

### Step 2: Create the Index

Create a simple script `create_index.py`:

```python
from indexer import OpenSearchIndexer

# Initialize indexer
indexer = OpenSearchIndexer()

# Create index
success = indexer.create_index(delete_if_exists=True)

if success:
    print("✅ Index created successfully!")
else:
    print("❌ Failed to create index")
```

Run it:
```bash
python create_index.py
```

### Step 3: Load Sample Data

Use the built-in sample data:

```bash
# Index sample movies
python indexer.py
```

**Expected Output:**
```
=== OpenSearch Indexer Demo ===

1. Creating index...
Successfully created index: movies
Vector field: title_vector, Dimension: 384

2. Indexing sample movies...
Generating embeddings...
Bulk indexing documents...
100%|████████████████████| 10/10

Indexing complete:
  Successful: 10
  Failed: 0

Force merging segments...
Force merge complete

3. Index statistics:
  Documents: 10
  Size: 245760 bytes
  Segments: 1
```

### Step 4: Verify the Index

```python
from indexer import OpenSearchIndexer

indexer = OpenSearchIndexer()
stats = indexer.get_index_stats()

print(f"Documents indexed: {stats['document_count']}")
print(f"Index size: {stats['size_bytes']} bytes")
print(f"Segments: {stats['segment_count']}")
```

---

## Implementing Search

### Step 1: Basic Semantic Search

Create `test_search.py`:

```python
from search import VectorSearchEngine

# Initialize search engine
search = VectorSearchEngine()

# Perform search
results = search.semantic_search(
    query="movie about friendship",
    k=3
)

# Display results
search.print_results(results)
```

Run it:
```bash
python test_search.py
```

**Expected Output:**
```
Search engine initialized for index: movies
  Found 10 documents, returning top 3
  Latency: 45.23 ms

1. Forrest Gump (1994)
   Rating: 8.8 | Score: 0.8234
   Genre: Drama, Romance
   Plot: The presidencies of Kennedy and Johnson...

2. The Shawshank Redemption (1994)
   Rating: 9.3 | Score: 0.7891
   Genre: Drama
   Plot: Two imprisoned men bond over a number of years...

3. Goodfellas (1990)
   Rating: 8.7 | Score: 0.7456
   Genre: Crime, Drama
   Plot: The story of Henry Hill and his life in the mob...
```

### Step 2: Add Filters

```python
# Search with rating filter
results = search.semantic_search(
    query="action movie",
    k=5,
    filters={"rating": {"gte": 8.5}}
)
```

### Step 3: Hybrid Search

```python
# Combine keyword and semantic search
results = search.hybrid_search(
    query="science fiction thriller",
    k=5,
    semantic_weight=0.7  # 70% semantic, 30% keyword
)
```

### Step 4: Compare Methods

```python
# Compare all search methods
comparison = search.compare_search_methods(
    query="crime drama",
    k=3
)

print("\nKeyword Results:")
search.print_results(comparison['keyword'])

print("\nSemantic Results:")
search.print_results(comparison['semantic'])

print("\nHybrid Results:")
search.print_results(comparison['hybrid'])
```

---

## Optimization

### Step 1: Tune HNSW Parameters

Edit `config.py`:

```python
# For higher recall (slower but more accurate)
HNSW_M = 32                    # Default: 16
HNSW_EF_CONSTRUCTION = 512     # Default: 256
HNSW_EF_SEARCH = 200           # Default: 100

# For lower latency (faster but less accurate)
HNSW_M = 8
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 50
```

### Step 2: Implement Caching

```python
from functools import lru_cache

class CachedSearchEngine(VectorSearchEngine):
    @lru_cache(maxsize=100)
    def cached_search(self, query: str, k: int = 10):
        """Cache search results for repeated queries"""
        return self.semantic_search(query, k)
```

### Step 3: Batch Queries

```python
# Process multiple queries efficiently
queries = [
    "action movie",
    "romantic comedy",
    "sci-fi thriller"
]

# Generate all embeddings at once
embeddings = search.embedding_generator.batch_generate(queries)

# Process each query
for query, embedding in zip(queries, embeddings):
    # Use pre-computed embedding
    results = search.semantic_search(query, k=5)
```

### Step 4: Monitor Performance

```python
import time

def search_with_metrics(search_engine, query):
    start = time.time()
    results = search_engine.semantic_search(query)
    latency_ms = (time.time() - start) * 1000
    
    print(f"Query: {query}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Results: {len(results)}")
    print(f"Top score: {results[0]['_score']:.4f}")
    
    return results, latency_ms
```

---

## Testing and Monitoring

### Step 1: Create Test Suite

Create `tests/test_search.py`:

```python
import unittest
from search import VectorSearchEngine

class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search = VectorSearchEngine()
    
    def test_semantic_search(self):
        """Test basic semantic search"""
        results = self.search.semantic_search("action movie", k=5)
        self.assertGreater(len(results), 0)
        self.assertIn('title', results[0])
        self.assertIn('_score', results[0])
    
    def test_filters(self):
        """Test search with filters"""
        results = self.search.semantic_search(
            "any movie",
            k=10,
            filters={"rating": {"gte": 9.0}}
        )
        for result in results:
            self.assertGreaterEqual(result['rating'], 9.0)
    
    def test_hybrid_search(self):
        """Test hybrid search"""
        results = self.search.hybrid_search("thriller", k=5)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m pytest tests/
```

### Step 2: Performance Benchmarking

Create `benchmark.py`:

```python
import time
import statistics
from search import VectorSearchEngine

def benchmark_search(queries, iterations=10):
    """Benchmark search performance"""
    search = VectorSearchEngine()
    
    latencies = []
    
    for _ in range(iterations):
        for query in queries:
            start = time.time()
            results = search.semantic_search(query, k=10)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
    
    print(f"Total queries: {len(latencies)}")
    print(f"P50 latency: {statistics.median(latencies):.2f} ms")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"P99 latency: {statistics.quantiles(latencies, n=100)[98]:.2f} ms")
    print(f"Mean latency: {statistics.mean(latencies):.2f} ms")

# Run benchmark
test_queries = [
    "action thriller",
    "romantic comedy",
    "sci-fi adventure",
    "crime drama",
    "horror movie"
]

benchmark_search(test_queries, iterations=20)
```

### Step 3: Monitor Index Health

```python
def check_index_health(indexer):
    """Check index health metrics"""
    stats = indexer.get_index_stats()
    
    print("Index Health Report")
    print("=" * 50)
    print(f"Document Count: {stats['document_count']}")
    print(f"Index Size: {stats['size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"Segment Count: {stats['segment_count']}")
    
    # Recommendations
    if stats['segment_count'] > 5:
        print("\n⚠️  Warning: High segment count detected")
        print("   Recommendation: Run force merge")
    
    if stats['document_count'] == 0:
        print("\n❌ Error: No documents in index")
    else:
        print("\n✅ Index is healthy")

# Run health check
from indexer import OpenSearchIndexer
indexer = OpenSearchIndexer()
check_index_health(indexer)
```

---

## Running the Web Application

### Step 1: Launch Streamlit App

```bash
# Run the Streamlit application
streamlit run app.py
```

### Step 2: Access the UI

Open your browser to: `http://localhost:8501`

### Step 3: Test Search

1. Enter a query like "action thriller"
2. Adjust minimum rating filter
3. Choose search method
4. Click "Search"
5. Compare results across methods

---

## Troubleshooting

### Issue 1: "OPENSEARCH_ENDPOINT not set"

**Solution:**
```bash
export AOSS_VECTORSEARCH_ENDPOINT=https://your-endpoint.aoss.amazonaws.com
```

### Issue 2: "Access Denied" errors

**Solution:**
- Check IAM permissions
- Verify data access policy in OpenSearch console
- Ensure AWS credentials are configured

### Issue 3: Slow embedding generation

**Solution:**
- Use GPU if available: `pip install sentence-transformers[cuda]`
- Use Bedrock for faster inference (cloud-based)
- Batch queries instead of one-by-one

### Issue 4: Low recall (missing relevant results)

**Solution:**
- Increase `ef_search` parameter
- Try hybrid search instead of pure semantic
- Fine-tune embedding model on your domain

### Issue 5: High latency

**Solution:**
- Decrease `ef_search` parameter
- Enable request caching
- Use quantization (FP16 or binary)
- Force merge segments

---

## Next Steps

### Advanced Topics

1. **Fine-tune Embeddings**
   - Collect domain-specific training data
   - Fine-tune sentence-transformers model
   - Evaluate recall improvement

2. **Implement Reranking**
   - Use cross-encoder for second-stage ranking
   - Improve top-K precision

3. **Add Multi-modal Search**
   - Index images with CLIP
   - Search by image or text

4. **Scale to Production**
   - Set up monitoring (CloudWatch)
   - Implement rate limiting
   - Add authentication/authorization

5. **A/B Testing**
   - Compare different embedding models
   - Test search algorithm parameters
   - Measure user engagement

---

## Resources

### Documentation
- [OpenSearch Documentation](https://opensearch.org/docs/)
- [AWS OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/)
- [Sentence Transformers](https://www.sbert.net/)

### Community
- [OpenSearch Forum](https://forum.opensearch.org/)
- [OpenSearch Slack](https://opensearch.org/slack.html)

### Example Datasets
- [IMDB Movies Dataset](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
- [MS MARCO](https://microsoft.github.io/msmarco/)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)

---

## Summary

You've learned to:
- ✅ Set up OpenSearch Serverless
- ✅ Generate embeddings
- ✅ Create vector indices
- ✅ Implement semantic search
- ✅ Build hybrid search
- ✅ Optimize performance
- ✅ Monitor and troubleshoot

**Congratulations! You now have a production-ready AI-powered search application!** 🎉
