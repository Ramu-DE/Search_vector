# AI-Powered Search Applications with AWS OpenSearch

A comprehensive guide from basics to advanced implementation of production-ready AI-powered search applications using Amazon OpenSearch and AWS Bedrock.

## 📚 Table of Contents

### 🎯 NEW: Comprehensive Search Engine Guides

**Complete standalone guides with theory, visualizations, and production code:**

- **[📖 Search Engines Index](./SEARCH_ENGINES_INDEX.md)** - Navigation hub for all guides
- **[🔍 Dense Vector Search Guide](./DENSE_VECTOR_SEARCH_GUIDE.md)** - Semantic search with embeddings (NDCG: 0.85)
- **[⚡ Sparse Encoding Guide](./SPARSE_ENCODING_COMPLETE_GUIDE.md)** - Fast, interpretable search (5ms latency, 10x smaller)
- **[🎯 Hybrid Search Guide](./HYBRID_SEARCH_COMPLETE_GUIDE.md)** - Best quality (NDCG: 0.91) - **Production recommended**
- **[📏 Distance Metrics Guide](./DISTANCE_METRICS_COMPLETE_GUIDE.md)** - All 5 metrics with examples

Each guide includes:
- ✅ Theory and core concepts
- ✅ Embedded visualizations (19 total)
- ✅ Step-by-step implementation
- ✅ Complete working code examples
- ✅ Performance benchmarks
- ✅ Production deployment guidance

---

### Original Documentation

1. [Introduction to OpenSearch](#introduction-to-opensearch)
2. [Search Evolution](#search-evolution)
3. [Vector Fundamentals](#vector-fundamentals)
4. [Semantic Search & KNN](#semantic-search--knn)
5. [Architecture Deep Dive](#architecture-deep-dive)
6. [Implementation Guide](#implementation-guide)
7. [Cost Optimization](#cost-optimization)
8. [Performance Tuning](#performance-tuning)
9. [Best Practices](#best-practices)

---

## Introduction to OpenSearch

### What is OpenSearch?

**OpenSearch** is a community-driven, open-source platform that provides:
- **Search** capabilities
- **Analytics** processing
- **Vector database** functionality
- Integrated tools for observability, security, and visualization

### OpenSearch by Numbers

| Metric | Value |
|--------|-------|
| Project Downloads | 1.4B+ |
| Monthly Page Views | 1M+ |
| Active Contributors | 3K+ |
| Active Organizations | 400+ |
| GitHub Repositories | 130+ |
| Community Members | 11K+ (Slack + Forums) |

### Amazon OpenSearch Service Features

```
┌─────────────────────────────────────────────────────────────┐
│              Amazon OpenSearch Service                      │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│  Automated  │ Cost-Opt    │  Resilient  │    Secure       │
│             │             │             │                 │
│ • API-driven│ • Reserved  │ • Multi-AZ  │ • Fine-grained  │
│   deploy    │   instances │ • 99.99% SLA│   access        │
│ • Patches   │ • Tiered    │ • Snapshots │ • Encryption    │
│ • Serverless│   storage   │ • Self-heal │ • SAML/IAM      │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

**Performance**: 6x improvement from OpenSearch 1.3 to current version
**Latency**: Single-digit milliseconds for lexical and vector queries

---

## Search Evolution

### The Four Stages of Search

```
Surface → Twilight → Midnight → Abyss
   ↓         ↓          ↓         ↓
Keyword   Semantic   Hybrid    Agentic
```

### 1. **Keyword Search (BM25/TF-IDF)**
```
┌──────────────────────────────────────┐
│  Query: "apple headphones"           │
│                                      │
│  Matching: Exact term matching       │
│  ✓ Fast and interpretable           │
│  ✗ Cannot understand meaning        │
│  ✗ Miss synonyms                    │
└──────────────────────────────────────┘
```

### 2. **Semantic Search (ML Vectors)**
```
┌──────────────────────────────────────┐
│  Query: "best laptop for students"   │
│                                      │
│  Matching: Meaning-based vectors     │
│  ✓ Understands synonyms             │
│  ✓ Context awareness                │
│  ✗ Higher compute cost              │
└──────────────────────────────────────┘
```

### 3. **Hybrid Search**
```
        Keyword (BM25)
              ↓
         Normalization  ←→  Semantic (Vector)
              ↓
      Combined Ranking
```

### 4. **Agentic Search** (OpenSearch 3.3+)
```
┌────────────────────────────────────────┐
│  Natural Language → Agent Plans        │
│  Agent → Tools → Memory → Execute      │
│  Result → Reasoning → Answer           │
└────────────────────────────────────────┘
```

---

## Vector Fundamentals

### What is a Vector?

A vector is a **list of numbers** representing attributes of an item.

#### Real-World Example: House Properties
```python
# Vector representation of houses
v1 = [5, 4, 854, 1100000]  # 5 bed, 4 bath, 854 sqm, $1.1M
v2 = [4, 3, 335, 530000]   # 4 bed, 3 bath, 335 sqm, $530K
v3 = [6, 4, 530, 2100000]  # 6 bed, 4 bath, 530 sqm, $2.1M
v4 = [6, 4, 500, 2500000]  # 6 bed, 4 bath, 500 sqm, $2.5M

# Attributes:
# 1. Number of bedrooms
# 2. Number of bathrooms
# 3. Size in square meters
# 4. Price in dollars
```

### Text as Vectors (Embeddings)

```
Text: "I am attending a great webinar about vector search"
        ↓ (Embedding Model)
Vector: [0.743, 0.720, -0.325, 0.195, 0.835, -0.945, ...]
        
        Magnitude + Direction = Semantic Meaning
```

### Data Types to Vectors

```
┌─────────────┐
│   IMAGE     │ ──→  ML Model  ──→  [0.35, 0.10, 0.00, 0.90, ...]
└─────────────┘

┌─────────────┐
│  DOCUMENT   │ ──→  Embedding ──→  [0.35, 0.10, 0.00, 0.80, ...]
└─────────────┘

┌─────────────┐
│   AUDIO     │ ──→  ML Model  ──→  [0.15, 0.10, 0.00, 0.70, ...]
└─────────────┘
```

### What is a Vector Database?

A vector database enables you to:
1. **Store** high-dimensional vectors at scale
2. **Query** efficiently using nearest neighbor algorithms
3. **Update/Delete** embeddings as data changes
4. **Filter** by similarity and metadata

---

## Semantic Search & KNN

### Content Similarity

```
Your Favorite Song:  [0.1, 0.5]
                        ↓
              Calculate Distance
                        ↓
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
    [0.2, 0.4]    [0.8, 0.2]    [0.15, 0.48]
    Similar #1    Different      Similar #2
```

### Distance Metrics

#### 1. **Euclidean Distance (L2)**
```
Formula: √[(x₂-x₁)² + (y₂-y₁)²]

Use cases:
• Measurements and counts
• Recommendation systems
• Physical distance calculations

      A(x₁,y₁) ────────→ B(x₂,y₂)
           Straight line
```

#### 2. **Cosine Similarity**
```
Formula: 1 - (A·B)/(||A||·||B||)

Use cases:
• Semantic search
• Document classification
• Text similarity

        A
       ╱
      ╱ θ (angle)
     ╱___________B
     
Measures angle, not distance
```

#### 3. **Dot Product**
```
Formula: A·B = Σ(aᵢ × bᵢ)

Use cases:
• Collaborative filtering
• Recommendation systems
```

### k-Nearest Neighbors (k-NN)

Find the "Top K" most similar items.

#### Exact (Brute-force) k-NN
```
Query: [0.1, 0.2, 0.6, ..., 0.3, 0.7]
         ↓
Calculate distance to ALL vectors
         ↓
Rank by distance
         ↓
Return Top K

⚠️ Slow for large datasets
✓ 100% accurate
```

#### Approximate k-NN (HNSW)

**Hierarchical Navigable Small Worlds**

```
Layer 2:  ●─────────●  (Entry point, few nodes)
          │         │
Layer 1:  ●───●───●───●  (Intermediate connections)
          │ ╱ │ ╲ │   │
Layer 0:  ●─●─●─●─●─●─●─●  (All vectors, dense graph)
          
Search: Navigate from top layer down
        Following closest neighbors
        
⚡ Fast (milliseconds)
✓ ~95-99% recall with proper tuning
```

### Dense vs Sparse Encoding

#### Dense Encoding
```
Text: "Apple products are expensive"
  ↓
[0.712, 0.049, 0.914, 0.930, 0.224, 0.913, 0.578, 0.364, ...]
  
Dimensions: 384-768 (typical)
Model: BERT, Cohere, Titan, OpenAI
Storage: Vector index (HNSW)
```

#### Sparse Encoding
```
Text: "Apple products are expensive"
  ↓
{
  "apple": 0.85,
  "products": 0.72,
  "expensive": 0.68,
  "gadget": 0.45,      # Expanded term
  "costly": 0.42       # Synonym
}

Dimensions: 30,522 (BERT vocab), ~1% non-zero
Storage: Inverted index (Lucene)
```

**Comparison:**

| Feature | Dense | Sparse |
|---------|-------|--------|
| Index Size | Large | 10x smaller |
| RAM at Query | High | 0 increase |
| Semantic Understanding | Deep | Moderate |
| Interpretability | Low | High |
| Speed | Fast (with optimization) | Very Fast |

---

## Architecture Deep Dive

### Overall System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      AWS Cloud                           │
│                                                          │
│  ┌────────────┐                                         │
│  │   User     │──1. Query──┐                           │
│  └────────────┘            │                           │
│                            ↓                            │
│  ┌─────────────────────────────────────────────┐       │
│  │         Application Layer (EC2)             │       │
│  │  ┌──────────────┐    ┌──────────────────┐  │       │
│  │  │  Embedding   │    │  Query Handler   │  │       │
│  │  │   Model      │    │                  │  │       │
│  │  └──────────────┘    └──────────────────┘  │       │
│  └─────────────────────────────────────────────┘       │
│              ↓                     ↓                    │
│  ┌──────────────────┐  ┌──────────────────────┐       │
│  │  Amazon Bedrock  │  │  OpenSearch          │       │
│  │  (Embeddings)    │  │  Serverless          │       │
│  │  - Titan         │  │  - Vector Index      │       │
│  │  - Cohere        │  │  - k-NN Search       │       │
│  └──────────────────┘  └──────────────────────┘       │
│                                    ↓                    │
│                         ┌──────────────────┐           │
│                         │   Amazon S3      │           │
│                         │   (Backup)       │           │
│                         └──────────────────┘           │
└──────────────────────────────────────────────────────────┘
```

### Ingestion Flow

```
Step 1: Raw Data
┌─────────────────────────────────────┐
│ {"title": "A Film for Friends",    │
│  "year": 2011,                      │
│  "plot": "A story about..."}        │
└─────────────────────────────────────┘
            ↓
Step 2: Generate Embeddings
┌─────────────────────────────────────┐
│  Amazon Bedrock / Local Model       │
│  • Titan Embeddings (dim: 1536)    │
│  • all-MiniLM-L6-v2 (dim: 384)     │
└─────────────────────────────────────┘
            ↓
Step 3: Enriched Document
┌─────────────────────────────────────┐
│ {"title": "A Film for Friends",    │
│  "year": 2011,                      │
│  "plot": "A story about...",        │
│  "title_vector": [0.0056, 0.0065,  │
│                   ..., 0.0234]}     │
└─────────────────────────────────────┘
            ↓
Step 4: Index in OpenSearch
┌─────────────────────────────────────┐
│     OpenSearch Serverless           │
│  ┌───────────────────────────┐     │
│  │  Inverted Index (Text)    │     │
│  │  Vector Index (HNSW)      │     │
│  └───────────────────────────┘     │
└─────────────────────────────────────┘
```

### Search Flow

```
Step 1: User Query
┌──────────────────────────────────┐
│ "Movie to watch with friends"    │
└──────────────────────────────────┘
            ↓
Step 2: Vectorize Query
┌──────────────────────────────────┐
│ Amazon Bedrock                   │
│ query_vector = [0.0048, 0.0089,  │
│                 ..., 0.0123]     │
└──────────────────────────────────┘
            ↓
Step 3: k-NN Search
┌──────────────────────────────────┐
│ OpenSearch Query:                │
│ {                                │
│   "knn": {                       │
│     "title_vector": {            │
│       "vector": query_vector,    │
│       "k": 10                    │
│     }                            │
│   },                             │
│   "filter": {"rating": {"gte":8}}│
│ }                                │
└──────────────────────────────────┘
            ↓
Step 4: Results
┌──────────────────────────────────┐
│ Top 10 Similar Movies            │
│ 1. "Friendship Forever" (0.95)   │
│ 2. "The Hangout" (0.92)         │
│ 3. "Buddies Unite" (0.89)       │
│ ...                              │
└──────────────────────────────────┘
```

### Tiered Vector Storage

```
┌─────────────────────────────────────────────────┐
│ Performance Tiers (Top to Bottom = Cost)        │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────────────────────┐          │
│  │   Exact k-NN (In-Memory)         │  $$$     │
│  │   • Most accurate                │          │
│  │   • Highest cost                 │          │
│  │   • Lowest latency               │          │
│  └──────────────────────────────────┘          │
│                ↓                                │
│  ┌──────────────────────────────────┐          │
│  │   In-Memory HNSW                 │  $$      │
│  │   • High performance             │          │
│  │   • 95-99% recall                │          │
│  │   • Millisecond latency          │          │
│  └──────────────────────────────────┘          │
│                ↓                                │
│  ┌──────────────────────────────────┐          │
│  │   Disk-Based Vectors             │  $       │
│  │   • 32x memory reduction         │          │
│  │   • Still performant             │          │
│  │   • SSD storage                  │          │
│  └──────────────────────────────────┘          │
│                ↓                                │
│  ┌──────────────────────────────────┐          │
│  │   S3 Vectors                     │  ¢       │
│  │   • Cheapest                     │          │
│  │   • Massive scale                │          │
│  │   • Higher latency OK            │          │
│  └──────────────────────────────────┘          │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Implementation Guide

### Prerequisites

```bash
# Required AWS Services
- Amazon OpenSearch Serverless
- Amazon Bedrock (with Titan access)
- Amazon EC2 (for application hosting)
- Amazon S3 (optional, for backups)
```

### Index Creation

```python
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json

# Setup AWS credentials
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    'us-west-2',
    'aoss',  # OpenSearch Serverless
    session_token=credentials.token
)

# Create OpenSearch client
client = OpenSearch(
    hosts=[{'host': 'your-collection-endpoint.aoss.amazonaws.com', 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

# Define index with vector field
index_body = {
    "settings": {
        "index.knn": True,  # Enable k-NN
        "index.knn.algo_param.ef_search": 100
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "plot": {"type": "text"},
            "year": {"type": "integer"},
            "rating": {"type": "float"},
            "genre": {"type": "keyword"},
            # Vector field
            "title_vector": {
                "type": "knn_vector",
                "dimension": 1536,  # Titan embedding dimension
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                    "parameters": {
                        "ef_construction": 256,
                        "m": 16
                    }
                }
            }
        }
    }
}

# Create the index
response = client.indices.create(
    index='movies',
    body=index_body
)
print(f"Index created: {response}")
```

### Data Ingestion with Embeddings

```python
import boto3
import json
from opensearchpy import helpers

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

def generate_embedding(text):
    """Generate embedding using Amazon Bedrock Titan"""
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({'inputText': text})
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def prepare_documents(movies_data):
    """Prepare documents with embeddings for bulk indexing"""
    documents = []
    
    for movie in movies_data:
        # Generate embedding for title
        title_embedding = generate_embedding(movie['title'])
        
        # Prepare document
        doc = {
            '_index': 'movies',
            '_source': {
                'title': movie['title'],
                'plot': movie.get('plot', ''),
                'year': movie.get('year'),
                'rating': movie.get('rating'),
                'genre': movie.get('genre', []),
                'title_vector': title_embedding
            }
        }
        documents.append(doc)
    
    return documents

# Sample data
movies_data = [
    {
        'title': 'A Film for Friends',
        'plot': 'A heartwarming story about friendship',
        'year': 2011,
        'rating': 7.8,
        'genre': ['Drama', 'Comedy']
    },
    {
        'title': 'The Great Adventure',
        'plot': 'An epic journey across continents',
        'year': 2015,
        'rating': 8.2,
        'genre': ['Adventure', 'Action']
    }
]

# Prepare documents with embeddings
documents = prepare_documents(movies_data)

# Bulk index
success, failed = helpers.bulk(
    client,
    documents,
    stats_only=False,
    raise_on_error=False
)

print(f"Successfully indexed: {success} documents")
print(f"Failed: {len(failed)} documents")
```

### Vector Search Query

```python
def search_movies(query_text, k=10, min_rating=7.0):
    """
    Perform semantic search with filters
    
    Args:
        query_text: Natural language query
        k: Number of results to return
        min_rating: Minimum rating filter
    """
    
    # Generate query embedding
    query_embedding = generate_embedding(query_text)
    
    # Construct OpenSearch query
    search_body = {
        "size": k,
        "_source": {
            "includes": ["title", "plot", "year", "rating", "genre"]
        },
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "title_vector": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "rating": {
                                "gte": min_rating
                            }
                        }
                    }
                ]
            }
        }
    }
    
    # Execute search
    response = client.search(
        body=search_body,
        index='movies'
    )
    
    # Process results
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'title': hit['_source']['title'],
            'plot': hit['_source']['plot'],
            'year': hit['_source']['year'],
            'rating': hit['_source']['rating'],
            'score': hit['_score']
        })
    
    return results

# Example usage
query = "Movie to watch with friends"
results = search_movies(query, k=5, min_rating=7.5)

print(f"\nSearch results for: '{query}'\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']} ({result['year']}) - Rating: {result['rating']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Plot: {result['plot'][:100]}...\n")
```

### Hybrid Search (Keyword + Semantic)

```python
def hybrid_search(query_text, k=10, min_rating=7.0, semantic_weight=0.6):
    """
    Hybrid search combining BM25 and k-NN
    
    Args:
        query_text: Search query
        k: Number of results
        min_rating: Minimum rating filter
        semantic_weight: Weight for semantic search (0-1)
    """
    
    query_embedding = generate_embedding(query_text)
    keyword_weight = 1 - semantic_weight
    
    search_body = {
        "size": k,
        "_source": ["title", "plot", "year", "rating", "genre"],
        "query": {
            "bool": {
                "should": [
                    # Keyword search (BM25)
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title^2", "plot"],  # Title boosted 2x
                            "type": "best_fields",
                            "boost": keyword_weight
                        }
                    },
                    # Semantic search (k-NN)
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "knn_score",
                                "lang": "knn",
                                "params": {
                                    "field": "title_vector",
                                    "query_value": query_embedding,
                                    "space_type": "cosinesimil"
                                }
                            },
                            "boost": semantic_weight
                        }
                    }
                ],
                "filter": [
                    {"range": {"rating": {"gte": min_rating}}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    
    response = client.search(body=search_body, index='movies')
    
    results = []
    for hit in response['hits']['hits']:
        results.append({
            'title': hit['_source']['title'],
            'plot': hit['_source']['plot'],
            'year': hit['_source']['year'],
            'rating': hit['_source']['rating'],
            'score': hit['_score']
        })
    
    return results

# Example usage
query = "Uplifting underdog story"
results = hybrid_search(query, k=5, semantic_weight=0.7)

print(f"\nHybrid search results for: '{query}'\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']} - Score: {result['score']:.4f}")
```

---

## Cost Optimization

### The Optimization Triangle

```
                  COST
                   $
                   │
                   │
        Pick ──────┼────── Two
        Any         │
        Two         │
                   │
    RECALL ────────┴────────── LATENCY
       ◎                          ⚡
```

### Quantization Strategies

```
Memory Savings Progression:
┌────────────────────────────────────────────────┐
│ FP32 (Full Precision)                          │
│ 4 bytes per value            [Baseline]        │
│ ────────────────────────────────────────────  │
│                                                │
│ FP16 (Half Precision)        50% savings       │
│ 2 bytes per value            ▓▓▓▓▓▓▓▓▓▓       │
│ ────────────────────────────────────────────  │
│                                                │
│ INT8 (8-bit Integer)         75% savings       │
│ 1 byte per value             ▓▓▓▓▓            │
│ ────────────────────────────────────────────  │
│                                                │
│ Binary Quantization          97% savings       │
│ 1 bit per value              ▓                │
│ (with rescoring: <2% recall loss)             │
└────────────────────────────────────────────────┘
```

### Cost Optimization Strategies

```python
# 1. Binary Quantization Example
index_settings = {
    "settings": {
        "index.knn": True
    },
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "parameters": {
                        "ef_construction": 256,
                        "m": 16
                    }
                },
                "data_type": "binary"  # Binary quantization
            }
        }
    }
}

# 2. Disk-based vectors (OpenSearch 2.17+)
index_settings_disk = {
    "settings": {
        "index.knn": True,
        "knn.vector.mode": "on_disk"  # Store on SSD
    },
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": 384,  # Reduced dimensions
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16
                    }
                }
            }
        }
    }
}

# 3. Right-sized dimensions
# Titan: 1536 → 768 (use v2 model)
# All-MiniLM: 384 (smaller, good for most use cases)
```

### Cost Comparison Table

| Strategy | Memory Savings | Recall Impact | Best For |
|----------|---------------|---------------|----------|
| FP32 (Baseline) | 0% | 100% | Maximum accuracy needed |
| FP16 | 50% | ~0.5% loss | Most production workloads |
| INT8 | 75% | ~1% loss | Large-scale deployments |
| Binary + Rescore | 97% | <2% loss | Cost-critical applications |
| Disk-based | 32x | Minimal | Cold data, massive scale |
| Sparse encoding | 10x index size | N/A | Document-only mode |

---

## Performance Tuning

### HNSW Parameters Deep Dive

```python
# Index-time parameters (affect build quality)
index_config = {
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "parameters": {
                        # Graph connectivity
                        "m": 16,  # Default: 16, Range: 2-100
                                 # Higher = better recall, more memory
                                 # Try 32 for high-recall scenarios
                        
                        # Build quality
                        "ef_construction": 256,  # Default: 128, Range: 100-1000
                                                # Higher = better graph, slower build
                                                # Only affects indexing time
                    }
                }
            }
        }
    }
}

# Query-time parameter (tune for recall vs latency)
search_query = {
    "size": 10,
    "query": {
        "knn": {
            "vector": {
                "vector": query_embedding,
                "k": 10,
                "ef_search": 100  # Default: 100, Range: 10-1000
                                  # THIS IS THE KEY TUNING KNOB!
                                  # Higher = better recall, slower search
                                  # Start at 100, increase until recall plateaus
            }
        }
    }
}
```

### Parameter Tuning Guide

```
                  HNSW Parameter Effects
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  m (graph connectivity)                                 │
│  ──────────────────────────────────────────────────    │
│  2   8   16   24   32                                   │
│  │   │   │    │    │                                    │
│  Fast      Balanced       High Recall                   │
│  Low Mem               High Memory                      │
│                                                         │
│  ef_construction (build quality)                        │
│  ──────────────────────────────────────────────────    │
│  100  256  512  1000                                    │
│  │    │    │    │                                       │
│  Fast Build    Slow Build                               │
│  Lower Quality High Quality                             │
│                                                         │
│  ef_search (query-time accuracy)                        │
│  ──────────────────────────────────────────────────    │
│  10   100  500  1000                                    │
│  │    │    │    │                                       │
│  Fast      Balanced       Exhaustive                    │
│  ~90% Recall          ~99% Recall                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Pre-filtering vs Post-filtering

```python
# PRE-FILTERING (Faster - Reduces candidate set)
# Use when filter is highly selective
pre_filter_query = {
    "size": 10,
    "query": {
        "bool": {
            "must": [
                {
                    "knn": {
                        "vector": {
                            "vector": query_embedding,
                            "k": 10
                        }
                    }
                }
            ],
            "filter": [  # Applied BEFORE k-NN search
                {"range": {"rating": {"gte": 8.5}}},  # Only 10% of docs
                {"term": {"genre": "Action"}}
            ]
        }
    }
}

# POST-FILTERING (Slower - Searches all, then filters)
# Use when filter is less selective
post_filter_query = {
    "size": 10,
    "query": {
        "knn": {
            "vector": {
                "vector": query_embedding,
                "k": 100  # Search 100, then filter to 10
            }
        }
    },
    "post_filter": {  # Applied AFTER k-NN search
        "range": {"rating": {"gte": 7.0}}  # 80% of docs pass
    }
}
```

### Force Merge for Better Performance

```python
# After bulk indexing, merge segments for 20-40% latency reduction
client.indices.forcemerge(
    index='movies',
    max_num_segments=1,  # Merge to single segment
    wait_for_completion=True
)

# Check segment count
stats = client.indices.stats(index='movies')
segment_count = stats['indices']['movies']['primaries']['segments']['count']
print(f"Segment count: {segment_count}")
```

### Latency Optimization Checklist

```
┌─────────────────────────────────────────────────────┐
│ Performance Optimization Checklist                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│ ☐ Tune ef_search (start 100, increase gradually)   │
│ ☐ Use pre-filtering for selective filters          │
│ ☐ Force-merge to 1 segment after bulk load         │
│ ☐ Enable request cache for repeated queries        │
│ ☐ Use FP16 quantization (50% memory, ~0.5% loss)   │
│ ☐ Consider disk-based for 32x memory reduction     │
│ ☐ Monitor P95/P99 latencies, not just averages     │
│ ☐ Use connection pooling in client                 │
│ ☐ Implement caching layer for hot queries          │
│ ☐ Batch requests when possible                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Best Practices

### When Each Search Method Fails

```
┌─────────────────────────────────────────────────────┐
│ When KEYWORD Search Fails                           │
├─────────────────────────────────────────────────────┤
│ • "Best laptop for students" - no exact "best"      │
│ • "Heart attack" ≠ "myocardial infarction"         │
│ • Conceptual: "things to do when bored"            │
│ • Ambiguous: "Python" (snake or language?)         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ When SEMANTIC Search Fails                          │
├─────────────────────────────────────────────────────┤
│ • Product SKUs, error codes: ERR-4032              │
│ • Domain jargon without fine-tuning                │
│ • Precision-critical (legal, compliance)           │
│ • Numerical and date filtering needs               │
└─────────────────────────────────────────────────────┘

         ↓
    SOLUTION: Use Hybrid Search
```

### Score Normalization Methods

```python
# 1. Arithmetic Mean (Equal weight)
def arithmetic_mean_normalize(bm25_score, knn_score):
    return (bm25_score + knn_score) / 2

# 2. Weighted Sum (Tunable)
def weighted_sum_normalize(bm25_score, knn_score, semantic_weight=0.6):
    keyword_weight = 1 - semantic_weight
    return (keyword_weight * bm25_score) + (semantic_weight * knn_score)

# 3. Harmonic Mean (Penalizes discrepancies)
def harmonic_mean_normalize(bm25_score, knn_score):
    if bm25_score == 0 or knn_score == 0:
        return 0
    return 2 * (bm25_score * knn_score) / (bm25_score + knn_score)
```

### Chunking Strategy

```python
def chunk_document(text, chunk_size=512, overlap=50):
    """
    Optimal chunking for semantic search
    
    Args:
        text: Document text
        chunk_size: Tokens per chunk (256-512 optimal)
        overlap: Overlapping tokens (10-20% of chunk_size)
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# Best practices:
# - Chunk size: 256-512 tokens
# - Overlap: 10-20% (prevents context loss at boundaries)
# - Too large = diluted semantics
# - Too small = lost context
```

### Multi-Vector per Document

```python
# Index multiple vector representations
index_body = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "body": {"type": "text"},
            "summary": {"type": "text"},
            
            # Separate vectors for different fields
            "title_vector": {
                "type": "knn_vector",
                "dimension": 768
            },
            "body_vector": {
                "type": "knn_vector",
                "dimension": 768
            },
            "summary_vector": {
                "type": "knn_vector",
                "dimension": 768
            }
        }
    }
}

# Search across multiple vectors
multi_vector_query = {
    "query": {
        "bool": {
            "should": [
                {"knn": {"title_vector": {"vector": query_emb, "k": 10}}},
                {"knn": {"body_vector": {"vector": query_emb, "k": 10}}},
                {"knn": {"summary_vector": {"vector": query_emb, "k": 10}}}
            ]
        }
    }
}
```

### Monitoring & Metrics

```python
# Key metrics to track
metrics = {
    "quality": {
        "recall_at_k": "Are top-K results relevant?",
        "ndcg": "Ranking quality (position matters)",
        "mrr": "Mean Reciprocal Rank",
        "ctr": "Click-through rate (ultimate signal)"
    },
    "operational": {
        "latency_p50": "Median response time",
        "latency_p95": "95th percentile (tail)",
        "latency_p99": "99th percentile (worst UX)",
        "index_size": "Storage consumption",
        "segment_count": "Affects query latency",
        "ocu_utilization": "Serverless compute units",
        "embedding_latency": "Bedrock/model inference time"
    },
    "cost": {
        "cost_per_query": "$/query",
        "cost_per_relevant_result": "Best single business metric"
    }
}

# Monitoring query
def log_search_metrics(query, results, latency_ms):
    """Log metrics for analysis"""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'num_results': len(results),
        'latency_ms': latency_ms,
        'top_score': results[0]['score'] if results else 0,
        'avg_score': sum(r['score'] for r in results) / len(results) if results else 0
    }
    # Send to CloudWatch, Elasticsearch, or your monitoring system
    return metrics
```

### Decision Framework

```
Step 1: Define Requirements
┌─────────────────────────────────────────────────┐
│ Latency SLA:  P99 < ____ ms                     │
│ Recall Target: Recall@10 > ____%                │
│ Cost Budget:   $____ per month                  │
└─────────────────────────────────────────────────┘

Step 2: Choose Architecture
┌─────────────────────────────────────────────────┐
│ E-commerce:                                     │
│   Low latency + High recall → In-memory HNSW   │
│   Pre-filtering, aggressive caching             │
├─────────────────────────────────────────────────┤
│ Internal Knowledge Base:                        │
│   Moderate latency OK → Optimize cost           │
│   Disk-based, sparse encoding, OCU autoscale    │
├─────────────────────────────────────────────────┤
│ Legal/Compliance:                               │
│   Maximum recall → Accept higher cost           │
│   Exact k-NN, hybrid search, reranking          │
├─────────────────────────────────────────────────┤
│ Chatbot/RAG:                                    │
│   Balance all three → Hybrid + caching          │
│   Fine-tuned model, quantized, tiered storage   │
└─────────────────────────────────────────────────┘
```

---

## Advanced Topics

### Sparse Similarity Scoring

```
Document 1: "Apple products are expensive"
Document 2: "An apple a day keeps doctor away"
Query: "apple headphones"

Sparse Vector Representation:
─────────────────────────────
D1: {
  "apple": 7.45,
  "products": 4.32,
  "expensive": 3.21,
  "gadget": 2.10      # Expanded term
}

D2: {
  "apple": 5.56,
  "day": 2.34,
  "doctor": 2.12
}

Q: {
  "apple": 7.89,
  "headphones": 6.54
}

Scoring Formula:
score(D, Q) = Σ(w_term_D × w_term_Q)

score(D1, Q) = 7.45 × 7.89 = 58.78 + ...  → 7.45 total
score(D2, Q) = 5.56 × 7.89 = 43.87 + ...  → 5.56 total

Result: D1 > D2 (D1 is more relevant)
```

### Agentic Search Flow

```
User Query: "Find me papers on quantum computing
             published in the last year at top conferences"
                        ↓
┌────────────────────────────────────────────────┐
│           Agentic Search Agent                 │
│                                                │
│  1. Parse intent:                              │
│     - Topic: quantum computing                 │
│     - Time: last year                          │
│     - Source: top conferences                  │
│                                                │
│  2. Plan search strategy:                      │
│     - Use semantic search for topic            │
│     - Filter by date range                     │
│     - Filter by venue prestige                 │
│                                                │
│  3. Execute multi-step search:                 │
│     a) Semantic search for "quantum computing" │
│     b) Filter: publication_date >= 2025-01-01  │
│     c) Filter: venue IN (ICML, NeurIPS, ...)  │
│     d) Re-rank by citation count               │
│                                                │
│  4. Reason about results:                      │
│     - Check coverage of sub-topics             │
│     - Identify gaps                            │
│     - Suggest related queries                  │
│                                                │
│  5. Remember context for follow-ups            │
│                                                │
└────────────────────────────────────────────────┘
                        ↓
         Ranked, contextualized results
```

### OpenSearch Serverless Architecture

```
┌─────────────────────────────────────────────────────┐
│         OpenSearch Serverless Collection            │
│                                                     │
│  ┌──────────────┐          ┌──────────────┐        │
│  │   Indexing   │          │    Search    │        │
│  │   Compute    │          │   Compute    │        │
│  │              │          │              │        │
│  │  Auto-scale  │          │  Auto-scale  │        │
│  └──────────────┘          └──────────────┘        │
│         ↓                          ↓                │
│  ┌─────────────────────────────────────────────┐   │
│  │         Storage Layer (Amazon S3)           │   │
│  │  • Index data                               │   │
│  │  • Vector data                              │   │
│  │  • Automatic replication                    │   │
│  │  • 100TB time-series collections support    │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Features:                                          │
│  ✓ No cluster management                            │
│  ✓ Automatic scaling                                │
│  ✓ Pay per use (OCU - OpenSearch Compute Units)    │
│  ✓ Enhanced resilience with S3                     │
│  ✓ Snapshot & restore                              │
└─────────────────────────────────────────────────────┘
```

---

## Complete Example: Movie Search Application

### Project Structure
```
movie-search/
├── requirements.txt
├── config.py
├── embeddings.py
├── indexer.py
├── search.py
└── app.py
```

See the implementation files for complete, production-ready code examples.

---

## Resources

### Official Documentation
- [OpenSearch Documentation](https://opensearch.org/docs/)
- [Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)

### Workshop Resources
- Workshop URL: https://tinyurl.com/ai-search-513
- GitHub: [vector-engine-demos](https://github.com/aws-samples/)

### Key Contacts
- Prashant Agrawal: prashagr@amazon.com
- Srinivas Margasahayam

---

## Summary

### Key Takeaways

1. **Search Evolution**: Keyword → Semantic → Hybrid → Agentic
2. **Vector Databases**: Enable semantic search at scale
3. **OpenSearch**: Complete platform for production search
4. **Optimization**: Balance cost, latency, and recall
5. **Hybrid Search**: Best approach for most use cases
6. **Monitoring**: Track P95/P99 latencies and recall metrics

### Quick Start Checklist

```
☐ Set up AWS account with OpenSearch Serverless
☐ Enable Amazon Bedrock with Titan embeddings
☐ Create OpenSearch collection
☐ Define index with knn_vector field
☐ Ingest data with embeddings
☐ Implement search queries
☐ Add filters and hybrid search
☐ Optimize for cost and performance
☐ Monitor and iterate
```

---

*Built with AWS OpenSearch and Amazon Bedrock*
