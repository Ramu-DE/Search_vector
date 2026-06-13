# Complete Edge Cases and Scenarios Guide

## 📚 Table of Contents

1. [Embedding Models - Deep Dive](#embedding-models-deep-dive)
2. [Vector Database Internals](#vector-database-internals)
3. [Search Edge Cases](#search-edge-cases)
4. [Production Scenarios](#production-scenarios)
5. [Error Handling](#error-handling)
6. [Performance Edge Cases](#performance-edge-cases)
7. [Data Quality Issues](#data-quality-issues)
8. [Security Considerations](#security-considerations)

---

## Embedding Models - Deep Dive

### What Exactly is an Embedding?

An embedding is a **learned representation** of data in a continuous vector space where semantic similarity is captured by geometric proximity.

```
Text: "The cat sat on the mat"
        ↓ (Embedding Model)
Vector: [0.234, -0.567, 0.891, ..., 0.123]  # 384-1536 dimensions

Key Properties:
1. Each dimension captures some semantic feature
2. Similar texts → Similar vectors (small distance)
3. Dissimilar texts → Dissimilar vectors (large distance)
4. Mathematical operations are meaningful
```

### How Are Embeddings Created?

#### Step 1: Tokenization
```
Text: "The cat sat on the mat"
  ↓
Tokens: ["The", "cat", "sat", "on", "the", "mat"]
  ↓
Token IDs: [101, 2054, 4938, 2006, 1996, 4303, 102]
```

#### Step 2: Token Embeddings
```
Each token → Dense vector (e.g., 768 dimensions)

"cat" → [-0.2, 0.5, 0.1, ..., 0.7]
"dog" → [-0.1, 0.4, 0.2, ..., 0.6]  # Similar to "cat"
"car" → [0.8, -0.3, 0.6, ..., 0.1]  # Different from "cat"
```

#### Step 3: Contextual Processing (Transformer)
```
Self-Attention Mechanism:
- Each token "looks at" every other token
- Context influences meaning
- "bank" in "river bank" ≠ "bank" in "money bank"

Example:
"Apple announced new iPhone"
  ↓
"Apple" embedding influenced by:
  - "announced" (corporate action)
  - "iPhone" (product)
  → Meaning: Company, not fruit
```

#### Step 4: Pooling to Sentence Embedding
```
Methods:
1. Mean Pooling (most common):
   Take average of all token embeddings
   
2. CLS Token:
   Use special [CLS] token embedding
   
3. Max Pooling:
   Take maximum value per dimension

Result: Single vector representing entire text
```

### Deep Dive: Dense vs Sparse Embeddings

#### Dense Embeddings (Traditional)

**Structure:**
```
Dimensions: 384-1536
Non-zero values: ~100%
Storage: 4 bytes × dimensions

Example (768d):
[0.234, -0.567, 0.891, 0.123, ..., -0.345]
 all values are non-zero
```

**Characteristics:**
- Every dimension is meaningful
- Captures deep semantic relationships
- Black box (hard to interpret)
- Requires vector index (HNSW, IVF)

**Example:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

text = "artificial intelligence machine learning"
embedding = model.encode(text)

print(embedding.shape)  # (768,)
print(embedding[:5])    # [0.234, -0.567, 0.891, 0.123, -0.456]
print(f"Non-zero: {(embedding != 0).sum()}")  # 768 (100%)
```

#### Sparse Embeddings (Neural Sparse)

**Structure:**
```
Dimensions: 30,522 (BERT vocab) or 105,879 (multilingual)
Non-zero values: ~1-3%
Storage: Only non-zero values

Example:
{
  "token_id_234": 0.85,    # "artificial"
  "token_id_567": 0.72,    # "ai"
  "token_id_891": 0.68,    # "intelligence"
  "token_id_1234": 0.45,   # "machine"
  "token_id_5678": 0.42    # "learning"
  # ~250 non-zero out of 30,522
}
```

**How It Works:**
```
Input: "artificial intelligence"
  ↓
1. Neural model expands with related terms:
   {
     "artificial": 0.85,
     "intelligence": 0.83,
     "ai": 0.72,              # Synonym added
     "machine": 0.45,         # Related term
     "learning": 0.40,        # Related term
     "smart": 0.38            # Expanded term
   }

2. Stored in inverted index (like BM25)
   Term → Documents containing term

3. At query time: Fast term matching + scoring
```

**OpenSearch Neural Sparse Model:**
```python
# Using OpenSearch neural-sparse-v2
query = "artificial intelligence"

sparse_vector = {
    # Explicitly mentioned terms (high weight)
    "artificial": 0.85,
    "intelligence": 0.83,
    
    # Learned synonyms (medium weight)
    "ai": 0.72,
    "smart": 0.68,
    
    # Contextual expansions (lower weight)
    "machine": 0.45,
    "learning": 0.40,
    "algorithm": 0.35
}
```

### Dense vs Sparse: Complete Comparison

#### Storage Example (1M documents, avg 50 tokens)

**Dense (768 dimensions):**
```
Storage per vector: 768 × 4 bytes = 3,072 bytes
Total: 1M × 3KB = 3GB

Plus HNSW graph:
- Edges: M × num_nodes = 16 × 1M = 16M edges
- Edge storage: ~64MB
Total: ~3.1GB
```

**Sparse (30,522 dimensions, 1% non-zero):**
```
Non-zero per vector: 305 tokens
Storage per vector: 305 × (4 + 4) bytes = 2,440 bytes
  (4 bytes: token ID, 4 bytes: weight)
Total: 1M × 2.4KB = 2.4GB

Inverted index (Lucene):
- Posting lists: ~500MB
Total: ~2.9GB

Difference: Sparse is 10x smaller in practice!
```

#### Query Time Comparison

**Dense Search:**
```
1. Generate query embedding: 5-20ms
2. HNSW traversal: 
   - ef_search=100: ~10-50ms
   - Compares ~100-1000 vectors
3. Total: 15-70ms

Parallelizable: Yes (multiple shards)
```

**Sparse Search:**
```
1. Generate sparse vector: 5-20ms (or 0ms if document-only)
2. Inverted index lookup: 1-5ms
   - Direct term matching
   - Very fast (like BM25)
3. Total: 6-25ms (or 1-5ms without query encoding)

Parallelizable: Yes (multiple shards)
```

### Edge Case: When Embeddings Fail

#### Case 1: Out-of-Vocabulary Words
```
Input: "The frobnicator needs maintenance"

Problem: "frobnicator" not in training data

Dense Embedding:
- Tokenizes to subwords: "frob", "##nicator"
- Still generates embedding (imperfect)
- May not capture exact meaning

Sparse Embedding:
- Term is preserved as-is
- Weight assigned based on context
- Better for rare/technical terms

Solution: Hybrid search (combines both)
```

#### Case 2: Domain-Specific Jargon
```
Medical: "MI" means "Myocardial Infarction"
Military: "MI" means "Military Intelligence"

Problem: Generic embeddings don't know domain

Generic Model:
query = "patient with MI"
→ May match military documents!

Solution:
1. Domain-specific fine-tuning
2. Add metadata filters (domain=medical)
3. Use hybrid search with keyword filters
```

#### Case 3: Negation
```
Query: "not a good movie"
Generic embedding may be similar to: "good movie"

Problem: Negation poorly captured

Example:
text1 = "This is a good movie"
text2 = "This is not a good movie"

similarity(embed(text1), embed(text2)) = 0.85
(Should be lower!)

Solutions:
1. Use models trained on negation (e.g., sentence-t5)
2. Implement negation detection in preprocessing
3. Use hybrid search with keyword matching
```

#### Case 4: Numerical Values
```
Query: "laptop under $500"

Problem: Embeddings don't preserve numerical semantics

embed("$500") ≈ embed("$499") ≈ embed("$5000")
(All similar because they're numbers)

Solution:
1. Extract numbers to structured fields
2. Use range filters, not semantic search
3. Combine: semantic search + numerical filters

query = {
  "knn": {"text_vector": query_embedding},
  "filter": {"price": {"lte": 500}}
}
```

#### Case 5: Code Snippets
```
Query: "function to sort array in Python"
Document: Contains actual code

Problem: Code structure ≠ natural language

Code:
def sort_array(arr):
    return sorted(arr)

Generic embedding treats it as text
Better: Use code-specific models

Solutions:
1. Use CodeBERT or CodeT5
2. Separate indices for code vs text
3. Hybrid: code syntax + semantic description
```

### Embedding Model Training (Simplified)

#### Training Data Format
```
Triplets: (anchor, positive, negative)

Example:
anchor:   "How to train a neural network?"
positive: "Guide to neural network training"
negative: "Recipe for chocolate cake"

Goal: Make anchor close to positive, far from negative
```

#### Training Process
```
1. Input triplet
2. Generate embeddings:
   anchor_vec = model(anchor)
   positive_vec = model(positive)
   negative_vec = model(negative)

3. Calculate loss (Triplet Loss):
   distance(anchor, positive) should be small
   distance(anchor, negative) should be large
   
   loss = max(0, dist(anchor, positive) - dist(anchor, negative) + margin)

4. Backpropagate and update weights
5. Repeat millions of times
```

#### Why Pre-trained Models Work

Models like all-mpnet-base-v2 trained on:
- 1 billion+ sentence pairs
- Diverse domains (web, books, papers)
- Multiple languages
- Various tasks (similarity, entailment, QA)

Result: Captures general semantic knowledge

### Vector Space Properties

#### Property 1: Semantic Clustering
```
Animals: dog, cat, bird
→ Cluster in same region of vector space

Vehicles: car, bike, plane
→ Different cluster

Distance within cluster < Distance between clusters
```

#### Property 2: Analogies
```
Vector arithmetic works!

king - man + woman ≈ queen
Paris - France + Italy ≈ Rome

How:
1. "king" embedding
2. Subtract "man" direction
3. Add "woman" direction
4. Result near "queen"
```

#### Property 3: Dimensionality Matters
```
Low dimensions (64-128):
- Fast
- Less nuanced
- Good for coarse similarity

Medium dimensions (384-768):
- Balanced
- Most common
- Good for general use

High dimensions (1024-3072):
- Slow
- Very nuanced
- Best quality

Too high (>4096):
- Overfitting risk
- Diminishing returns
- Very slow
```

---

## Vector Database Internals

### What is a Vector Database?

A vector database is a specialized database optimized for:
1. **Storing** high-dimensional vectors
2. **Indexing** vectors for fast similarity search
3. **Querying** by similarity (not exact match)

### OpenSearch as Vector Database

#### Index Structure
```
OpenSearch Index:
├── Inverted Index (for text fields)
│   ├── Term → Document IDs
│   └── Fast keyword search
│
└── Vector Index (for knn_vector fields)
    ├── HNSW Graph
    ├── Stores vectors
    └── Fast k-NN search

Both can be queried together (Hybrid Search)!
```

#### Storage Layout
```
Document:
{
  "title": "The Matrix",
  "plot": "A hacker discovers reality...",
  "title_vector": [0.234, -0.567, ..., 0.891],  # 768 dimensions
  "genre": ["Sci-Fi", "Action"]
}

On Disk:
1. Text fields → Lucene segments
2. Vectors → Separate vector segments
3. HNSW graph → Adjacency lists

Total storage per document:
- Text: ~1-10KB
- Vector: 768 × 4 bytes = 3KB
- HNSW edges: ~M × 8 bytes = 128 bytes (M=16)
- Total: ~4-13KB per document
```

### HNSW Algorithm - Deep Dive

#### What is HNSW?

**Hierarchical Navigable Small Worlds**

Key idea: Build a multi-layer graph where:
- Top layers: Few nodes, long-distance connections
- Bottom layer: All nodes, local connections

Like a highway system:
- Highways (top): Fast, long distance
- Local roads (bottom): Detailed, short distance

#### HNSW Construction

```
Step 1: Insert first vector
┌─────────────────┐
│ Layer 2:  A     │ (Entry point)
├─────────────────┤
│ Layer 1:  A     │
├─────────────────┤
│ Layer 0:  A     │
└─────────────────┘

Step 2: Insert second vector B
Choose layer randomly (exponentially decaying probability)
Say B goes to layer 1

┌─────────────────┐
│ Layer 2:  A     │
├─────────────────┤
│ Layer 1:  A─────B     │
├─────────────────┤
│ Layer 0:  A─────B     │
└─────────────────┘

Step 3: Insert third vector C (goes to layer 0)
┌─────────────────┐
│ Layer 2:  A     │
├─────────────────┤
│ Layer 1:  A─────B     │
├─────────────────┤
│ Layer 0:  A─────B     │
│           │  ╱  │     │
│           C     │     │
└─────────────────┘

Each node connects to M nearest neighbors at each layer
```

#### HNSW Parameters Explained

##### Parameter: M (Graph Connectivity)
```
M = Number of bidirectional links per node

Low M (e.g., 8):
  ┌───┐    ┌───┐    ┌───┐
  │ A │────│ B │────│ C │
  └───┘    └───┘    └───┘
  Few connections → Fast build, lower recall

High M (e.g., 32):
  ┌───┐    ┌───┐    ┌───┐
  │ A │────│ B │────│ C │
  └───┘╲  ╱└───┘╲  ╱└───┘
       ╲╱        ╲╱
       ╱╲        ╱╲
      ╱  ╲      ╱  ╲
  More connections → Slower build, higher recall

Trade-off:
- Memory: O(M × N) edges
- Build time: O(M × log N) per insertion
- Search quality: Higher M → Better recall
- Search speed: Higher M → More candidates to check

Recommendation:
- M=8: Fast build, memory-constrained
- M=16: Balanced (default)
- M=32: High recall, production quality
```

##### Parameter: ef_construction (Build Quality)
```
ef_construction = Size of dynamic candidate list during construction

Low ef_construction (e.g., 128):
  Build process:
  1. Find nearest neighbors quickly
  2. May miss some good connections
  3. Faster build, lower quality graph
  
High ef_construction (e.g., 512):
  Build process:
  1. Explore more candidates
  2. Find better connections
  3. Slower build, higher quality graph

Example (inserting new node):
ef_construction=128:
  Check ~128 nodes → Pick M=16 best → Connect
  Time: Fast

ef_construction=512:
  Check ~512 nodes → Pick M=16 best → Connect
  Time: 4x slower, but better quality

Trade-off:
- Only affects BUILD time, not SEARCH time
- Higher → Better graph → Better search recall
- Once built, quality is locked in

Recommendation:
- ef_construction=128: Fast builds, development
- ef_construction=256: Balanced (default)
- ef_construction=512: Production quality
```

##### Parameter: ef_search (Query Accuracy)
```
ef_search = Size of dynamic candidate list during SEARCH

This is the KEY tuning parameter for queries!

Low ef_search (e.g., 50):
  Search process:
  1. Start at entry point
  2. Maintain queue of 50 candidates
  3. Return top K
  
  Result: Fast, lower recall

High ef_search (e.g., 200):
  Search process:
  1. Start at entry point
  2. Maintain queue of 200 candidates
  3. Return top K
  
  Result: Slower, higher recall

Example (searching for K=10 results):
ef_search=50:
  Explores ~50-100 nodes
  Returns top 10
  Recall: ~92-95%
  Latency: 10ms

ef_search=100:
  Explores ~100-200 nodes
  Returns top 10
  Recall: ~96-98%
  Latency: 20ms

ef_search=200:
  Explores ~200-400 nodes
  Returns top 10
  Recall: ~98-99%
  Latency: 40ms

Trade-off:
- Runtime tunable (can change per query!)
- Higher → Better recall, higher latency
- Linear relationship: 2x ef_search ≈ 2x latency

Recommendation:
- Start at 100
- If recall too low: increase to 200
- If latency too high: decrease to 50
- Monitor P95/P99 latencies
```

#### HNSW Search Algorithm

```
function HNSW_SEARCH(query, K, ef_search):
    ep = entry_point  # Start at top layer
    
    # Phase 1: Navigate down layers (greedy)
    for layer in [max_layer, max_layer-1, ..., 1]:
        ep = SEARCH_LAYER(query, ep, ef=1, layer)
        # Move to closest neighbor at each layer
    
    # Phase 2: Search bottom layer (beam search)
    W = SEARCH_LAYER(query, ep, ef=ef_search, layer=0)
    
    # Phase 3: Return top K
    return top_K(W)

function SEARCH_LAYER(query, entry_points, ef, layer):
    visited = set()
    candidates = priority_queue()  # Min heap
    results = priority_queue()     # Max heap
    
    for ep in entry_points:
        dist = distance(query, ep)
        candidates.push(ep, dist)
        results.push(ep, dist)
        visited.add(ep)
    
    while candidates.not_empty():
        c = candidates.pop()  # Closest candidate
        f = results.top()     # Farthest result
        
        if distance(c, query) > distance(f, query):
            break  # All remaining candidates are farther
        
        for neighbor in neighbors(c, layer):
            if neighbor not in visited:
                visited.add(neighbor)
                f = results.top()
                dist = distance(neighbor, query)
                
                if dist < distance(f, query) or results.size() < ef:
                    candidates.push(neighbor, dist)
                    results.push(neighbor, dist)
                    
                    if results.size() > ef:
                        results.pop()  # Remove farthest
    
    return results

Complexity:
- Time: O(ef_search × log(ef_search) × avg_degree)
- Space: O(ef_search)
- In practice: Sub-linear in dataset size!
```

#### Exact k-NN vs HNSW

```
Exact k-NN (Brute Force):
  for each vector in database:
      distance = calculate(query, vector)
      if distance in top K:
          add to results
  
  Time: O(N × D)
    N = number of vectors
    D = dimensions
  
  Example: 1M vectors, 768D
  Comparisons: 1,000,000
  Time: ~500ms - 5 seconds

HNSW:
  Follow graph structure
  Check only ~ef_search vectors
  
  Time: O(log(N) × D × ef_search)
  
  Example: 1M vectors, 768D, ef_search=100
  Comparisons: ~100-200
  Time: ~10-50ms
  
  Speed-up: 10-100x faster!
```

### Vector Similarity Metrics - Deep Dive

#### 1. Cosine Similarity

**Formula:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
A · B = Dot product = Σ(ai × bi)
||A|| = Magnitude = √(Σ(ai²))
```

**Properties:**
```
Range: [-1, 1]
  1.0 = Identical direction
  0.0 = Perpendicular (no similarity)
 -1.0 = Opposite direction

Magnitude invariant: Only angle matters
  [1, 1] and [2, 2] have cosine similarity = 1.0
  (Same direction, different magnitude)
```

**Best For:**
```
✓ Text similarity (most common)
✓ Document retrieval
✓ Sentence embeddings
✓ When magnitude doesn't matter

✗ When magnitude is meaningful
✗ Sparse vectors with many zeros
```

**Example:**
```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([2, 4, 6])  # 2× A

# Cosine similarity
dot_product = np.dot(A, B)  # 28
norm_A = np.linalg.norm(A)  # 3.74
norm_B = np.linalg.norm(B)  # 7.48
cosine_sim = dot_product / (norm_A * norm_B)  # 1.0 (perfect!)

# Even though magnitudes different, direction same
```

**OpenSearch Config:**
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "space_type": "cosinesimil",  // or "cosine"
    "engine": "lucene"
  }
}
```

#### 2. Euclidean Distance (L2)

**Formula:**
```
euclidean_distance(A, B) = √(Σ(ai - bi)²)

Example:
A = [1, 2, 3]
B = [4, 5, 6]

distance = √((1-4)² + (2-5)² + (3-6)²)
         = √(9 + 9 + 9)
         = √27
         = 5.196
```

**Properties:**
```
Range: [0, ∞)
  0 = Identical vectors
  Large = Very different

Magnitude sensitive:
  [1, 1] and [2, 2] have distance = √2 ≈ 1.414
  (Not identical even though same direction!)
```

**Best For:**
```
✓ When magnitude matters (e.g., counts, measurements)
✓ Physical distance in space
✓ Image embeddings
✓ Recommendation systems

✗ Text when you want scale-invariance
```

**Example:**
```python
A = np.array([1, 2, 3])
B = np.array([2, 4, 6])

euclidean = np.linalg.norm(A - B)  # 3.74
# Different from cosine! Magnitude matters
```

**OpenSearch Config:**
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "space_type": "l2",
    "engine": "lucene"
  }
}
```

#### 3. Dot Product

**Formula:**
```
dot_product(A, B) = Σ(ai × bi)

Example:
A = [1, 2, 3]
B = [4, 5, 6]

dot = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

**Properties:**
```
Range: [-∞, ∞]
  High positive = Similar and large magnitude
  Near 0 = Orthogonal or small magnitude
  Negative = Opposite direction

Magnitude amplifies:
  [1, 1] · [1, 1] = 2
  [2, 2] · [2, 2] = 8 (4× larger!)
```

**Best For:**
```
✓ Collaborative filtering
✓ When both direction AND magnitude matter
✓ Pre-normalized vectors
✓ Multi-modal embeddings

Note: If vectors are normalized (||v|| = 1),
      dot product = cosine similarity!
```

**Example:**
```python
A = np.array([1, 2, 3])
B = np.array([2, 4, 6])

dot = np.dot(A, B)  # 28

# Normalize vectors
A_norm = A / np.linalg.norm(A)
B_norm = B / np.linalg.norm(B)

dot_normalized = np.dot(A_norm, B_norm)  # 1.0
# Same as cosine similarity!
```

**OpenSearch Config:**
```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "space_type": "innerproduct",  // Dot product
    "engine": "lucene"
  }
}
```

#### Metric Comparison Example

```python
from scipy.spatial.distance import cosine, euclidean

# Three movie descriptions
movie1 = np.array([0.8, 0.6, 0.2])  # Action movie
movie2 = np.array([0.7, 0.5, 0.3])  # Similar action movie
movie3 = np.array([0.2, 0.3, 0.9])  # Romance movie

query = np.array([0.9, 0.7, 0.1])   # Query: "action thriller"

# Cosine Similarity
cosine_1 = 1 - cosine(query, movie1)  # 0.997 (very similar)
cosine_2 = 1 - cosine(query, movie2)  # 0.993 (very similar)
cosine_3 = 1 - cosine(query, movie3)  # 0.612 (different)

# Euclidean Distance
euclidean_1 = euclidean(query, movie1)  # 0.173 (close)
euclidean_2 = euclidean(query, movie2)  # 0.374 (farther)
euclidean_3 = euclidean(query, movie3)  # 1.249 (far)

# Dot Product
dot_1 = np.dot(query, movie1)  # 1.29 (high)
dot_2 = np.dot(query, movie2)  # 1.11 (medium)
dot_3 = np.dot(query, movie3)  # 0.48 (low)

# Ranking (best match first)
# All metrics agree: movie1 > movie2 > movie3
```

### Quantization - Memory Reduction

#### Full Precision (FP32)
```
Original: 32 bits per value
Vector: [0.234567, -0.567890, 0.891234, ...]
Storage: 768 × 4 bytes = 3,072 bytes

Precision: Very high (~7 decimal places)
```

#### Half Precision (FP16)
```
Reduced: 16 bits per value
Vector: [0.2346, -0.5679, 0.8912, ...]
Storage: 768 × 2 bytes = 1,536 bytes

Savings: 50%
Precision loss: Minimal (<0.5% recall loss)
```

#### 8-bit Integer (INT8)
```
Reduced: 8 bits per value
Quantization: Map float range to [-128, 127]

Process:
1. Find min/max: [-1.0, 1.0]
2. Scale: value → (value × 127)
3. Round to integer

Vector: [30, -72, 113, ...]
Storage: 768 × 1 byte = 768 bytes

Savings: 75%
Precision loss: Small (~1% recall loss)
```

#### Binary Quantization
```
Extreme: 1 bit per value
Rule: value ≥ 0 → 1, value < 0 → 0

Original: [0.234, -0.567, 0.891, -0.123, ...]
Binary:   [1, 0, 1, 0, ...]
Storage: 768 / 8 = 96 bytes

Savings: 97%!

Search Process:
1. Initial search with binary (fast)
2. Re-score top candidates with full precision
3. Return refined results

Precision loss: <2% with rescoring
```

**Quantization Example:**
```python
import numpy as np

# Original vector (FP32)
original = np.random.randn(768).astype(np.float32)
print(f"FP32 size: {original.nbytes} bytes")  # 3072 bytes

# FP16 quantization
fp16 = original.astype(np.float16)
print(f"FP16 size: {fp16.nbytes} bytes")  # 1536 bytes

# INT8 quantization
scale = 127.0 / np.max(np.abs(original))
int8 = np.round(original * scale).astype(np.int8)
print(f"INT8 size: {int8.nbytes} bytes")  # 768 bytes

# Binary quantization
binary = (original >= 0).astype(np.uint8)
packed = np.packbits(binary)
print(f"Binary size: {packed.nbytes} bytes")  # 96 bytes

# Calculate similarity preservation
original_norm = original / np.linalg.norm(original)
fp16_norm = fp16.astype(np.float32) / np.linalg.norm(fp16)

similarity = np.dot(original_norm, fp16_norm)
print(f"FP32 vs FP16 similarity: {similarity:.6f}")  # ~0.999998
```

---

## Search Edge Cases

### Case 1: Empty Results

**Scenario:**
```python
query = "quantum blockchain AI metaverse"
results = search.semantic_search(query, k=10)
# Results: [] (empty!)
```

**Why:**
- No documents match filters
- Query too specific
- Index empty or wrong index

**Handling:**
```python
def safe_search(query, k=10, filters=None):
    results = search.semantic_search(query, k, filters)
    
    if not results:
        # Try without filters
        results = search.semantic_search(query, k)
        
        if not results:
            # Try keyword search
            results = search.keyword_search(query, k)
            
            if not results:
                # Try broader query
                broader_query = extract_key_terms(query)
                results = search.semantic_search(broader_query, k)
    
    return results or []
```

### Case 2: All Results Have Low Scores

**Scenario:**
```python
results = [
    {"title": "Movie A", "_score": 0.23},
    {"title": "Movie B", "_score": 0.21},
    {"title": "Movie C", "_score": 0.19}
]
# All scores < 0.3 (low confidence!)
```

**Why:**
- Query doesn't match corpus well
- Poor query phrasing
- Wrong semantic domain

**Handling:**
```python
def filter_by_confidence(results, threshold=0.5):
    confident = [r for r in results if r['_score'] >= threshold]
    
    if not confident:
        return {
            "results": results[:3],  # Return top 3 anyway
            "warning": "Low confidence matches. Try rephrasing."
        }
    
    return {"results": confident}
```

### Case 3: Duplicate Results

**Scenario:**
```python
results = [
    {"title": "The Matrix", "_score": 0.95, "_id": "1"},
    {"title": "The Matrix", "_score": 0.94, "_id": "2"},  # Duplicate!
    {"title": "The Matrix Reloaded", "_score": 0.88}
]
```

**Why:**
- Same movie indexed multiple times
- Near-duplicates with slight variations
- Different editions/versions

**Handling:**
```python
def deduplicate_results(results, key='title', similarity_threshold=0.9):
    seen = {}
    deduped = []
    
    for result in results:
        title = result[key].lower().strip()
        
        # Check if similar title already seen
        is_duplicate = False
        for seen_title in seen:
            if similarity(title, seen_title) > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen[title] = True
            deduped.append(result)
    
    return deduped
```

### Case 4: Filter Conflicts

**Scenario:**
```python
# Contradictory filters
filters = {
    "rating": {"gte": 9.0, "lte": 7.0}  # Impossible!
}

results = search.semantic_search(query, filters=filters)
# Results: [] (no movie has rating >= 9 AND <= 7)
```

**Handling:**
```python
def validate_filters(filters):
    errors = []
    
    for field, condition in filters.items():
        if isinstance(condition, dict):
            if 'gte' in condition and 'lte' in condition:
                if condition['gte'] > condition['lte']:
                    errors.append(f"{field}: gte > lte")
    
    if errors:
        raise ValueError(f"Invalid filters: {errors}")
    
    return filters
```

### Case 5: Very Long Queries

**Scenario:**
```python
query = """I'm looking for a movie that has action scenes with great 
cinematography and involves a complex plot about time travel with 
multiple timelines and features a strong female lead character who 
is a scientist and was released in the last 5 years..."""  # 500+ words!
```

**Why this is a problem:**
- Most models have max token limit (512 tokens)
- Long queries dilute important terms
- Slower embedding generation

**Handling:**
```python
def handle_long_query(query, max_tokens=512):
    tokens = tokenize(query)
    
    if len(tokens) <= max_tokens:
        return query
    
    # Strategy 1: Truncate
    truncated = detokenize(tokens[:max_tokens])
    
    # Strategy 2: Extract key phrases
    key_phrases = extract_key_phrases(query)
    concise_query = " ".join(key_phrases[:10])
    
    # Strategy 3: Split into multiple searches
    chunks = chunk_query(query, max_tokens)
    results = []
    for chunk in chunks:
        results.extend(search.semantic_search(chunk, k=5))
    
    # Deduplicate and re-rank
    return deduplicate_and_rerank(results)
```

### Case 6: Multilingual Queries

**Scenario:**
```python
# English index, Spanish query
query = "película de acción con explosiones"
results = search.semantic_search(query, k=10)
# Results: Poor matches (if using English-only model)
```

**Handling:**
```python
def multilingual_search(query, k=10):
    # Detect language
    lang = detect_language(query)
    
    if lang != 'en':
        # Option 1: Translate query
        query_en = translate(query, target='en')
        results = search.semantic_search(query_en, k)
        
        # Option 2: Use multilingual model
        multilingual_gen = LocalEmbedding(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        results = search.semantic_search(
            query, k, 
            embedding_generator=multilingual_gen
        )
    
    return results
```

### Case 7: Ambiguous Queries

**Scenario:**
```python
query = "matrix"
# Could mean:
# 1. The movie "The Matrix"
# 2. Mathematical matrix
# 3. Organizational matrix
```

**Handling:**
```python
def handle_ambiguous_query(query, k=10):
    # Get diverse results
    results = search.semantic_search(query, k=k*3)
    
    # Cluster results by topic
    clusters = cluster_by_topic(results)
    
    # Return top results from each cluster
    diverse_results = []
    for cluster in clusters:
        diverse_results.extend(cluster[:k//len(clusters)])
    
    return {
        "results": diverse_results,
        "clusters": [c['topic'] for c in clusters],
        "suggestion": "Did you mean one of these topics?"
    }
```

### Case 8: Temporal Queries

**Scenario:**
```python
query = "recent action movies"
# "recent" is relative!
```

**Handling:**
```python
from datetime import datetime, timedelta

def temporal_search(query, k=10):
    # Extract temporal keywords
    temporal_terms = {
        'recent': timedelta(days=365),
        'latest': timedelta(days=180),
        'new': timedelta(days=90),
        'old': timedelta(days=-365*10),
        'classic': timedelta(days=-365*20)
    }
    
    # Find temporal term in query
    time_filter = None
    for term, delta in temporal_terms.items():
        if term in query.lower():
            cutoff_date = datetime.now() + delta
            if delta.days > 0:
                time_filter = {"gte": cutoff_date.year}
            else:
                time_filter = {"lte": cutoff_date.year}
            query = query.replace(term, '')  # Remove from query
            break
    
    filters = {}
    if time_filter:
        filters['year'] = time_filter
    
    return search.semantic_search(query.strip(), k, filters)
```

---

## Production Scenarios

### Scenario 1: Cold Start (Empty Index)

**Problem:** New system with no data

**Solution:**
```python
class SearchSystemWithColdStart:
    def __init__(self):
        self.search = VectorSearchEngine()
        self.cold_start = True
        self.fallback_results = load_popular_items()
    
    def search(self, query, k=10):
        results = self.search.semantic_search(query, k)
        
        if not results and self.cold_start:
            # Return popular items
            return {
                "results": self.fallback_results[:k],
                "cold_start": True,
                "message": "Showing popular items"
            }
        
        if len(results) >= 100:
            self.cold_start = False  # Enough data now
        
        return {"results": results}
```

### Scenario 2: Index Refresh

**Problem:** Need to update index without downtime

**Solution:**
```python
class ZeroDowntimeIndexRefresh:
    def __init__(self):
        self.active_index = "movies_v1"
        self.standby_index = "movies_v2"
    
    def refresh_index(self, new_documents):
        # Step 1: Build new index in background
        print(f"Building {self.standby_index}...")
        indexer = OpenSearchIndexer(index_name=self.standby_index)
        indexer.create_index(delete_if_exists=True)
        indexer.index_documents(new_documents)
        
        # Step 2: Verify new index
        test_queries = ["action movie", "romance", "thriller"]
        for query in test_queries:
            results = search_index(query, self.standby_index)
            assert len(results) > 0, f"No results for: {query}"
        
        # Step 3: Atomic swap (using alias)
        swap_alias("movies", self.standby_index)
        
        # Step 4: Cleanup old index
        delete_index(self.active_index)
        
        # Step 5: Swap roles
        self.active_index, self.standby_index = \
            self.standby_index, self.active_index
        
        print("Index refresh complete!")
```

### Scenario 3: Rate Limiting

**Problem:** Too many concurrent searches

**Solution:**
```python
from threading import Semaphore
import time

class RateLimitedSearch:
    def __init__(self, max_concurrent=10, requests_per_second=100):
        self.search = VectorSearchEngine()
        self.semaphore = Semaphore(max_concurrent)
        self.requests_per_second = requests_per_second
        self.request_times = []
    
    def search(self, query, k=10):
        # Check rate limit
        now = time.time()
        self.request_times = [t for t in self.request_times 
                             if now - t < 1.0]
        
        if len(self.request_times) >= self.requests_per_second:
            raise RateLimitError("Too many requests per second")
        
        self.request_times.append(now)
        
        # Limit concurrent searches
        with self.semaphore:
            return self.search.semantic_search(query, k)
```

### Scenario 4: Gradual Rollout (A/B Testing)

**Problem:** Want to test new embedding model

**Solution:**
```python
import random

class ABTestingSearch:
    def __init__(self):
        self.search_a = VectorSearchEngine()  # Old model
        self.search_b = VectorSearchEngine(
            embedding_generator=new_model
        )
        self.ab_percentage = 10  # 10% to new model
    
    def search(self, query, user_id, k=10):
        # Consistent per-user assignment
        use_b = hash(user_id) % 100 < self.ab_percentage
        
        search_engine = self.search_b if use_b else self.search_a
        results = search_engine.semantic_search(query, k)
        
        # Log for analysis
        log_search(user_id, query, results, variant='B' if use_b else 'A')
        
        return results
```

### Scenario 5: Caching Strategy

**Problem:** Same queries repeated often

**Solution:**
```python
from functools import lru_cache
import hashlib
import json

class CachedSearch:
    def __init__(self, cache_size=1000, ttl=3600):
        self.search = VectorSearchEngine()
        self.cache = {}
        self.cache_size = cache_size
        self.ttl = ttl
    
    def _cache_key(self, query, k, filters):
        data = json.dumps({
            'query': query, 
            'k': k, 
            'filters': filters
        }, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
    
    def search(self, query, k=10, filters=None):
        cache_key = self._cache_key(query, k, filters)
        
        # Check cache
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['results']  # Cache hit!
        
        # Cache miss - perform search
        results = self.search.semantic_search(query, k, filters)
        
        # Store in cache
        if len(self.cache) >= self.cache_size:
            # Evict oldest entry (LRU)
            oldest_key = min(self.cache, 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        return results
```

### Scenario 6: Fallback Strategy

**Problem:** Primary search fails

**Solution:**
```python
class RobustSearch:
    def __init__(self):
        self.semantic_search = VectorSearchEngine()
        self.keyword_search = KeywordSearchEngine()
        self.fallback_results = PopularItems()
    
    def search(self, query, k=10, filters=None):
        try:
            # Try semantic search
            results = self.semantic_search.semantic_search(
                query, k, filters
            )
            if results:
                return {"results": results, "method": "semantic"}
        except Exception as e:
            log_error("Semantic search failed", e)
        
        try:
            # Fallback to keyword search
            results = self.keyword_search.search(query, k, filters)
            if results:
                return {"results": results, "method": "keyword"}
        except Exception as e:
            log_error("Keyword search failed", e)
        
        # Last resort: popular items
        return {
            "results": self.fallback_results.get(k),
            "method": "fallback"
        }
```

---

*This is part 1 of the complete guide. Continue reading...*
