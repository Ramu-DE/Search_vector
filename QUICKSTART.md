# Quick Start Guide

Get your AI-powered search application running in 5 minutes!

## Prerequisites

- Python 3.8+
- AWS account (optional, for production)

## Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Local Testing (No AWS Required)

### Step 1: Generate Visualizations

```bash
# Create visual diagrams to understand concepts
python visualizations.py
```

This creates:
- `vector_basics.png` - Understanding vectors
- `similarity_metrics.png` - Distance calculations
- `knn_process.png` - k-NN algorithm visualization
- `search_evolution.png` - Search method comparison
- `performance_tradeoffs.png` - Optimization strategies
- `embedding_space.png` - How embeddings cluster

### Step 2: Test Embeddings (Local)

```bash
# Test local embedding generation
python embeddings.py
```

**Expected Output:**
```
Loading local model: all-MiniLM-L6-v2
Loaded model with dimension: 384
Text: This is a test sentence for embedding generation
Embedding dimension: 384
First 10 values: [0.048, -0.023, 0.156, ...]
```

## With AWS OpenSearch

### Step 1: Set Environment Variable

```bash
# Set your OpenSearch endpoint
export AOSS_VECTORSEARCH_ENDPOINT=https://your-endpoint.aoss.amazonaws.com
export AWS_REGION=us-west-2
```

### Step 2: Create Index and Load Data

```bash
# This will:
# - Create the index
# - Generate embeddings for sample movies
# - Index 10 sample movies
python indexer.py
```

### Step 3: Test Search

```bash
# Interactive search demo
python search.py
```

Try queries like:
- "movie to watch with friends"
- "uplifting underdog story"
- "crime thriller with plot twists"

### Step 4: Launch Web UI

```bash
# Start Streamlit app
streamlit run app.py
```

Open browser to: http://localhost:8501

## Latest Models (2024-2026)

This project uses the **latest and most performant models**:

- **Cloud**: Amazon Titan Embed Text v2 (1024 dimensions)
- **Local**: all-mpnet-base-v2 (768 dimensions) - Best quality

See **LATEST_MODELS.md** for full comparison and alternatives.

## Example Usage

### Python API

```python
from search import VectorSearchEngine

# Initialize (uses latest models by default)
search = VectorSearchEngine()

# Semantic search
results = search.semantic_search("action thriller", k=5)

# With filters
results = search.semantic_search(
    query="sci-fi movie",
    k=5,
    filters={"rating": {"gte": 8.0}}
)

# Hybrid search (best for most cases)
results = search.hybrid_search(
    query="funny comedy",
    k=5,
    semantic_weight=0.7  # 70% semantic, 30% keyword
)

# Print results
search.print_results(results)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Your Application                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. TEXT INPUT                                          │
│     "action thriller movie"                             │
│          ↓                                              │
│  2. EMBEDDING MODEL                                     │
│     [0.23, -0.45, 0.67, ...]                           │
│          ↓                                              │
│  3. OPENSEARCH k-NN                                     │
│     Vector similarity search                            │
│          ↓                                              │
│  4. RESULTS                                             │
│     • The Dark Knight (9.0)                             │
│     • The Matrix (8.7)                                  │
│     • Inception (8.8)                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── README.md              # Comprehensive documentation
├── TUTORIAL.md            # Step-by-step tutorial
├── QUICKSTART.md          # This file
├── requirements.txt       # Python dependencies
│
├── config.py              # Configuration settings
├── embeddings.py          # Embedding generation
├── indexer.py             # Index creation and data loading
├── search.py              # Search implementation
├── app.py                 # Streamlit web UI
├── visualizations.py      # Generate diagrams
│
└── *.png                  # Generated visualization files
```

## Common Commands

```bash
# Test embedding generation
python embeddings.py

# Create index and load data
python indexer.py

# Interactive search
python search.py

# Launch web UI
streamlit run app.py

# Generate visualizations
python visualizations.py

# Run tests
python -m pytest tests/
```

## Configuration

Edit `config.py` to customize:

```python
# Embedding settings (Latest Models)
LOCAL_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # Best quality (768d)
# Fast alternatives:
# 'sentence-transformers/all-MiniLM-L12-v2'  # Balanced (384d)
# 'sentence-transformers/all-MiniLM-L6-v2'   # Fastest (384d)

# Bedrock (Cloud)
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2'  # Latest (1024d)

# Search settings
DEFAULT_K = 10                           # Number of results
SEMANTIC_WEIGHT = 0.6                    # Hybrid search weight

# HNSW parameters
HNSW_M = 16                              # Graph connectivity
HNSW_EF_CONSTRUCTION = 256               # Index build quality
HNSW_EF_SEARCH = 100                     # Query-time accuracy
```

## Performance Tips

### For Speed
```python
# Lower ef_search
HNSW_EF_SEARCH = 50

# Use smaller model
LOCAL_MODEL_NAME = 'all-MiniLM-L6-v2'  # 384 dim

# Enable caching
@lru_cache(maxsize=100)
def cached_search(query):
    return search.semantic_search(query)
```

### For Accuracy
```python
# Higher ef_search
HNSW_EF_SEARCH = 200

# Better model
LOCAL_MODEL_NAME = 'all-mpnet-base-v2'  # 768 dim

# Use hybrid search
results = search.hybrid_search(query, semantic_weight=0.7)
```

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "OPENSEARCH_ENDPOINT not set"
```bash
export AOSS_VECTORSEARCH_ENDPOINT=https://your-endpoint.aoss.amazonaws.com
```

### "Model download slow"
First time downloading the model takes 1-2 minutes. It's cached after that.

### "No results found"
Make sure you've indexed data:
```bash
python indexer.py
```

## Next Steps

1. ✅ Read the full [README.md](README.md) for concepts
2. ✅ Follow [TUTORIAL.md](TUTORIAL.md) for detailed guide
3. ✅ Experiment with different queries
4. ✅ Try different embedding models
5. ✅ Deploy to production with AWS

## Example Queries to Try

**General:**
- "movie to watch with friends"
- "something uplifting and inspirational"
- "best thriller with plot twists"

**Specific:**
- "sci-fi action adventure in space"
- "emotional drama about family"
- "funny comedy for date night"

**Semantic Understanding:**
- "feel-good movie" (understands emotion)
- "mind-bending narrative" (understands style)
- "underdog story" (understands theme)

## Support

- 📖 Full Documentation: [README.md](README.md)
- 📚 Tutorial: [TUTORIAL.md](TUTORIAL.md)
- 🐛 Issues: Check troubleshooting section
- 💬 Community: OpenSearch forums

---

**Ready to build something amazing? Start now! 🚀**
