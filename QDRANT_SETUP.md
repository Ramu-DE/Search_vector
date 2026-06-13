# Qdrant Vector Database Setup Guide

## Overview

This project now supports **Qdrant** as the vector database, providing a simpler alternative to OpenSearch Serverless with no AWS IAM permission requirements.

## ✅ Why Qdrant?

- **No AWS Permissions Required** - Works with your existing Bedrock access
- **Free Tier Available** - 1GB free cluster on Qdrant Cloud
- **Simple Setup** - Just URL + API key
- **Fast & Reliable** - Purpose-built for vector search
- **Long Context Compatible** - Works seamlessly with 1M context Claude models

## 🚀 Quick Setup (3 Steps)

### Step 1: Get Qdrant Credentials

**Option A: Qdrant Cloud (Free Tier)**

1. Sign up at: https://cloud.qdrant.io/
2. Create a new cluster (free 1GB tier available)
3. Copy your cluster URL (format: `https://xxxxx.qdrant.io`)
4. Generate an API key from the cluster dashboard

**Option B: Self-Hosted Qdrant**

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Your URL will be: http://localhost:6333
# No API key needed for local deployment
```

### Step 2: Configure Environment

Add to your `.env` file:

```bash
# Qdrant Configuration
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your-api-key-here

# AWS Bedrock (already configured)
AWS_REGION=us-west-2
```

### Step 3: Setup & Ingest

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install qdrant-client

# Create collection
python setup_qdrant.py

# Load movie data
python ingest_data_qdrant.py
```

## 📊 What Gets Created

The setup script creates:

- **Collection**: `movies`
- **Vector Dimension**: 1024 (matches Bedrock Titan v2)
- **Distance Metric**: Cosine similarity
- **Metadata Fields**:
  - `title` (text)
  - `plot` (text)
  - `genre` (keyword)
  - `year` (integer)
  - `rating` (float)
  - `director` (text)
  - `cast` (text)

## 🧪 Test Connection

```bash
python qdrant_client.py
```

Expected output:
```
✓ Connected to Qdrant: https://xxxxx.qdrant.io
  Collection: movies
  Dimension: 1024
```

## 💡 Usage in Your Code

```python
from qdrant_client import QdrantVectorStore
import numpy as np

# Initialize
store = QdrantVectorStore()

# Search
query_embedding = generate_embedding("space adventure")  # Your embedding function
results = store.search(
    query_embedding=np.array(query_embedding),
    k=10,
    min_rating=7.0,
    genre="Sci-Fi"
)

# Results contain title, plot, score, etc.
for result in results:
    print(f"{result['title']}: {result['score']:.3f}")
```

## 🔄 Architecture

```
User Query
    ↓
AWS Bedrock Titan v2 (1024-dim embedding)
    ↓
Qdrant Vector Search (cosine similarity)
    ↓
Filtered Results (rating, year, genre)
    ↓
Claude Opus 4.8 (1M context for summarization)
    ↓
Final Results
```

## 📈 Performance

**Qdrant Free Tier:**
- Storage: 1GB (enough for ~1M 1024-dim vectors)
- Queries: Unlimited
- Latency: <50ms for most queries

**Bedrock Embeddings:**
- Cost: $0.0001 per 1K input tokens
- Latency: ~100ms per request
- Dimension: 1024

**Claude Opus 4.8:**
- Context: 1M tokens
- Cost: $5 input / $25 output per 1M tokens

## 🔍 Comparison: Qdrant vs OpenSearch

| Feature | Qdrant Cloud | OpenSearch Serverless |
|---------|--------------|----------------------|
| **Setup** | 5 minutes | 30+ minutes |
| **IAM Permissions** | None required | Multiple policies |
| **Cost** | Free tier available | ~$700/month minimum |
| **Performance** | <50ms queries | <100ms queries |
| **Maintenance** | Managed | Managed |
| **Best For** | Development, Prototypes | Production, AWS-native |

## 🛠️ Troubleshooting

### "QDRANT_URL not set"
Add credentials to `.env` file:
```bash
QDRANT_URL=https://xxxxx.qdrant.io
QDRANT_API_KEY=your-key-here
```

### "Connection refused"
- Check cluster is running (Qdrant Cloud dashboard)
- Verify URL format (must start with `https://`)
- Test with: `curl https://your-cluster.qdrant.io/collections`

### "Collection not found"
Run setup:
```bash
python setup_qdrant.py
```

### "Dimension mismatch"
Ensure config.py has:
```python
BEDROCK_EMBEDDING_DIMENSION = 1024
```

## 📚 Sample Data

The ingestion script includes 15 highly-rated movies:
- The Shawshank Redemption (9.3)
- The Godfather (9.2)
- Interstellar (8.6)
- Inception (8.8)
- The Matrix (8.7)
- And more...

Each movie has:
- **Semantic search**: Title + plot embeddings
- **Metadata filtering**: Rating, year, genre
- **Rich results**: Director, cast, ratings

## 🎯 Next Steps

Once Qdrant is set up:

1. ✅ **Test search**: `python qdrant_client.py`
2. ✅ **Build UI**: `streamlit run app.py`
3. ✅ **Add Claude integration**: Use example_usage.py patterns
4. ✅ **Scale up**: Add more movies or your own data

## 🔗 Resources

- **Qdrant Cloud**: https://cloud.qdrant.io/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Python Client**: https://github.com/qdrant/qdrant-client
- **AWS Bedrock**: Already configured ✅
- **Claude API**: Ready for 1M context ✅

---

**Current Configuration:**
- ✅ AWS Bedrock Titan v2 (1024-dim embeddings)
- ✅ Claude Opus 4.8 (1M context, auto-fallback from Mythos 5)
- ⏳ Qdrant (waiting for your credentials)

**Total Setup Time:** ~5 minutes after you provide Qdrant credentials
