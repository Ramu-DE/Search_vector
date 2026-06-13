# Latest AI Models for Search (2024-2026)

## 🚀 Currently Active Models

This project is configured to use the **latest and most performant models** available.

---

## AWS Bedrock Models (Cloud-based)

### ✨ Currently Active: Amazon Titan Embed Text v2

**Model ID**: `amazon.titan-embed-text-v2`
- **Dimensions**: 1024
- **Max Input**: 8,192 tokens
- **Languages**: English (primary), 100+ languages
- **Release**: 2024
- **Cost**: ~$0.0001 per 1K tokens
- **Best for**: Production deployments, multilingual content

**Advantages over v1**:
- 33% smaller embeddings (1024 vs 1536)
- Better retrieval accuracy
- Lower latency
- Reduced storage costs

### Alternative Bedrock Models

#### Cohere Embed English v3
```python
BEDROCK_MODEL_ID = 'cohere.embed-english-v3'
```
- **Dimensions**: 1024
- **Best for**: English-only, high accuracy
- **Features**: Built-in compression

#### Cohere Embed Multilingual v3
```python
BEDROCK_MODEL_ID = 'cohere.embed-multilingual-v3'
```
- **Dimensions**: 1024
- **Languages**: 100+ languages
- **Best for**: Multilingual applications

---

## Local Models (Sentence Transformers)

### ✨ Currently Active: all-mpnet-base-v2

**Model Name**: `sentence-transformers/all-mpnet-base-v2`
- **Dimensions**: 768
- **Max Input**: 384 word pieces (~512 tokens)
- **Size**: 420MB
- **Speed**: ~2500 sentences/sec (on GPU)
- **Best for**: General-purpose semantic search, high quality

**Why this is the default**:
- Best overall quality in MTEB benchmark
- Good balance of speed and accuracy
- Well-tested and widely used
- Excellent for most use cases

### Alternative Local Models

#### 🏃 Fast: all-MiniLM-L12-v2
```python
LOCAL_MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
```
- **Dimensions**: 384
- **Size**: 120MB
- **Speed**: ~14000 sentences/sec (on GPU)
- **Best for**: Fast inference, resource-constrained

#### ⚡ Fastest: all-MiniLM-L6-v2
```python
LOCAL_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
```
- **Dimensions**: 384
- **Size**: 80MB
- **Speed**: ~19000 sentences/sec (on GPU)
- **Best for**: Real-time applications, edge devices

#### 🎯 Search-Optimized: multi-qa-mpnet-base-dot-v1
```python
LOCAL_MODEL_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
```
- **Dimensions**: 768
- **Trained on**: Question-Answer pairs
- **Best for**: Q&A systems, information retrieval
- **Note**: Use dot product similarity

#### 🌍 Multilingual: paraphrase-multilingual-mpnet-base-v2
```python
LOCAL_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
```
- **Dimensions**: 768
- **Languages**: 50+ languages
- **Best for**: Multilingual applications

#### 🔬 Latest Research: msmarco-distilbert-base-v4
```python
LOCAL_MODEL_NAME = 'sentence-transformers/msmarco-distilbert-base-v4'
```
- **Dimensions**: 768
- **Trained on**: MS MARCO dataset (8.8M passages)
- **Best for**: Document retrieval, passage ranking

---

## Model Comparison Table

| Model | Dimensions | Speed | Quality | Size | Use Case |
|-------|-----------|-------|---------|------|----------|
| **Titan v2** | 1024 | Fast | Excellent | N/A | Production (cloud) |
| **all-mpnet-base-v2** ⭐ | 768 | Medium | Excellent | 420MB | General purpose |
| **all-MiniLM-L12-v2** | 384 | Fast | Very Good | 120MB | Balanced |
| **all-MiniLM-L6-v2** | 384 | Very Fast | Good | 80MB | Speed-critical |
| **multi-qa-mpnet** | 768 | Medium | Excellent | 420MB | Q&A systems |
| **multilingual-mpnet** | 768 | Medium | Excellent | 420MB | Multilingual |
| **msmarco-distilbert** | 768 | Fast | Excellent | 250MB | Search/retrieval |

---

## Benchmark Results (MTEB Leaderboard)

### Retrieval Task Performance

```
Model                           Avg Score
─────────────────────────────────────────
all-mpnet-base-v2              63.3%  ⭐
msmarco-distilbert-base-v4     62.1%
multi-qa-mpnet-base-dot-v1     61.8%
all-MiniLM-L12-v2              56.1%
all-MiniLM-L6-v2               51.3%
```

### Speed Comparison (sentences/sec on CPU)

```
Model                           Speed
─────────────────────────────────────────
all-MiniLM-L6-v2               1200 s/s  ⚡
all-MiniLM-L12-v2              800 s/s
msmarco-distilbert-base-v4     600 s/s
all-mpnet-base-v2              500 s/s
multi-qa-mpnet-base-dot-v1     500 s/s
```

---

## How to Change Models

### Option 1: Edit config.py

```python
# For AWS Bedrock
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2'

# For Local
LOCAL_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
```

### Option 2: Environment Variables

```bash
export BEDROCK_MODEL_ID=amazon.titan-embed-text-v2
export LOCAL_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

### Option 3: Code Override

```python
from embeddings import LocalEmbedding, BedrockEmbedding

# Use specific local model
gen = LocalEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Use specific Bedrock model
gen = BedrockEmbedding(model_id='cohere.embed-english-v3')
```

---

## Model Selection Guide

### 🎯 For Best Quality
→ **all-mpnet-base-v2** (local) or **Titan v2** (cloud)

### ⚡ For Best Speed
→ **all-MiniLM-L6-v2** (local)

### 💰 For Best Cost
→ **all-MiniLM-L6-v2** (local, free) or **Titan v2** (cloud, cheapest)

### 🌍 For Multilingual
→ **paraphrase-multilingual-mpnet-base-v2** (local) or **Cohere Multilingual v3** (cloud)

### 🔍 For Q&A/Search
→ **multi-qa-mpnet-base-dot-v1** (local) or **msmarco-distilbert-base-v4**

### 📱 For Edge/Mobile
→ **all-MiniLM-L6-v2** (smallest, fastest)

---

## Future Models (Coming Soon)

### OpenAI (via AWS Bedrock)
- **text-embedding-3-small** (1536d)
- **text-embedding-3-large** (3072d)

### Anthropic (Expected)
- Claude Embeddings API (dimensions TBD)

### Open Source
- **E5-mistral-7b-instruct** - Latest large model
- **BGE-M3** - Multilingual with 8192 token context
- **Jina-embeddings-v2** - 8192 token context

---

## Migration Notes

### From v1 to v2 (Titan)

```python
# Old
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v1'  # 1536 dimensions

# New
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2'  # 1024 dimensions
```

**Important**: You must re-index all documents when changing models!

### From all-MiniLM-L6-v2 to all-mpnet-base-v2

```python
# Old (384 dimensions)
LOCAL_MODEL_NAME = 'all-MiniLM-L6-v2'

# New (768 dimensions)
LOCAL_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
```

**Steps**:
1. Update config.py
2. Delete existing index
3. Re-create index with new dimension
4. Re-index all documents

---

## Performance Tips

### GPU Acceleration

```bash
# Install CUDA support
pip install sentence-transformers[cuda]

# 10-50x faster embedding generation
```

### Batch Processing

```python
# Generate embeddings in batches
embeddings = gen.batch_generate(texts, batch_size=32)

# Optimal batch sizes:
# - CPU: 8-16
# - GPU: 32-128
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    return gen.generate(text)
```

---

## Cost Comparison (1M documents)

```
Model                          Storage    Inference    Total
──────────────────────────────────────────────────────────────
Titan v2 (1024d)              $0.98      $100         $100.98
Titan v1 (1536d)              $1.47      $100         $101.47
all-mpnet-base-v2 (768d)      $0.74      $0           $0.74
all-MiniLM-L6-v2 (384d)       $0.37      $0           $0.37

Storage = dimensions × 4 bytes × 1M docs / 1GB × $0.024/GB/month
Inference = AWS Bedrock pricing (local is free)
```

---

## Recommended Configuration

### Development/Testing
```python
LOCAL_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Fast, small
HNSW_EF_SEARCH = 50  # Lower for speed
```

### Production (Quality)
```python
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2'  # Best cloud
# or
LOCAL_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # Best local
HNSW_EF_SEARCH = 100  # Balanced
```

### Production (Speed)
```python
LOCAL_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
HNSW_EF_SEARCH = 50
QUANTIZATION = 'fp16'  # 50% memory savings
```

---

## Testing Different Models

```python
from embeddings import LocalEmbedding
from search import VectorSearchEngine

# Test different models
models = [
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/all-MiniLM-L6-v2'
]

for model_name in models:
    print(f"\nTesting: {model_name}")
    
    gen = LocalEmbedding(model_name)
    search = VectorSearchEngine(embedding_generator=gen)
    
    results = search.semantic_search("test query", k=5)
    print(f"Results: {len(results)}")
```

---

## Resources

- [Sentence Transformers Models](https://www.sbert.net/docs/pretrained_models.html)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [AWS Bedrock Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Model Comparison Tool](https://huggingface.co/spaces/mteb/leaderboard)

---

**Last Updated**: June 2026
**Project Version**: 1.0
**Models Status**: All models active and tested ✅
