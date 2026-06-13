# OpenSearch Serverless Setup Guide

## Current Status

✅ **AWS Credentials**: Valid (IAM role-based authentication)  
✅ **Bedrock Access**: Working  
❌ **OpenSearch Permissions**: Missing required permissions

## Required IAM Permissions

Your IAM role needs the following permissions to create and manage OpenSearch Serverless collections:

### Option 1: Using AWS Managed Policy (Easiest)

Attach the AWS managed policy: `AmazonOpenSearchServerlessFullAccess`

```bash
# If you have admin access, run:
aws iam attach-role-policy \
    --role-name code-server-CodeServerInstanceBootstrapRole-zxTXISgZoLSG \
    --policy-arn arn:aws:iam::aws:policy/AmazonOpenSearchServerlessFullAccess
```

### Option 2: Custom Policy (Minimal Permissions)

Create and attach this custom policy to your role:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "aoss:CreateSecurityPolicy",
                "aoss:GetSecurityPolicy",
                "aoss:UpdateSecurityPolicy",
                "aoss:ListSecurityPolicies",
                "aoss:CreateAccessPolicy",
                "aoss:GetAccessPolicy",
                "aoss:UpdateAccessPolicy",
                "aoss:CreateCollection",
                "aoss:DeleteCollection",
                "aoss:UpdateCollection",
                "aoss:BatchGetCollection",
                "aoss:ListCollections",
                "aoss:CreateIndex",
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadDocument",
                "aoss:WriteDocument"
            ],
            "Resource": "*"
        }
    ]
}
```

## Alternative: Use Pre-Created Collection

If you cannot modify IAM permissions, ask your AWS administrator to:

1. **Create an OpenSearch Serverless collection** named `movies-vector` with:
   - Type: `VECTORSEARCH`
   - Region: `us-west-2`

2. **Grant your role access** by adding this principal to the data access policy:
   ```
   arn:aws:iam::023457584489:role/code-server-CodeServerInstanceBootstrapRole-zxTXISgZoLSG
   ```

3. **Provide you with the collection endpoint** (format: `https://xxxxx.us-west-2.aoss.amazonaws.com`)

4. Add the endpoint to your `.env` file:
   ```bash
   AOSS_VECTORSEARCH_ENDPOINT=https://xxxxx.us-west-2.aoss.amazonaws.com
   ```

## Manual Setup via AWS Console

If you have console access:

1. **Go to OpenSearch Service Console**
   - Navigate to: https://console.aws.amazon.com/aos/home?region=us-west-2

2. **Create Serverless Collection**
   - Click "Collections" → "Create collection"
   - Collection name: `movies-vector`
   - Collection type: **Vector search**
   - Deployment type: **Serverless**

3. **Configure Security**
   - **Encryption**: Use AWS owned key
   - **Network access**: Public
   - **Data access**: Add your IAM role ARN:
     ```
     arn:aws:iam::023457584489:role/code-server-CodeServerInstanceBootstrapRole-zxTXISgZoLSG
     ```

4. **Wait for Collection Creation** (2-3 minutes)

5. **Copy the Endpoint**
   - From the collection details page
   - Format: `https://xxxxx.us-west-2.aoss.amazonaws.com`

6. **Add to .env file**:
   ```bash
   echo "AOSS_VECTORSEARCH_ENDPOINT=https://xxxxx.us-west-2.aoss.amazonaws.com" >> .env
   ```

7. **Create the Index**
   ```bash
   source .venv/bin/activate
   python -c "from setup_opensearch import OpenSearchSetup; s = OpenSearchSetup(); s.create_index('YOUR_ENDPOINT_HERE', 'movies')"
   ```

## Using Alternative Storage (For Testing)

If you want to test the application without OpenSearch Serverless:

### Option A: Local FAISS Vector Store

Create `vector_store_faiss.py`:
```python
import faiss
import numpy as np
import pickle
from typing import List, Dict

class FAISSVectorStore:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def add_vectors(self, embeddings: np.ndarray, docs: List[Dict]):
        self.index.add(embeddings)
        self.documents.extend(docs)
    
    def search(self, query_embedding: np.ndarray, k: int = 10):
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
    
    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)
```

Add to `requirements.txt`:
```
faiss-cpu>=1.7.4
```

### Option B: ChromaDB (Persistent Local Storage)

```python
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = client.create_collection(
    name="movies",
    metadata={"hnsw:space": "l2"}
)
```

Add to `requirements.txt`:
```
chromadb>=0.4.0
```

## Verification

Once you have OpenSearch configured, verify with:

```bash
source .venv/bin/activate
python test_aws_bedrock.py
```

You should see:
```
✓ OpenSearch Config: CONFIGURED
```

## Next Steps

After OpenSearch is configured:

1. ✅ Load movie data: `python ingest_data.py`
2. ✅ Start search UI: `streamlit run app.py`
3. ✅ Test vector search with 1M context Claude models

## Cost Estimation

**OpenSearch Serverless Vector Search Collection:**
- Minimum: ~$700/month (1 OCU indexing + 1 OCU search)
- Each OCU: ~$0.24/hour

**Alternative (No Additional Cost):**
- Use local FAISS or ChromaDB for testing (free, runs on your EC2 instance)
- Bedrock embeddings: Pay per API call ($0.0001 per 1K tokens)

---

**Your Current IAM Role:**
```
arn:aws:iam::023457584489:role/code-server-CodeServerInstanceBootstrapRole-zxTXISgZoLSG
```

**Account ID:** `023457584489`  
**Region:** `us-west-2`
