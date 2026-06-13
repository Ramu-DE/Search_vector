# 🚀 Production Deployment Guide

## Quick Start

### Local Development

```bash
# 1. Start the application
./RUN_APP.sh

# Or directly:
source .venv/bin/activate
streamlit run app.py          # Web UI
python api.py                 # REST API
python intelligent_search.py  # CLI Demo
```

---

## Application Options

### 1. Streamlit Web UI (Recommended)

**Best for**: Interactive user-facing application

```bash
streamlit run app.py
```

**Access**: http://localhost:8501

**Features**:
- Beautiful, responsive UI
- Real-time AI search
- Model selection (Opus, Sonnet, Haiku)
- Query enhancement toggle
- AI summary display
- Advanced options panel

**Production**:
```bash
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```

### 2. FastAPI REST API

**Best for**: Backend service, mobile apps, integrations

```bash
python api.py
```

**Access**:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**Endpoints**:
- `POST /search` - Full search with all options
- `GET /search?query=...` - Simple search (URL parameters)
- `GET /models` - List available Claude models
- `POST /enhance-query` - Query enhancement only
- `POST /extract-preferences` - Preference extraction only

**Example Request**:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "epic space adventure",
    "k": 5,
    "enhance_query": true,
    "summarize": true
  }'
```

**Production**:
```bash
uvicorn api:app --host 0.0.0.0 --port 80 --workers 4
```

### 3. CLI Demo

**Best for**: Testing, scripting, automation

```bash
python intelligent_search.py
```

---

## Docker Deployment

### Build Image

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8501 8000

# Default to Streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
# Build
docker build -t ai-movie-search .

# Run Streamlit
docker run -p 8501:8501 \
  -e AWS_REGION=us-west-2 \
  -e QDRANT_URL=https://your-cluster.qdrant.io \
  -e QDRANT_API_KEY=your-key \
  ai-movie-search

# Run API
docker run -p 8000:8000 \
  -e AWS_REGION=us-west-2 \
  -e QDRANT_URL=https://your-cluster.qdrant.io \
  -e QDRANT_API_KEY=your-key \
  ai-movie-search python api.py
```

---

## AWS Deployment Options

### Option 1: EC2 Instance

```bash
# 1. Launch EC2 instance (t3.medium recommended)
# 2. Install dependencies
sudo yum update -y
sudo yum install python3.12 git -y

# 3. Clone and setup
git clone <your-repo>
cd Search_Vector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Configure environment
cat > .env << EOF
AWS_REGION=us-west-2
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
EOF

# 5. Run with systemd
sudo nano /etc/systemd/system/ai-search.service
```

**systemd service file**:
```ini
[Unit]
Description=AI Movie Search API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/Search_Vector
Environment="PATH=/home/ec2-user/Search_Vector/.venv/bin"
ExecStart=/home/ec2-user/Search_Vector/.venv/bin/python api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ai-search
sudo systemctl start ai-search
```

### Option 2: ECS Fargate

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AWS_REGION=us-west-2
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    command: streamlit run app.py --server.address 0.0.0.0

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AWS_REGION=us-west-2
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    command: python api.py
```

### Option 3: Lambda + API Gateway

**Note**: Lambda is challenging for this use case due to:
- Large dependencies (boto3, qdrant-client, etc.)
- 1M context requires longer timeouts
- Cold start latency

**Recommended**: Use ECS Fargate or EC2 instead

---

## Environment Variables

### Required

```bash
# AWS Configuration
AWS_REGION=us-west-2

# Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

### Optional

```bash
# Collection name (default: movies)
QDRANT_COLLECTION=movies

# Bedrock model (default: amazon.titan-embed-text-v2)
BEDROCK_MODEL_ID=amazon.titan-embed-text-v2

# Embedding dimensions (default: 1024)
BEDROCK_EMBEDDING_DIMENSION=1024
```

---

## Scaling & Performance

### Streamlit

**Concurrent users**: 10-50 (single instance)

**Scale**:
- Run multiple instances behind load balancer
- Use nginx for reverse proxy
- Enable caching with `@st.cache_resource`

### FastAPI

**Concurrent users**: 100-1000 (4 workers)

**Scale**:
```bash
# Multiple workers
uvicorn api:app --workers 4 --host 0.0.0.0 --port 8000

# Behind nginx
upstream api_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
    }
}
```

### Database

**Qdrant**:
- Free tier: 1GB, sufficient for 1M vectors
- Paid plans: Unlimited, with clustering

**Scale**:
- Add more data to Qdrant (up to millions of movies)
- Use Qdrant's horizontal scaling
- Enable quantization for memory savings

### Bedrock

**Embeddings**:
- Fully managed, auto-scales
- ~100ms latency per request
- Batch requests when possible

**Claude Models**:
- Fully managed, auto-scales
- ~1-3s for 1M context queries
- Use Haiku for high-volume, Opus for quality

---

## Monitoring

### Health Checks

```bash
# API Health
curl http://localhost:8000/health

# Streamlit (check process)
ps aux | grep streamlit
```

### Logging

```python
# Add to config.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

**CloudWatch** (if on AWS):
```python
import boto3
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='AIMovieSearch',
    MetricData=[
        {
            'MetricName': 'SearchLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds'
        }
    ]
)
```

---

## Security

### API Keys

```bash
# Never commit .env files
echo ".env" >> .gitignore

# Use AWS Secrets Manager (production)
aws secretsmanager create-secret \
    --name ai-search/qdrant \
    --secret-string '{"url":"...","key":"..."}'
```

### Authentication

**Add to FastAPI**:
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

@app.post("/search")
async def search_movies(
    request: SearchRequest,
    credentials: HTTPAuthorizationCredentials = Security(verify_token)
):
    # ... search logic
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/search")
@limiter.limit("10/minute")
async def search_movies(request: SearchRequest):
    # ... search logic
```

---

## Cost Optimization

### Model Selection

| Workload | Recommended Model | Cost/1K searches |
|----------|------------------|------------------|
| High volume | Haiku 4.5 | ~$1 |
| Production | Sonnet 4.6 | ~$6 |
| Premium | Opus 4.8 | ~$12 |

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_embedding(text: str):
    return generate_embedding(text)

# Or use Redis
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

def get_or_generate_embedding(text: str):
    key = hashlib.md5(text.encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    
    embedding = generate_embedding(text)
    r.setex(key, 86400, json.dumps(embedding))  # Cache 24h
    return embedding
```

### Optimization Tips

1. **Disable query enhancement** for simple searches (-50% cost)
2. **Cache embeddings** for common queries (-30% latency)
3. **Use Haiku** for non-critical searches (-80% cost)
4. **Batch similar requests** (if applicable)
5. **Set max_tokens limits** on Claude (faster responses)

---

## Troubleshooting

### "Connection refused" (Qdrant)
```bash
# Check cluster status
curl https://your-cluster.qdrant.io:6333/collections

# Verify API key
echo $QDRANT_API_KEY
```

### "Model not found" (Bedrock)
```bash
# List available models
aws bedrock list-foundation-models --region us-west-2

# Use inference profile
# Change: anthropic.claude-sonnet-4-6
# To: us.anthropic.claude-sonnet-4-6
```

### "Slow responses"
```bash
# Check model
# Haiku 4.5: Fast (200ms)
# Sonnet 4.6: Medium (1-2s)
# Opus 4.8: Slow (2-5s)

# Disable features
enhance_query=False  # Skip query expansion
summarize=False      # Skip AI summary
```

### "High costs"
```python
# Switch to cheaper model
search = IntelligentMovieSearch(claude_model='haiku-4.5')

# Reduce features
result = search.search(
    query=query,
    k=5,
    enhance_query=False,  # Save on Claude calls
    summarize=False       # Save on Claude calls
)
```

---

## Production Checklist

- [ ] Environment variables configured
- [ ] .env file in .gitignore
- [ ] Health check endpoint working
- [ ] Logging enabled
- [ ] Error handling tested
- [ ] Rate limiting configured
- [ ] Authentication added (if public)
- [ ] Monitoring setup
- [ ] Backup strategy for Qdrant
- [ ] Cost alerts configured
- [ ] Load testing completed
- [ ] Documentation updated

---

## Support & Resources

- **Qdrant Docs**: https://qdrant.tech/documentation/
- **AWS Bedrock**: https://docs.aws.amazon.com/bedrock/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/

---

*Ready for production deployment!* 🚀
