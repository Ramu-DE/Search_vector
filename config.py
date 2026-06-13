"""
Configuration settings for the AI-powered search application
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""

    # AWS Settings
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-west-2')

    # Vector Database Settings
    # Qdrant (Cloud or Self-hosted)
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_COLLECTION: str = 'movies'

    # OpenSearch Serverless (Alternative)
    OPENSEARCH_ENDPOINT: str = os.getenv('AOSS_VECTORSEARCH_ENDPOINT', '')
    OPENSEARCH_INDEX: str = 'movies'

    # Bedrock Settings (Latest Models)
    BEDROCK_MODEL_ID: str = 'amazon.titan-embed-text-v2'  # Latest: v2 with 1024 dimensions
    BEDROCK_EMBEDDING_DIMENSION: int = 1024

    # Alternative: Local embedding model (Latest Models)
    # Best overall: 'sentence-transformers/all-mpnet-base-v2' (768d, best quality)
    # Fast & good: 'sentence-transformers/all-MiniLM-L12-v2' (384d, balanced)
    # Fastest: 'sentence-transformers/all-MiniLM-L6-v2' (384d, fastest)
    LOCAL_MODEL_NAME: str = 'sentence-transformers/all-mpnet-base-v2'  # Latest recommended
    LOCAL_EMBEDDING_DIMENSION: int = 768

    # Search Settings
    DEFAULT_K: int = 10
    DEFAULT_MIN_RATING: float = 7.0

    # HNSW Parameters (Optimized for latest models)
    HNSW_M: int = 16  # Graph connectivity (8-32, higher = better recall)
    HNSW_EF_CONSTRUCTION: int = 256  # Index build quality (128-512)
    HNSW_EF_SEARCH: int = 100  # Query accuracy (50-200, tune this for speed/accuracy)

    # OpenSearch 2.17+ features
    USE_DISK_BASED_VECTORS: bool = False  # Enable for 32x memory reduction
    QUANTIZATION: str = 'none'  # Options: 'none', 'fp16', 'int8', 'binary'

    # Hybrid Search
    SEMANTIC_WEIGHT: float = 0.6  # 60% semantic, 40% keyword

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if cls.QDRANT_URL and cls.QDRANT_API_KEY:
            return True
        if cls.OPENSEARCH_ENDPOINT:
            return True
        print("Warning: No vector database configured")
        print("Set either QDRANT_URL + QDRANT_API_KEY or OPENSEARCH_ENDPOINT")
        return False

    @classmethod
    def get_qdrant_config(cls) -> dict:
        """Get Qdrant configuration"""
        return {
            'url': cls.QDRANT_URL,
            'api_key': cls.QDRANT_API_KEY,
            'collection': cls.QDRANT_COLLECTION
        }

    @classmethod
    def get_opensearch_config(cls) -> dict:
        """Get OpenSearch configuration"""
        return {
            'endpoint': cls.OPENSEARCH_ENDPOINT,
            'index': cls.OPENSEARCH_INDEX,
            'region': cls.AWS_REGION
        }

    @classmethod
    def get_bedrock_config(cls) -> dict:
        """Get Bedrock configuration"""
        return {
            'model_id': cls.BEDROCK_MODEL_ID,
            'dimension': cls.BEDROCK_EMBEDDING_DIMENSION,
            'region': cls.AWS_REGION
        }

    @classmethod
    def get_hnsw_config(cls) -> dict:
        """Get HNSW algorithm configuration"""
        return {
            'm': cls.HNSW_M,
            'ef_construction': cls.HNSW_EF_CONSTRUCTION,
            'ef_search': cls.HNSW_EF_SEARCH
        }
