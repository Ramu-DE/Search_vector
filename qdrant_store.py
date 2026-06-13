#!/usr/bin/env python3
"""
Qdrant Vector Database Client
Handles vector storage and similarity search using Qdrant
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    MatchValue
)
from config import Config


class QdrantVectorStore:
    """Qdrant vector database client for movie search"""

    def __init__(self):
        """Initialize Qdrant client"""
        config = Config.get_qdrant_config()

        if not config['url'] or not config['api_key']:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

        self.client = QdrantClient(
            url=config['url'],
            api_key=config['api_key'],
            timeout=30
        )

        self.collection_name = config['collection']
        self.dimension = Config.BEDROCK_EMBEDDING_DIMENSION

        print(f"✓ Connected to Qdrant: {config['url']}")
        print(f"  Collection: {self.collection_name}")
        print(f"  Dimension: {self.dimension}")

    def create_collection(self, recreate: bool = False):
        """Create or recreate the collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists and recreate:
                print(f"  Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                exists = False

            if not exists:
                print(f"  Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"✓ Collection created")
            else:
                print(f"  Collection already exists: {self.collection_name}")

            return True

        except Exception as e:
            print(f"✗ Error creating collection: {e}")
            return False

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add documents with their embeddings to Qdrant

        Args:
            documents: List of document dicts with metadata
            embeddings: Numpy array of shape (n_documents, embedding_dim)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    'title': doc.get('title', ''),
                    'plot': doc.get('plot', ''),
                    'genre': doc.get('genre', ''),
                    'year': doc.get('year', 0),
                    'rating': doc.get('rating', 0.0),
                    'director': doc.get('director', ''),
                    'cast': doc.get('cast', ''),
                }
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  Uploaded batch {i//batch_size + 1} ({len(batch)} documents)")

        print(f"✓ Added {len(documents)} documents to Qdrant")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_rating: Optional[float] = None,
        year_range: Optional[tuple] = None,
        genre: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return
            min_rating: Minimum movie rating filter
            year_range: (min_year, max_year) tuple
            genre: Genre to filter by

        Returns:
            List of matching documents with scores
        """
        # Build filter
        must_conditions = []

        if min_rating is not None:
            must_conditions.append(
                FieldCondition(
                    key="rating",
                    range=Range(gte=min_rating)
                )
            )

        if year_range is not None:
            min_year, max_year = year_range
            must_conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(gte=min_year, lte=max_year)
                )
            )

        if genre is not None:
            must_conditions.append(
                FieldCondition(
                    key="genre",
                    match=MatchValue(value=genre)
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
            limit=k,
            query_filter=query_filter
        )

        # Format results
        formatted_results = []
        for result in results:
            doc = {
                'id': result.id,
                'score': result.score,
                **result.payload
            }
            formatted_results.append(doc)

        return formatted_results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.config.params.vectors.size,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            print(f"✗ Error getting collection info: {e}")
            return {}

    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ Error deleting collection: {e}")
            return False


def test_qdrant_connection():
    """Test Qdrant connection and configuration"""
    print("=" * 70)
    print("Testing Qdrant Connection")
    print("=" * 70)

    try:
        store = QdrantVectorStore()

        # List collections
        collections = store.client.get_collections()
        print(f"\n✓ Available collections: {len(collections.collections)}")
        for col in collections.collections:
            print(f"  • {col.name}")

        # Check if movies collection exists
        if any(c.name == store.collection_name for c in collections.collections):
            info = store.get_collection_info()
            print(f"\n✓ Collection '{store.collection_name}' exists")
            print(f"  Points: {info.get('points_count', 0)}")
            print(f"  Status: {info.get('status', 'unknown')}")
        else:
            print(f"\n⚠️  Collection '{store.collection_name}' does not exist yet")
            print("  Run setup_qdrant.py to create it")

        return True

    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\n📝 Add to your .env file:")
        print("QDRANT_URL=https://your-cluster.qdrant.io")
        print("QDRANT_API_KEY=your-api-key-here")
        return False

    except Exception as e:
        print(f"\n✗ Connection error: {e}")
        return False


if __name__ == "__main__":
    test_qdrant_connection()
