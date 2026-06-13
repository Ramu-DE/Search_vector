"""
OpenSearch indexer for creating and populating vector search indices
"""
import json
import boto3
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth
from tqdm import tqdm

from config import Config
from embeddings import get_embedding_generator, EmbeddingGenerator


class OpenSearchIndexer:
    """Manages OpenSearch index creation and data ingestion"""

    def __init__(self,
                 endpoint: str = None,
                 region: str = None,
                 index_name: str = None,
                 embedding_generator: EmbeddingGenerator = None):
        """
        Initialize OpenSearch indexer

        Args:
            endpoint: OpenSearch endpoint
            region: AWS region
            index_name: Index name
            embedding_generator: Embedding generator instance
        """
        self.endpoint = endpoint or Config.OPENSEARCH_ENDPOINT
        self.region = region or Config.AWS_REGION
        self.index_name = index_name or Config.OPENSEARCH_INDEX
        self.embedding_generator = embedding_generator or get_embedding_generator(use_bedrock=False)

        # Setup AWS authentication
        credentials = boto3.Session().get_credentials()
        self.awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
            'aoss',  # OpenSearch Serverless
            session_token=credentials.token
        )

        # Create OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': self.endpoint.replace('https://', ''), 'port': 443}],
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )

        print(f"Connected to OpenSearch at: {self.endpoint}")

    def create_index(self,
                    vector_field: str = 'title_vector',
                    dimension: int = None,
                    delete_if_exists: bool = False) -> bool:
        """
        Create OpenSearch index with vector field

        Args:
            vector_field: Name of the vector field
            dimension: Embedding dimension
            delete_if_exists: Delete existing index if present

        Returns:
            True if successful
        """
        dimension = dimension or self.embedding_generator.dimension

        # Delete existing index if requested
        if delete_if_exists and self.client.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.client.indices.delete(index=self.index_name)

        # Check if index already exists
        if self.client.indices.exists(index=self.index_name):
            print(f"Index already exists: {self.index_name}")
            return True

        # Define index mapping
        hnsw_config = Config.get_hnsw_config()

        index_body = {
            "settings": {
                "index.knn": True,
                "index.knn.algo_param.ef_search": hnsw_config['ef_search'],
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    # Text fields
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "plot": {"type": "text"},
                    "genre": {"type": "keyword"},

                    # Numeric fields
                    "year": {"type": "integer"},
                    "rating": {"type": "float"},
                    "runtime": {"type": "integer"},

                    # Vector field
                    vector_field: {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": "cosinesimil",
                            "parameters": {
                                "ef_construction": hnsw_config['ef_construction'],
                                "m": hnsw_config['m']
                            }
                        }
                    }
                }
            }
        }

        try:
            response = self.client.indices.create(
                index=self.index_name,
                body=index_body
            )
            print(f"Successfully created index: {self.index_name}")
            print(f"Vector field: {vector_field}, Dimension: {dimension}")
            return True

        except Exception as e:
            print(f"Error creating index: {e}")
            return False

    def index_documents(self,
                       documents: List[Dict[str, Any]],
                       vector_field: str = 'title_vector',
                       text_field: str = 'title',
                       batch_size: int = 100) -> Dict[str, int]:
        """
        Index documents with embeddings

        Args:
            documents: List of documents to index
            vector_field: Name of the vector field
            text_field: Field to generate embeddings from
            batch_size: Batch size for indexing

        Returns:
            Dict with success and failure counts
        """
        print(f"\nIndexing {len(documents)} documents...")

        # Generate embeddings for all documents
        texts = [doc.get(text_field, '') for doc in documents]
        print("Generating embeddings...")
        embeddings = self.embedding_generator.batch_generate(texts)

        # Prepare documents for bulk indexing
        actions = []
        for doc, embedding in zip(documents, embeddings):
            doc_with_vector = doc.copy()
            doc_with_vector[vector_field] = embedding

            action = {
                '_index': self.index_name,
                '_source': doc_with_vector
            }
            actions.append(action)

        # Bulk index documents
        print("Bulk indexing documents...")
        success_count = 0
        failed_count = 0

        for i in tqdm(range(0, len(actions), batch_size)):
            batch = actions[i:i + batch_size]

            try:
                success, failed = helpers.bulk(
                    self.client,
                    batch,
                    stats_only=True,
                    raise_on_error=False
                )
                success_count += success
                failed_count += len(failed) if failed else 0

            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                failed_count += len(batch)

        print(f"\nIndexing complete:")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}")

        # Force merge for better search performance
        if success_count > 0:
            print("Force merging segments...")
            self.client.indices.forcemerge(
                index=self.index_name,
                max_num_segments=1
            )
            print("Force merge complete")

        return {
            'success': success_count,
            'failed': failed_count
        }

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = self.client.indices.stats(index=self.index_name)

        primary_stats = stats['indices'][self.index_name]['primaries']

        return {
            'document_count': primary_stats['docs']['count'],
            'size_bytes': primary_stats['store']['size_in_bytes'],
            'segment_count': primary_stats['segments']['count']
        }


# Sample movie data for testing
SAMPLE_MOVIES = [
    {
        "title": "The Shawshank Redemption",
        "year": 1994,
        "rating": 9.3,
        "genre": ["Drama"],
        "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "runtime": 142
    },
    {
        "title": "The Godfather",
        "year": 1972,
        "rating": 9.2,
        "genre": ["Crime", "Drama"],
        "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "runtime": 175
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "rating": 9.0,
        "genre": ["Action", "Crime", "Drama"],
        "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.",
        "runtime": 152
    },
    {
        "title": "Forrest Gump",
        "year": 1994,
        "rating": 8.8,
        "genre": ["Drama", "Romance"],
        "plot": "The presidencies of Kennedy and Johnson, the Vietnam War, and other historical events unfold from the perspective of an Alabama man with an IQ of 75.",
        "runtime": 142
    },
    {
        "title": "Inception",
        "year": 2010,
        "rating": 8.8,
        "genre": ["Action", "Sci-Fi", "Thriller"],
        "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.",
        "runtime": 148
    },
    {
        "title": "The Matrix",
        "year": 1999,
        "rating": 8.7,
        "genre": ["Action", "Sci-Fi"],
        "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "runtime": 136
    },
    {
        "title": "Goodfellas",
        "year": 1990,
        "rating": 8.7,
        "genre": ["Crime", "Drama"],
        "plot": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners.",
        "runtime": 146
    },
    {
        "title": "Pulp Fiction",
        "year": 1994,
        "rating": 8.9,
        "genre": ["Crime", "Drama"],
        "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
        "runtime": 154
    },
    {
        "title": "The Lord of the Rings: The Return of the King",
        "year": 2003,
        "rating": 9.0,
        "genre": ["Action", "Adventure", "Drama"],
        "plot": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom.",
        "runtime": 201
    },
    {
        "title": "Fight Club",
        "year": 1999,
        "rating": 8.8,
        "genre": ["Drama"],
        "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.",
        "runtime": 139
    }
]


# Example usage
if __name__ == "__main__":
    print("=== OpenSearch Indexer Demo ===\n")

    # Note: This requires valid OpenSearch endpoint
    # For testing without OpenSearch, comment out the following

    try:
        # Initialize indexer with local embeddings
        indexer = OpenSearchIndexer()

        # Create index
        print("\n1. Creating index...")
        indexer.create_index(delete_if_exists=True)

        # Index sample documents
        print("\n2. Indexing sample movies...")
        results = indexer.index_documents(SAMPLE_MOVIES)

        # Get stats
        print("\n3. Index statistics:")
        stats = indexer.get_index_stats()
        print(f"  Documents: {stats['document_count']}")
        print(f"  Size: {stats['size_bytes']} bytes")
        print(f"  Segments: {stats['segment_count']}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This demo requires a valid OpenSearch endpoint.")
        print("Set AOSS_VECTORSEARCH_ENDPOINT environment variable or update config.py")
