"""
Search functionality for OpenSearch vector database
Supports lexical, semantic, and hybrid search
"""
import time
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

from config import Config
from embeddings import get_embedding_generator, EmbeddingGenerator


class VectorSearchEngine:
    """OpenSearch vector search engine"""

    def __init__(self,
                 endpoint: str = None,
                 region: str = None,
                 index_name: str = None,
                 embedding_generator: EmbeddingGenerator = None):
        """
        Initialize search engine

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
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
            'aoss',
            session_token=credentials.token
        )

        # Create OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': self.endpoint.replace('https://', ''), 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60
        )

        print(f"Search engine initialized for index: {self.index_name}")

    def keyword_search(self,
                      query: str,
                      k: int = 10,
                      filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform keyword (BM25) search

        Args:
            query: Search query
            k: Number of results
            filters: Optional filters (e.g., {"rating": {"gte": 8}})

        Returns:
            List of search results
        """
        # Build query
        query_body = {
            "size": k,
            "_source": {"excludes": ["*_vector"]},
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "plot", "genre"],
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            }
        }

        # Add filters if provided
        if filters:
            query_body["query"]["bool"]["filter"] = self._build_filters(filters)

        # Execute search
        start_time = time.time()
        response = self.client.search(body=query_body, index=self.index_name)
        latency_ms = (time.time() - start_time) * 1000

        # Process results
        results = self._process_results(response, latency_ms)
        return results

    def semantic_search(self,
                       query: str,
                       k: int = 10,
                       vector_field: str = 'title_vector',
                       filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic (vector) search using k-NN

        Args:
            query: Search query
            k: Number of results
            vector_field: Vector field name
            filters: Optional filters

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate(query)

        # Build k-NN query
        query_body = {
            "size": k,
            "_source": {"excludes": ["*_vector"]},
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                vector_field: {
                                    "vector": query_embedding,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            }
        }

        # Add filters if provided
        if filters:
            query_body["query"]["bool"]["filter"] = self._build_filters(filters)

        # Execute search
        start_time = time.time()
        response = self.client.search(body=query_body, index=self.index_name)
        latency_ms = (time.time() - start_time) * 1000

        # Process results
        results = self._process_results(response, latency_ms)
        return results

    def hybrid_search(self,
                     query: str,
                     k: int = 10,
                     vector_field: str = 'title_vector',
                     semantic_weight: float = 0.6,
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (keyword + semantic)

        Args:
            query: Search query
            k: Number of results
            vector_field: Vector field name
            semantic_weight: Weight for semantic search (0-1)
            filters: Optional filters

        Returns:
            List of search results
        """
        keyword_weight = 1 - semantic_weight

        # Generate query embedding
        query_embedding = self.embedding_generator.generate(query)

        # Build hybrid query
        query_body = {
            "size": k,
            "_source": {"excludes": ["*_vector"]},
            "query": {
                "bool": {
                    "should": [
                        # Keyword search (BM25)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "plot", "genre"],
                                "type": "best_fields",
                                "boost": keyword_weight
                            }
                        },
                        # Semantic search (k-NN)
                        {
                            "knn": {
                                vector_field: {
                                    "vector": query_embedding,
                                    "k": k,
                                    "boost": semantic_weight
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        # Add filters if provided
        if filters:
            query_body["query"]["bool"]["filter"] = self._build_filters(filters)

        # Execute search
        start_time = time.time()
        response = self.client.search(body=query_body, index=self.index_name)
        latency_ms = (time.time() - start_time) * 1000

        # Process results
        results = self._process_results(response, latency_ms)
        return results

    def compare_search_methods(self,
                               query: str,
                               k: int = 5,
                               filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare all three search methods

        Args:
            query: Search query
            k: Number of results
            filters: Optional filters

        Returns:
            Dict with results from all methods
        """
        print(f"\nComparing search methods for query: '{query}'")
        print("=" * 60)

        # Keyword search
        print("\n1. Keyword Search (BM25):")
        keyword_results = self.keyword_search(query, k, filters)

        # Semantic search
        print("\n2. Semantic Search (k-NN):")
        semantic_results = self.semantic_search(query, k, filters)

        # Hybrid search
        print("\n3. Hybrid Search (Combined):")
        hybrid_results = self.hybrid_search(query, k, filters)

        return {
            "query": query,
            "keyword": keyword_results,
            "semantic": semantic_results,
            "hybrid": hybrid_results
        }

    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build OpenSearch filter clauses"""
        filter_clauses = []

        for field, condition in filters.items():
            if isinstance(condition, dict):
                # Range filter
                if any(op in condition for op in ['gte', 'lte', 'gt', 'lt']):
                    filter_clauses.append({
                        "range": {field: condition}
                    })
                # Match filter
                else:
                    filter_clauses.append({
                        "term": {field: condition}
                    })
            else:
                # Simple term filter
                filter_clauses.append({
                    "term": {field: condition}
                })

        return filter_clauses

    def _process_results(self,
                        response: Dict[str, Any],
                        latency_ms: float) -> List[Dict[str, Any]]:
        """Process OpenSearch response"""
        hits = response['hits']['hits']
        total = response['hits']['total']['value']

        results = []
        for hit in hits:
            result = hit['_source'].copy()
            result['_score'] = hit['_score']
            result['_id'] = hit['_id']
            results.append(result)

        print(f"  Found {total} documents, returning top {len(results)}")
        print(f"  Latency: {latency_ms:.2f} ms")

        return results

    def print_results(self, results: List[Dict[str, Any]], max_plot_length: int = 100):
        """Pretty print search results"""
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('title', 'N/A')} ({result.get('year', 'N/A')})")
            print(f"   Rating: {result.get('rating', 'N/A')} | Score: {result['_score']:.4f}")
            print(f"   Genre: {', '.join(result.get('genre', []))}")

            plot = result.get('plot', '')
            if len(plot) > max_plot_length:
                plot = plot[:max_plot_length] + '...'
            print(f"   Plot: {plot}")


def interactive_search_demo():
    """Interactive search demo"""
    print("=" * 60)
    print("AI-Powered Movie Search Demo")
    print("=" * 60)

    try:
        # Initialize search engine
        search_engine = VectorSearchEngine()

        while True:
            print("\n" + "=" * 60)
            query = input("\nEnter search query (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            # Get filters
            min_rating = input("Minimum rating (press Enter to skip): ").strip()
            filters = {}
            if min_rating:
                try:
                    filters['rating'] = {'gte': float(min_rating)}
                except ValueError:
                    print("Invalid rating, skipping filter")

            # Search
            print("\nChoose search method:")
            print("1. Keyword (BM25)")
            print("2. Semantic (k-NN)")
            print("3. Hybrid (Both)")
            print("4. Compare All")

            choice = input("Enter choice (1-4): ").strip()

            if choice == '1':
                results = search_engine.keyword_search(query, k=5, filters=filters)
                search_engine.print_results(results)

            elif choice == '2':
                results = search_engine.semantic_search(query, k=5, filters=filters)
                search_engine.print_results(results)

            elif choice == '3':
                results = search_engine.hybrid_search(query, k=5, filters=filters)
                search_engine.print_results(results)

            elif choice == '4':
                comparison = search_engine.compare_search_methods(query, k=3, filters=filters)

                print("\n" + "=" * 60)
                print("KEYWORD RESULTS:")
                search_engine.print_results(comparison['keyword'])

                print("\n" + "=" * 60)
                print("SEMANTIC RESULTS:")
                search_engine.print_results(comparison['semantic'])

                print("\n" + "=" * 60)
                print("HYBRID RESULTS:")
                search_engine.print_results(comparison['hybrid'])

            else:
                print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Run interactive demo
    # Note: Requires valid OpenSearch endpoint with indexed data

    print("Starting interactive search demo...")
    print("Note: Requires AOSS_VECTORSEARCH_ENDPOINT environment variable")

    if Config.validate():
        interactive_search_demo()
    else:
        print("\nExample search queries:")
        print("  - 'movie to watch with friends'")
        print("  - 'uplifting underdog story'")
        print("  - 'crime thriller with plot twists'")
        print("  - 'sci-fi action adventure'")
