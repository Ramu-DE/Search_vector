#!/usr/bin/env python3
"""
Intelligent Movie Search with Long Context Claude
Combines: Qdrant + Bedrock Embeddings + Claude Sonnet 4.6 (1M context)
"""

import boto3
import json
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from bedrock_claude import BedrockClaude
from config import Config


class IntelligentMovieSearch:
    """
    AI-powered movie search with Claude's 1M context window
    """

    def __init__(self, claude_model: str = 'sonnet-4.6'):
        """
        Initialize intelligent search

        Args:
            claude_model: Claude model to use (default: sonnet-4.6 with 1M context)
        """
        # Initialize components
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)

        config = Config.get_qdrant_config()
        self.qdrant = QdrantClient(url=config['url'], api_key=config['api_key'])
        self.collection = config['collection']

        self.claude = BedrockClaude(model=claude_model)

        print(f"\n✓ Intelligent Search System Ready")
        print(f"  Embeddings: Bedrock Titan v2 (1024-dim)")
        print(f"  Vector DB: Qdrant ({self.collection})")
        print(f"  LLM: Claude {claude_model} (1M context)")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Bedrock Titan"""
        body = json.dumps({"inputText": text})
        response = self.bedrock_runtime.invoke_model(
            modelId=f"{Config.BEDROCK_MODEL_ID}:0",
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response['body'].read())
        return np.array(response_body['embedding'])

    def search(
        self,
        query: str,
        k: int = 10,
        enhance_query: bool = True,
        summarize: bool = True
    ) -> Dict[str, Any]:
        """
        Intelligent search with optional query enhancement and summarization

        Args:
            query: User's search query
            k: Number of results to return
            enhance_query: Use Claude to enhance the query
            summarize: Use Claude to summarize results

        Returns:
            Dict with results, summary, and metadata
        """
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")

        # Step 1: Optionally enhance query with Claude
        search_queries = [query]
        if enhance_query:
            print("\n1️⃣  Enhancing query with Claude...")
            enhanced = self.claude.enhance_query(query)
            search_queries.extend(enhanced)
            print(f"   Original: {query}")
            for i, eq in enumerate(enhanced, 1):
                print(f"   Enhanced {i}: {eq}")

        # Step 2: Extract search preferences
        print("\n2️⃣  Extracting preferences...")
        preferences = self.claude.extract_preferences(query)
        print(f"   Genre: {preferences.get('genre', 'Any')}")
        print(f"   Min Rating: {preferences.get('min_rating', 'Any')}")
        print(f"   Year: {preferences.get('min_year', 'Any')} - {preferences.get('max_year', 'Any')}")

        # Step 3: Generate embeddings for all queries
        print(f"\n3️⃣  Generating embeddings...")
        all_results = []
        seen_titles = set()

        for search_query in search_queries:
            embedding = self.generate_embedding(search_query)

            # Search Qdrant with preferences
            results = self.qdrant.query_points(
                collection_name=self.collection,
                query=embedding.tolist(),
                limit=k * 2  # Get more to deduplicate
            ).points

            # Deduplicate and collect
            for result in results:
                title = result.payload['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_results.append({
                        'id': result.id,
                        'title': title,
                        'plot': result.payload['plot'],
                        'genre': result.payload['genre'],
                        'year': result.payload['year'],
                        'rating': result.payload['rating'],
                        'director': result.payload['director'],
                        'cast': result.payload['cast'],
                        'score': result.score
                    })

        # Sort by score and limit
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:k]

        print(f"   Found {len(all_results)} unique results")

        # Step 4: Display results
        print(f"\n4️⃣  Top Results:")
        for i, result in enumerate(all_results[:5], 1):
            print(f"\n   {i}. {result['title']} ({result['year']})")
            print(f"      ⭐ {result['rating']}/10  |  {result['genre']}  |  Score: {result['score']:.3f}")
            print(f"      {result['plot']}")

        # Step 5: Optionally summarize with Claude
        summary = None
        if summarize and all_results:
            print(f"\n5️⃣  Generating AI summary...")
            summary = self.claude.summarize_results(all_results, query)
            print(f"\n📝 AI Summary:")
            print(f"{summary}")

        return {
            'query': query,
            'enhanced_queries': search_queries[1:] if enhance_query else [],
            'preferences': preferences,
            'results': all_results,
            'summary': summary,
            'count': len(all_results)
        }

    def conversational_search(self, query: str) -> str:
        """
        Conversational search interface - returns natural language response

        Args:
            query: Natural language question

        Returns:
            Natural language answer with recommendations
        """
        # Search
        search_result = self.search(query, k=5, enhance_query=True, summarize=True)

        # Format as conversational response
        if not search_result['results']:
            return "I couldn't find any movies matching your query. Try different keywords?"

        response_parts = []

        # Add summary
        if search_result['summary']:
            response_parts.append(search_result['summary'])

        # Add top recommendations
        response_parts.append("\n\n📽️ **Top Recommendations:**")
        for i, movie in enumerate(search_result['results'][:3], 1):
            response_parts.append(
                f"\n{i}. **{movie['title']}** ({movie['year']}) - ⭐ {movie['rating']}/10\n"
                f"   {movie['plot']}\n"
                f"   Director: {movie['director']}"
            )

        return "\n".join(response_parts)


def demo():
    """Interactive demo"""
    print("=" * 70)
    print("🎬 Intelligent Movie Search with Long Context Claude")
    print("=" * 70)

    # Initialize
    search = IntelligentMovieSearch(claude_model='sonnet-4.6')

    # Demo queries
    demo_queries = [
        "I want an epic space adventure",
        "dark psychological thriller that will mess with my mind",
        "highly rated drama from the 90s about hope and redemption",
        "crime movies with complex plots"
    ]

    for query in demo_queries:
        print("\n\n" + "=" * 70)
        input(f"Press Enter to search: '{query}'...")

        result = search.search(
            query=query,
            k=5,
            enhance_query=True,
            summarize=True
        )

        print(f"\n✓ Search complete: {result['count']} results")

    print("\n\n" + "=" * 70)
    print("🎉 Demo Complete!")
    print("=" * 70)
    print("\nYour system combines:")
    print("  • 1M context Claude Sonnet 4.6 (via Bedrock)")
    print("  • 1024-dim semantic embeddings (Titan v2)")
    print("  • Fast vector search (Qdrant)")
    print("  • Query enhancement + result summarization")
    print("\nReady for production use! 🚀")


if __name__ == "__main__":
    demo()
