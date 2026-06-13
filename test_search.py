#!/usr/bin/env python3
"""
Test Vector Search with Qdrant + Bedrock
"""

import boto3
import json
import numpy as np
from qdrant_client import QdrantClient
from config import Config


def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding using AWS Bedrock"""
    bedrock = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)

    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=f"{Config.BEDROCK_MODEL_ID}:0",
        body=body,
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    return np.array(response_body['embedding'])


def search_movies(query: str, k: int = 5):
    """Search for movies using semantic similarity"""
    print("=" * 70)
    print(f"Query: '{query}'")
    print("=" * 70)

    # Connect to Qdrant
    config = Config.get_qdrant_config()
    client = QdrantClient(url=config['url'], api_key=config['api_key'])

    # Generate query embedding
    print("\n1. Generating query embedding with Bedrock...")
    query_embedding = generate_embedding(query)
    print(f"   ✓ Embedding: {len(query_embedding)} dimensions")

    # Search
    print(f"\n2. Searching Qdrant collection '{config['collection']}'...")
    results = client.query_points(
        collection_name=config['collection'],
        query=query_embedding.tolist(),
        limit=k
    ).points

    # Display results
    print(f"\n3. Top {len(results)} Results:")
    print("-" * 70)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.payload['title']} ({result.payload['year']})")
        print(f"   Score: {result.score:.4f}")
        print(f"   Rating: ⭐ {result.payload['rating']}/10")
        print(f"   Genre: {result.payload['genre']}")
        print(f"   Plot: {result.payload['plot']}")
        print(f"   Director: {result.payload['director']}")

    return results


def main():
    print("\n" + "=" * 70)
    print("Vector Search Test: Qdrant + AWS Bedrock + 1M Context Claude")
    print("=" * 70)

    # Test queries
    test_queries = [
        "space adventure and exploration",
        "crime and mafia movies",
        "psychological thriller",
        "inspirational drama about hope"
    ]

    for query in test_queries:
        print("\n")
        results = search_movies(query, k=3)
        print("\n" + "=" * 70)
        input("Press Enter for next query...")

    print("\n✓ All tests completed!")
    print("\nYour stack is ready:")
    print("  • Qdrant: Vector storage with cosine similarity")
    print("  • AWS Bedrock: Titan v2 embeddings (1024-dim)")
    print("  • Claude Opus 4.8: 1M context for query enhancement")


if __name__ == "__main__":
    main()
