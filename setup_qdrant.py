#!/usr/bin/env python3
"""
Setup Qdrant Vector Database
Creates collection and prepares for data ingestion
"""

from qdrant_client import QdrantClient
from config import Config


def setup_qdrant():
    """Setup Qdrant collection"""
    print("=" * 70)
    print("Qdrant Vector Database Setup")
    print("=" * 70)

    # Get configuration
    config = Config.get_qdrant_config()

    if not config['url'] or not config['api_key']:
        print("\n✗ Missing Qdrant configuration")
        print("\n📝 Add these to your .env file:")
        print("QDRANT_URL=https://your-cluster.qdrant.io")
        print("QDRANT_API_KEY=your-api-key-here")
        print("\nGet your free Qdrant Cloud account at: https://cloud.qdrant.io/")
        return False

    print(f"\nQdrant URL: {config['url']}")
    print(f"Collection: {config['collection']}")
    print(f"Dimension: {Config.BEDROCK_EMBEDDING_DIMENSION}")

    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=config['url'],
            api_key=config['api_key'],
            timeout=30
        )

        print("\n✓ Connected to Qdrant")

        # Check existing collections
        collections = client.get_collections()
        existing = [c.name for c in collections.collections]

        if existing:
            print(f"\nExisting collections: {', '.join(existing)}")

        # Create collection if needed
        if config['collection'] in existing:
            print(f"\n⚠️  Collection '{config['collection']}' already exists")
            response = input("Recreate collection? This will delete all data (y/N): ")

            if response.lower() == 'y':
                print(f"  Deleting collection: {config['collection']}")
                client.delete_collection(config['collection'])
                print(f"  ✓ Deleted")
            else:
                print("  Keeping existing collection")
                return True

        # Create collection
        print(f"\nCreating collection: {config['collection']}")

        from qdrant_client.models import Distance, VectorParams

        client.create_collection(
            collection_name=config['collection'],
            vectors_config=VectorParams(
                size=Config.BEDROCK_EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )

        print(f"✓ Collection created successfully")
        print(f"\nCollection details:")
        print(f"  Name: {config['collection']}")
        print(f"  Vector size: {Config.BEDROCK_EMBEDDING_DIMENSION}")
        print(f"  Distance metric: Cosine")

        # Verify
        info = client.get_collection(config['collection'])
        print(f"  Status: {info.status}")

        print("\n" + "=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: python ingest_data_qdrant.py (to load movie data)")
        print("2. Run: streamlit run app.py (to start the search UI)")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your QDRANT_URL is correct")
        print("2. Verify your QDRANT_API_KEY is valid")
        print("3. Ensure your Qdrant cluster is running")
        return False


if __name__ == "__main__":
    success = setup_qdrant()
    exit(0 if success else 1)
