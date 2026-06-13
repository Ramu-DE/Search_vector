#!/usr/bin/env python3
"""
Ingest Movie Data into Qdrant
Generates embeddings using AWS Bedrock and stores in Qdrant
"""

import boto3
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from qdrant_store import QdrantVectorStore
from config import Config


def generate_embedding(text: str, bedrock_client) -> List[float]:
    """Generate embedding using AWS Bedrock Titan model"""
    body = json.dumps({"inputText": text})

    response = bedrock_client.invoke_model(
        modelId=f"{Config.BEDROCK_MODEL_ID}:0",
        body=body,
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    return response_body['embedding']


def load_sample_movies() -> List[Dict]:
    """Load sample movie data"""
    # Sample movies dataset
    movies = [
        {
            "title": "The Shawshank Redemption",
            "plot": "Two imprisoned men bond over years, finding redemption through acts of common decency.",
            "genre": "Drama",
            "year": 1994,
            "rating": 9.3,
            "director": "Frank Darabont",
            "cast": "Tim Robbins, Morgan Freeman"
        },
        {
            "title": "The Godfather",
            "plot": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
            "genre": "Crime",
            "year": 1972,
            "rating": 9.2,
            "director": "Francis Ford Coppola",
            "cast": "Marlon Brando, Al Pacino"
        },
        {
            "title": "Interstellar",
            "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
            "genre": "Sci-Fi",
            "year": 2014,
            "rating": 8.6,
            "director": "Christopher Nolan",
            "cast": "Matthew McConaughey, Anne Hathaway"
        },
        {
            "title": "Inception",
            "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task.",
            "genre": "Sci-Fi",
            "year": 2010,
            "rating": 8.8,
            "director": "Christopher Nolan",
            "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt"
        },
        {
            "title": "The Matrix",
            "plot": "A computer hacker learns about the true nature of his reality and his role in the war against its controllers.",
            "genre": "Sci-Fi",
            "year": 1999,
            "rating": 8.7,
            "director": "Wachowski Brothers",
            "cast": "Keanu Reeves, Laurence Fishburne"
        },
        {
            "title": "Forrest Gump",
            "plot": "The presidencies of Kennedy and Johnson unfold through the perspective of an Alabama man.",
            "genre": "Drama",
            "year": 1994,
            "rating": 8.8,
            "director": "Robert Zemeckis",
            "cast": "Tom Hanks, Robin Wright"
        },
        {
            "title": "Pulp Fiction",
            "plot": "The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine in four tales of violence.",
            "genre": "Crime",
            "year": 1994,
            "rating": 8.9,
            "director": "Quentin Tarantino",
            "cast": "John Travolta, Uma Thurman"
        },
        {
            "title": "The Dark Knight",
            "plot": "Batman must accept one of the greatest psychological tests to fight injustice.",
            "genre": "Action",
            "year": 2008,
            "rating": 9.0,
            "director": "Christopher Nolan",
            "cast": "Christian Bale, Heath Ledger"
        },
        {
            "title": "Fight Club",
            "plot": "An insomniac office worker forms an underground fight club with a soap salesman.",
            "genre": "Drama",
            "year": 1999,
            "rating": 8.8,
            "director": "David Fincher",
            "cast": "Brad Pitt, Edward Norton"
        },
        {
            "title": "Goodfellas",
            "plot": "The story of Henry Hill and his life in the mob covering his relationship with his wife.",
            "genre": "Crime",
            "year": 1990,
            "rating": 8.7,
            "director": "Martin Scorsese",
            "cast": "Robert De Niro, Ray Liotta"
        },
        {
            "title": "The Silence of the Lambs",
            "plot": "A young FBI cadet must receive help from an incarcerated cannibal killer.",
            "genre": "Thriller",
            "year": 1991,
            "rating": 8.6,
            "director": "Jonathan Demme",
            "cast": "Jodie Foster, Anthony Hopkins"
        },
        {
            "title": "Saving Private Ryan",
            "plot": "Following the Normandy landings, a group of soldiers go behind enemy lines to retrieve a paratrooper.",
            "genre": "War",
            "year": 1998,
            "rating": 8.6,
            "director": "Steven Spielberg",
            "cast": "Tom Hanks, Matt Damon"
        },
        {
            "title": "Schindler's List",
            "plot": "In German-occupied Poland, industrialist Oskar Schindler becomes concerned for his Jewish workforce.",
            "genre": "Drama",
            "year": 1993,
            "rating": 9.0,
            "director": "Steven Spielberg",
            "cast": "Liam Neeson, Ben Kingsley"
        },
        {
            "title": "Parasite",
            "plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between families.",
            "genre": "Thriller",
            "year": 2019,
            "rating": 8.5,
            "director": "Bong Joon Ho",
            "cast": "Song Kang-ho, Lee Sun-kyun"
        },
        {
            "title": "Gladiator",
            "plot": "A former Roman General sets out to exact vengeance against the corrupt emperor.",
            "genre": "Action",
            "year": 2000,
            "rating": 8.5,
            "director": "Ridley Scott",
            "cast": "Russell Crowe, Joaquin Phoenix"
        }
    ]

    return movies


def ingest_movies():
    """Main ingestion function"""
    print("=" * 70)
    print("Movie Data Ingestion - Qdrant + AWS Bedrock")
    print("=" * 70)

    # Initialize clients
    print("\n1. Initializing connections...")

    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)
        print("  ✓ AWS Bedrock connected")
    except Exception as e:
        print(f"  ✗ AWS Bedrock error: {e}")
        return False

    try:
        vector_store = QdrantVectorStore()
    except Exception as e:
        print(f"  ✗ Qdrant error: {e}")
        return False

    # Ensure collection exists
    print("\n2. Setting up collection...")
    if not vector_store.create_collection(recreate=False):
        return False

    # Load movies
    print("\n3. Loading movie data...")
    movies = load_sample_movies()
    print(f"  ✓ Loaded {len(movies)} movies")

    # Generate embeddings
    print("\n4. Generating embeddings with AWS Bedrock...")
    embeddings = []

    for movie in tqdm(movies, desc="  Processing"):
        # Combine title and plot for embedding
        text = f"{movie['title']}. {movie['plot']}"

        try:
            embedding = generate_embedding(text, bedrock_client)
            embeddings.append(embedding)
        except Exception as e:
            print(f"\n  ✗ Error embedding '{movie['title']}': {e}")
            return False

    embeddings_array = np.array(embeddings)
    print(f"  ✓ Generated {len(embeddings)} embeddings ({embeddings_array.shape[1]}D)")

    # Upload to Qdrant
    print("\n5. Uploading to Qdrant...")
    try:
        vector_store.add_documents(movies, embeddings_array)
    except Exception as e:
        print(f"  ✗ Upload error: {e}")
        return False

    # Verify
    print("\n6. Verifying...")
    info = vector_store.get_collection_info()
    print(f"  ✓ Collection points: {info.get('points_count', 0)}")

    # Test search
    print("\n7. Testing search...")
    test_query = "space adventure movie"
    try:
        test_embedding = generate_embedding(test_query, bedrock_client)
        results = vector_store.search(np.array(test_embedding), k=3)

        print(f"  Query: '{test_query}'")
        print(f"  Top results:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['title']} (score: {result['score']:.3f})")
    except Exception as e:
        print(f"  ✗ Search test error: {e}")

    print("\n" + "=" * 70)
    print("Ingestion Complete!")
    print("=" * 70)
    print(f"\n✓ Ingested {len(movies)} movies")
    print(f"✓ Vector dimensions: {Config.BEDROCK_EMBEDDING_DIMENSION}")
    print(f"✓ Collection: {Config.QDRANT_COLLECTION}")

    print("\nNext step:")
    print("  streamlit run app.py")

    return True


if __name__ == "__main__":
    success = ingest_movies()
    exit(0 if success else 1)
