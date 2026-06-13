#!/usr/bin/env python3
"""Quick test of the API endpoints"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("Testing API endpoints...")
    print("=" * 70)

    # Test health
    print("\n1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # Test models
    print("\n2. List Models")
    response = requests.get(f"{BASE_URL}/models")
    print(json.dumps(response.json(), indent=2))

    # Test search
    print("\n3. Search Movies")
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": "epic space adventure",
            "k": 3,
            "enhance_query": True,
            "summarize": True
        }
    )
    result = response.json()
    print(f"Query: {result['query']}")
    print(f"Results: {result['count']}")
    for i, movie in enumerate(result['results'], 1):
        print(f"\n{i}. {movie['title']} ({movie['year']})")
        print(f"   Score: {movie['score']:.3f} | Rating: {movie['rating']}/10")

    if result['summary']:
        print(f"\nAI Summary:\n{result['summary'][:200]}...")

if __name__ == "__main__":
    print("Start the API first: python api.py")
    print("Then run this test in another terminal")
