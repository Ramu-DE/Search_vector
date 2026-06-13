#!/usr/bin/env python3
"""
Claude Models via AWS Bedrock
No Anthropic API key needed - uses your existing AWS credentials
"""

import boto3
import json
from typing import List, Dict, Any, Optional
from config import Config


class BedrockClaude:
    """Claude client using AWS Bedrock"""

    # Available Claude models via Bedrock Inference Profiles
    # Using US cross-region profiles for better availability
    MODELS = {
        'opus-4.8': 'us.anthropic.claude-opus-4-8',           # 1M context, most capable
        'opus-4.7': 'us.anthropic.claude-opus-4-7',           # 1M context
        'opus-4.6': 'us.anthropic.claude-opus-4-6-v1',        # 1M context
        'opus-4.5': 'us.anthropic.claude-opus-4-5-20251101-v1:0',  # 1M context
        'sonnet-4.6': 'us.anthropic.claude-sonnet-4-6',       # 1M context, best balance
        'sonnet-4.5': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',  # 1M context
        'haiku-4.5': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',  # 200K context, fastest
        'sonnet-3.5-haiku': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',  # 200K context
        'sonnet-3': 'us.anthropic.claude-3-sonnet-20240229-v1:0',  # 200K context
        'fable-5': 'us.anthropic.claude-fable-5',             # 1M context, cutting edge
    }

    # Default to Claude Sonnet 4.6 (1M context, excellent performance/cost balance)
    DEFAULT_MODEL = 'sonnet-4.6'

    def __init__(self, model: str = None, region: str = None):
        """
        Initialize Bedrock Claude client

        Args:
            model: Model key from MODELS dict (default: opus-4.8)
            region: AWS region (default: from Config)
        """
        self.region = region or Config.AWS_REGION
        self.client = boto3.client('bedrock-runtime', region_name=self.region)

        # Set model
        model_key = model or self.DEFAULT_MODEL
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODELS.keys())}")

        self.model_id = self.MODELS[model_key]
        self.model_name = model_key

        print(f"✓ Bedrock Claude initialized")
        print(f"  Model: {self.model_name} ({self.model_id})")
        print(f"  Region: {self.region}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with Claude via Bedrock

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system: System prompt
            stop_sequences: Stop generation at these sequences

        Returns:
            Dict with 'text', 'stop_reason', 'usage'
        """
        # Build request
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if system:
            request_body["system"] = system

        if stop_sequences:
            request_body["stop_sequences"] = stop_sequences

        # Invoke model
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )

        # Parse response
        response_body = json.loads(response['body'].read())

        return {
            'text': response_body['content'][0]['text'],
            'stop_reason': response_body.get('stop_reason'),
            'usage': response_body.get('usage', {}),
            'model': self.model_id
        }

    def summarize_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Summarize search results using Claude

        Args:
            results: List of search results with title, plot, rating, etc.
            query: Original user query

        Returns:
            Natural language summary
        """
        # Format results
        results_text = "\n".join([
            f"{i+1}. {r['title']} ({r['year']}) - Rating: {r['rating']}/10"
            f"\n   Plot: {r['plot']}"
            f"\n   Genre: {r['genre']}"
            for i, r in enumerate(results[:5])  # Top 5
        ])

        prompt = f"""User searched for: "{query}"

Here are the top matching movies:

{results_text}

Provide a brief 2-3 sentence summary highlighting:
1. Common themes across these recommendations
2. Why they match the user's search
3. A quick recommendation on which to watch first

Be conversational and helpful."""

        response = self.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            system="You are a helpful movie recommendation assistant."
        )

        return response['text']

    def enhance_query(self, query: str) -> List[str]:
        """
        Enhance user query with semantic variations

        Args:
            query: Original search query

        Returns:
            List of enhanced/related queries
        """
        prompt = f"""Given this movie search query: "{query}"

Generate 3 semantically similar queries that would help find relevant movies.
Consider:
- Similar themes and concepts
- Related genres and styles
- Alternative phrasings

Return ONLY the 3 queries, one per line, no numbering or explanation."""

        response = self.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=0.8
        )

        queries = [q.strip() for q in response['text'].strip().split('\n') if q.strip()]
        return queries[:3]

    def extract_preferences(self, query: str) -> Dict[str, Any]:
        """
        Extract search preferences from natural language query

        Args:
            query: User's natural language query

        Returns:
            Dict with genre, year_range, min_rating, themes
        """
        prompt = f"""Analyze this movie search query: "{query}"

Extract the following (if mentioned, otherwise return null):
- genre: Main genre (Action, Drama, Sci-Fi, etc.)
- min_year: Earliest year mentioned
- max_year: Latest year mentioned
- min_rating: Minimum rating (1-10 scale)
- themes: List of key themes/elements

Return ONLY a valid JSON object, no other text."""

        response = self.generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.3,
            system="You are a precise JSON extractor. Return only valid JSON."
        )

        try:
            return json.loads(response['text'])
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                'genre': None,
                'min_year': None,
                'max_year': None,
                'min_rating': None,
                'themes': []
            }


def test_bedrock_claude():
    """Test Bedrock Claude integration"""
    print("=" * 70)
    print("Testing Claude via AWS Bedrock")
    print("=" * 70)

    # List available models
    print("\n📋 Available Models:")
    for key, model_id in BedrockClaude.MODELS.items():
        default = " (default)" if key == BedrockClaude.DEFAULT_MODEL else ""
        print(f"  • {key}: {model_id}{default}")

    # Initialize with default model
    print("\n" + "=" * 70)
    print("Initializing Client")
    print("=" * 70)
    claude = BedrockClaude()

    # Test 1: Simple generation
    print("\n" + "=" * 70)
    print("Test 1: Simple Generation")
    print("=" * 70)
    response = claude.generate(
        prompt="In one sentence, what makes a great movie?",
        max_tokens=256
    )
    print(f"Response: {response['text']}")
    print(f"Tokens: {response['usage'].get('output_tokens', 'N/A')}")

    # Test 2: Query enhancement
    print("\n" + "=" * 70)
    print("Test 2: Query Enhancement")
    print("=" * 70)
    query = "epic space adventure"
    enhanced = claude.enhance_query(query)
    print(f"Original: '{query}'")
    print(f"Enhanced queries:")
    for i, q in enumerate(enhanced, 1):
        print(f"  {i}. {q}")

    # Test 3: Extract preferences
    print("\n" + "=" * 70)
    print("Test 3: Extract Preferences")
    print("=" * 70)
    query = "I want a highly rated sci-fi movie from the 2010s"
    prefs = claude.extract_preferences(query)
    print(f"Query: '{query}'")
    print(f"Extracted preferences:")
    print(json.dumps(prefs, indent=2))

    # Test 4: Summarize results
    print("\n" + "=" * 70)
    print("Test 4: Summarize Search Results")
    print("=" * 70)
    mock_results = [
        {
            'title': 'Interstellar',
            'year': 2014,
            'rating': 8.6,
            'plot': 'A team of explorers travel through a wormhole in space.',
            'genre': 'Sci-Fi'
        },
        {
            'title': 'The Matrix',
            'year': 1999,
            'rating': 8.7,
            'plot': 'A hacker discovers the true nature of reality.',
            'genre': 'Sci-Fi'
        },
        {
            'title': 'Inception',
            'year': 2010,
            'rating': 8.8,
            'plot': 'A thief enters dreams to steal secrets.',
            'genre': 'Sci-Fi'
        }
    ]
    summary = claude.summarize_results(mock_results, "mind-bending sci-fi")
    print(f"Summary:\n{summary}")

    print("\n" + "=" * 70)
    print("✓ All Tests Passed!")
    print("=" * 70)
    print("\nYour Bedrock Claude setup is working!")
    print("Models use your existing AWS credentials - no API key needed.")


if __name__ == "__main__":
    test_bedrock_claude()
