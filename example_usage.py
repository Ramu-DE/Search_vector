#!/usr/bin/env python3
"""
Example: Using Claude Models in Your Search Application

This demonstrates how to integrate Claude models with your vector search
"""

from model_config import ClaudeModelConfig


def example_query_enhancement():
    """Example: Use Claude to enhance user queries"""
    try:
        # Get the best available Claude model
        model = ClaudeModelConfig.get_model()
        client = ClaudeModelConfig.get_client()

        user_query = "movies about space"

        # Use Claude to expand and enhance the query
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Given this search query: "{user_query}"

Generate 3 semantically similar queries that would help find relevant movies.
Return only the queries, one per line, no numbering or explanation."""
            }]
        )

        enhanced_queries = response.content[0].text.strip().split('\n')
        print(f"Original query: {user_query}")
        print(f"Enhanced queries ({model}):")
        for q in enhanced_queries:
            print(f"  - {q.strip()}")

        return enhanced_queries

    except ValueError as e:
        print(f"⚠️  {e}")
        print("Set ANTHROPIC_API_KEY to use Claude models")
        return [user_query]  # Return original query as fallback


def example_result_summarization():
    """Example: Use Claude to summarize search results"""
    try:
        model = ClaudeModelConfig.get_model()
        client = ClaudeModelConfig.get_client()

        # Mock search results
        results = [
            {"title": "Interstellar", "rating": 8.6, "plot": "Space exploration drama"},
            {"title": "Gravity", "rating": 7.7, "plot": "Astronaut survival story"},
            {"title": "The Martian", "rating": 8.0, "plot": "Mars survival mission"},
        ]

        # Use Claude to generate a natural language summary
        results_text = "\n".join([
            f"{r['title']} (Rating: {r['rating']}) - {r['plot']}"
            for r in results
        ])

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": f"""Summarize these movie search results in 2-3 sentences:

{results_text}

Focus on common themes and recommendations."""
            }]
        )

        summary = response.content[0].text
        print(f"\nSearch Results Summary ({model}):")
        print(summary)

        return summary

    except ValueError as e:
        print(f"⚠️  {e}")
        return "Search returned multiple space-themed movies."


def example_semantic_understanding():
    """Example: Use Claude to understand complex queries"""
    try:
        model = ClaudeModelConfig.get_model()
        client = ClaudeModelConfig.get_client()

        complex_query = "I want something like Inception but less confusing"

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""User query: "{complex_query}"

Extract:
1. Genre preferences
2. Style preferences
3. What to avoid
4. Suggested movie attributes to search for

Be concise, one line each."""
            }]
        )

        analysis = response.content[0].text
        print(f"\nQuery Analysis ({model}):")
        print(analysis)

        return analysis

    except ValueError as e:
        print(f"⚠️  {e}")
        return None


def main():
    print("=" * 60)
    print("Claude Model Integration Examples")
    print("=" * 60)

    print("\n📌 Current Configuration:")
    print(f"   Model: {ClaudeModelConfig.get_model()}")
    print(f"   API Key: {'✓ Set' if ClaudeModelConfig.ANTHROPIC_API_KEY else '✗ Not set'}")

    if not ClaudeModelConfig.ANTHROPIC_API_KEY:
        print("\n⚠️  No API key found. Set ANTHROPIC_API_KEY to run examples.")
        print("   Example: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    print("\n" + "=" * 60)
    print("Example 1: Query Enhancement")
    print("=" * 60)
    example_query_enhancement()

    print("\n" + "=" * 60)
    print("Example 2: Result Summarization")
    print("=" * 60)
    example_result_summarization()

    print("\n" + "=" * 60)
    print("Example 3: Semantic Understanding")
    print("=" * 60)
    example_semantic_understanding()


if __name__ == "__main__":
    main()
