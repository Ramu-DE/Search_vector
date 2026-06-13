#!/usr/bin/env python3
"""
Test script for Claude model configuration
Tests access to Claude Mythos 5 and falls back to best available model
"""

from model_config import ClaudeModelConfig


def main():
    print("=" * 60)
    print("Claude Model Configuration Test")
    print("=" * 60)

    # Get the configured model
    model = ClaudeModelConfig.get_model()
    print(f"\n📌 Default model: {model}")

    # Try to get a client and make a test call
    try:
        client = ClaudeModelConfig.get_client()
        print(f"\n✓ Anthropic client initialized")

        print(f"\n🧪 Testing {model}...")
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": "What model are you?"
            }]
        )

        print(f"\n✓ Response from {response.model}:")
        print(f"   {response.content[0].text[:200]}...")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n📝 To use Claude models:")
        print("   1. Get an API key from: https://console.anthropic.com/settings/keys")
        print("   2. Set it: export ANTHROPIC_API_KEY='your-key-here'")
        print("   3. For Claude Mythos 5, you need Project Glasswing access")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("Available Models (in preference order):")
    print("=" * 60)
    for i, model_id in enumerate(ClaudeModelConfig.MODEL_PREFERENCES, 1):
        status = "✓ Current" if model_id == ClaudeModelConfig.DEFAULT_MODEL else "○"
        print(f"{status} {i}. {model_id}")

        if model_id == "claude-mythos-5":
            print("      → Requires Project Glasswing access")
        elif model_id == "claude-opus-4-8":
            print("      → Most capable widely available model")
        elif model_id == "claude-sonnet-4-6":
            print("      → Best speed/intelligence balance")

    print("=" * 60)


if __name__ == "__main__":
    main()
