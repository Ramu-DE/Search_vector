"""
Claude Model Configuration
Manages Claude API model selection with automatic fallback
"""
import os
from typing import Optional

class ClaudeModelConfig:
    """Configuration for Claude API models (separate from Bedrock embeddings)"""

    # Default model preferences (in order of preference)
    MODEL_PREFERENCES = [
        "claude-mythos-5",      # Most capable (requires Project Glasswing)
        "claude-opus-4-8",      # Current best Opus
        "claude-sonnet-4-6",    # Best balance
    ]

    # Active model (set during initialization)
    DEFAULT_MODEL: Optional[str] = None

    # API Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')

    @classmethod
    def test_model_access(cls, model_id: str) -> bool:
        """
        Test if a specific model is accessible

        Args:
            model_id: Model ID to test (e.g., 'claude-mythos-5')

        Returns:
            bool: True if model is accessible, False otherwise
        """
        if not cls.ANTHROPIC_API_KEY:
            return False

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=cls.ANTHROPIC_API_KEY)

            response = client.messages.create(
                model=model_id,
                max_tokens=64,
                messages=[{"role": "user", "content": "test"}]
            )
            return True

        except anthropic.NotFoundError:
            return False  # Model not found
        except anthropic.PermissionDeniedError:
            return False  # No permission
        except anthropic.AuthenticationError:
            return False  # Invalid API key
        except Exception:
            return False  # Other error

    @classmethod
    def initialize_default_model(cls) -> str:
        """
        Initialize and return the best available model

        Returns:
            str: Model ID of the best available model
        """
        if cls.DEFAULT_MODEL:
            return cls.DEFAULT_MODEL

        if not cls.ANTHROPIC_API_KEY:
            print("⚠️  No ANTHROPIC_API_KEY found. Set it to use Claude models.")
            print("   For now, using claude-opus-4-8 as default (will require key at runtime)")
            cls.DEFAULT_MODEL = "claude-opus-4-8"
            return cls.DEFAULT_MODEL

        print("🔍 Testing Claude model access...")

        for model_id in cls.MODEL_PREFERENCES:
            print(f"   Testing {model_id}...", end=" ")
            if cls.test_model_access(model_id):
                print("✓ Available")
                cls.DEFAULT_MODEL = model_id
                print(f"\n✓ Default model set to: {model_id}")
                return cls.DEFAULT_MODEL
            else:
                print("✗ Not available")

        # Fallback to Opus 4.8 if nothing works
        cls.DEFAULT_MODEL = "claude-opus-4-8"
        print(f"\n⚠️  Using fallback: {cls.DEFAULT_MODEL}")
        return cls.DEFAULT_MODEL

    @classmethod
    def get_model(cls, override: Optional[str] = None) -> str:
        """
        Get the model to use

        Args:
            override: Optional model ID to override default

        Returns:
            str: Model ID to use
        """
        if override:
            return override

        if not cls.DEFAULT_MODEL:
            cls.initialize_default_model()

        return cls.DEFAULT_MODEL

    @classmethod
    def get_client(cls):
        """
        Get authenticated Anthropic client

        Returns:
            anthropic.Anthropic: Authenticated client

        Raises:
            ValueError: If no API key is configured
        """
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Set it as an environment variable or in .env file"
            )

        try:
            import anthropic
            return anthropic.Anthropic(api_key=cls.ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")


# Auto-initialize on import (only if API key is present)
if ClaudeModelConfig.ANTHROPIC_API_KEY:
    ClaudeModelConfig.initialize_default_model()
else:
    ClaudeModelConfig.DEFAULT_MODEL = "claude-opus-4-8"
