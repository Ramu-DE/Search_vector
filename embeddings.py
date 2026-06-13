"""
Embedding generation utilities
Supports both Amazon Bedrock and local models
"""
import json
import boto3
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config import Config


class EmbeddingGenerator:
    """Base class for embedding generation"""

    def generate(self, text: str) -> List[float]:
        """Generate embedding for text"""
        raise NotImplementedError

    def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.generate(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError


class BedrockEmbedding(EmbeddingGenerator):
    """
    Amazon Bedrock embedding generator

    Latest Models:
    - amazon.titan-embed-text-v2 (1024d) - Latest, best quality
    - amazon.titan-embed-text-v1 (1536d) - Previous version
    - cohere.embed-english-v3 (1024d) - Alternative
    - cohere.embed-multilingual-v3 (1024d) - Multilingual support
    """

    def __init__(self, model_id: str = None, region: str = None):
        """
        Initialize Bedrock client

        Args:
            model_id: Bedrock model ID (default: amazon.titan-embed-text-v2)
            region: AWS region
        """
        self.model_id = model_id or Config.BEDROCK_MODEL_ID
        self.region = region or Config.AWS_REGION

        # Set dimension based on model
        if 'titan-embed-text-v2' in self.model_id:
            self._dimension = 1024
        elif 'titan-embed-text-v1' in self.model_id:
            self._dimension = 1536
        elif 'cohere.embed' in self.model_id:
            self._dimension = 1024
        else:
            self._dimension = Config.BEDROCK_EMBEDDING_DIMENSION

        # Initialize Bedrock client
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.region
        )

        print(f"Initialized Bedrock embedding with model: {self.model_id}")

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding using Amazon Bedrock

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({'inputText': text})
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            print(f"Error generating Bedrock embedding: {e}")
            raise

    def batch_generate(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """
        Generate embeddings in batches

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.generate(text) for text in batch]
            embeddings.extend(batch_embeddings)

        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(EmbeddingGenerator):
    """
    Local embedding generator using sentence-transformers

    Latest Recommended Models (2024):
    - all-mpnet-base-v2 (768d) - Best quality, recommended default
    - all-MiniLM-L12-v2 (384d) - Balanced speed/quality
    - all-MiniLM-L6-v2 (384d) - Fastest, good quality
    - multi-qa-mpnet-base-dot-v1 (768d) - Best for Q&A/search
    - paraphrase-multilingual-mpnet-base-v2 (768d) - Multilingual

    Specialized Models:
    - msmarco-distilbert-base-v4 (768d) - Optimized for search/retrieval
    - sentence-t5-large (768d) - Latest T5-based model
    """

    def __init__(self, model_name: str = None):
        """
        Initialize local embedding model

        Args:
            model_name: Sentence-transformers model name (default: all-mpnet-base-v2)
        """
        self.model_name = model_name or Config.LOCAL_MODEL_NAME

        print(f"Loading local model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

        print(f"Loaded model with dimension: {self._dimension}")

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding using local model

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def batch_generate(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings in batches (optimized)

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_generator(use_bedrock: bool = True) -> EmbeddingGenerator:
    """
    Factory function to get embedding generator

    Args:
        use_bedrock: If True, use Bedrock; otherwise use local model

    Returns:
        EmbeddingGenerator instance
    """
    if use_bedrock:
        try:
            return BedrockEmbedding()
        except Exception as e:
            print(f"Failed to initialize Bedrock: {e}")
            print("Falling back to local model")
            return LocalEmbedding()
    else:
        return LocalEmbedding()


# Example usage and testing
if __name__ == "__main__":
    # Test local embeddings
    print("\n=== Testing Local Embeddings ===")
    local_gen = LocalEmbedding()

    test_text = "This is a test sentence for embedding generation"
    embedding = local_gen.generate(test_text)

    print(f"Text: {test_text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")

    # Test batch generation
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]
    batch_embeddings = local_gen.batch_generate(texts)
    print(f"\nBatch embeddings generated: {len(batch_embeddings)}")

    # Optionally test Bedrock (if configured)
    # print("\n=== Testing Bedrock Embeddings ===")
    # bedrock_gen = BedrockEmbedding()
    # bedrock_embedding = bedrock_gen.generate(test_text)
    # print(f"Bedrock embedding dimension: {len(bedrock_embedding)}")
