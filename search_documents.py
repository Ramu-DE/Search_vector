#!/usr/bin/env python3
"""
Document Search with Claude Summarization
Search ingested PDFs with AI-powered answers
"""

import boto3
import json
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from bedrock_claude import BedrockClaude
from config import Config
import sys


class DocumentSearch:
    """Search documents with AI summarization"""

    def __init__(self, collection_name: str = 'documents', claude_model: str = 'sonnet-4.6'):
        """Initialize document search"""
        # Bedrock for embeddings
        self.bedrock = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)

        # Qdrant for vector search
        config = Config.get_qdrant_config()
        self.qdrant = QdrantClient(url=config['url'], api_key=config['api_key'])
        self.collection = collection_name

        # Claude for AI answers
        self.claude = BedrockClaude(model=claude_model)

        print(f"✓ Document Search initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Claude: {claude_model}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Bedrock Titan"""
        body = json.dumps({"inputText": text})
        response = self.bedrock.invoke_model(
            modelId=f"{Config.BEDROCK_MODEL_ID}:0",
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response['body'].read())
        return np.array(response_body['embedding'])

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of matching document chunks
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Search Qdrant
        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_embedding.tolist(),
            limit=k
        ).points

        # Format results
        formatted = []
        for result in results:
            formatted.append({
                'text': result.payload['text'],
                'page': result.payload['page_num'],
                'source': result.payload['source'],
                'score': result.score
            })

        return formatted

    def answer_question(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using document search + Claude

        Args:
            question: User's question
            k: Number of document chunks to retrieve

        Returns:
            Dict with answer, sources, and context
        """
        print(f"\n🔍 Searching documents...")
        results = self.search(question, k=k)

        if not results:
            return {
                'answer': "I couldn't find relevant information in the documents.",
                'sources': [],
                'context': []
            }

        print(f"   Found {len(results)} relevant chunks")

        # Build context from search results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} - Page {result['page']}]\n{result['text']}"
            )

        context = "\n\n".join(context_parts)

        # Ask Claude to answer based on context
        print(f"\n🤖 Generating answer with Claude...")

        prompt = f"""Based on the following document excerpts, answer this question: "{question}"

Document Context:
{context}

Instructions:
- Answer the question directly and clearly
- Only use information from the provided context
- If the context doesn't contain the answer, say so
- Cite which source(s) you're using (e.g., "According to Source 1...")
- Be concise but complete

Answer:"""

        response = self.claude.generate(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.3,
            system="You are a helpful assistant that answers questions based on provided document context. Always cite your sources."
        )

        answer = response['text']

        return {
            'answer': answer,
            'sources': [
                {
                    'page': r['page'],
                    'source': r['source'],
                    'text': r['text'][:200] + "..." if len(r['text']) > 200 else r['text'],
                    'score': r['score']
                }
                for r in results
            ],
            'context': context
        }


def main():
    """CLI for document search"""
    if len(sys.argv) < 2:
        print("Usage: python search_documents.py '<question>' [collection_name]")
        print("\nExamples:")
        print("  python search_documents.py 'What is NVIDIA DGX Spark?'")
        print("  python search_documents.py 'Tell me about AI development challenges'")
        sys.exit(1)

    question = sys.argv[1]
    collection = sys.argv[2] if len(sys.argv) > 2 else 'documents'

    print("=" * 70)
    print("Document Search with AI")
    print("=" * 70)

    # Initialize search
    search = DocumentSearch(collection_name=collection)

    # Answer question
    result = search.answer_question(question, k=5)

    # Display results
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result['answer'])

    print("\n" + "=" * 70)
    print(f"SOURCES ({len(result['sources'])} documents)")
    print("=" * 70)

    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['source']} - Page {source['page']} (relevance: {source['score']:.3f})")
        print(f"   {source['text']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
