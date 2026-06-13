#!/usr/bin/env python3
"""
PDF Document Ingestion Pipeline
Processes PDFs, generates embeddings, and stores in Qdrant
"""

import boto3
import json
import numpy as np
from typing import List, Dict, Any
import re
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import Config
import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page numbers

    Returns:
        List of dicts with page_num and text
    """
    print(f"📄 Extracting text from: {pdf_path}")

    pages = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        print(f"   Total pages: {total_pages}")

        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            if text.strip():  # Only add non-empty pages
                pages.append({
                    'page_num': page_num + 1,
                    'text': text.strip()
                })

    print(f"   ✓ Extracted {len(pages)} pages with content")
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Split by sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_pdf_to_chunks(
    pdf_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Process PDF into searchable chunks with metadata

    Returns:
        List of chunks with metadata
    """
    pages = extract_text_from_pdf(pdf_path)

    all_chunks = []
    chunk_id = 0

    print(f"\n📝 Creating chunks (size={chunk_size}, overlap={overlap})...")

    for page in pages:
        page_chunks = chunk_text(page['text'], chunk_size, overlap)

        for chunk_content in page_chunks:
            if len(chunk_content) > 50:  # Only meaningful chunks
                all_chunks.append({
                    'id': chunk_id,
                    'text': chunk_content,
                    'page_num': page['page_num'],
                    'source': Path(pdf_path).name,
                    'chunk_size': len(chunk_content)
                })
                chunk_id += 1

    print(f"   ✓ Created {len(all_chunks)} chunks")
    return all_chunks


def generate_embedding(text: str, bedrock_client) -> List[float]:
    """Generate embedding using AWS Bedrock Titan"""
    body = json.dumps({"inputText": text})

    response = bedrock_client.invoke_model(
        modelId=f"{Config.BEDROCK_MODEL_ID}:0",
        body=body,
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    return response_body['embedding']


def ingest_pdf_to_qdrant(
    pdf_path: str,
    collection_name: str = 'documents',
    chunk_size: int = 500,
    recreate_collection: bool = False
):
    """
    Complete PDF ingestion pipeline

    Args:
        pdf_path: Path to PDF file
        collection_name: Qdrant collection name
        chunk_size: Size of text chunks
        recreate_collection: Whether to recreate the collection
    """
    print("=" * 70)
    print("PDF Document Ingestion Pipeline")
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print(f"Collection: {collection_name}")
    print(f"Chunk size: {chunk_size}")
    print()

    # 1. Process PDF
    chunks = process_pdf_to_chunks(pdf_path, chunk_size=chunk_size)

    if not chunks:
        print("❌ No chunks extracted from PDF!")
        return False

    # 2. Initialize clients
    print("\n🔌 Connecting to services...")

    # Bedrock for embeddings
    bedrock = boto3.client('bedrock-runtime', region_name=Config.AWS_REGION)
    print("   ✓ AWS Bedrock connected")

    # Qdrant
    config = Config.get_qdrant_config()
    qdrant = QdrantClient(url=config['url'], api_key=config['api_key'])
    print("   ✓ Qdrant connected")

    # 3. Create/recreate collection
    print(f"\n📦 Setting up collection: {collection_name}")

    try:
        collections = qdrant.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists and recreate_collection:
            print(f"   Deleting existing collection...")
            qdrant.delete_collection(collection_name)
            exists = False

        if not exists:
            print(f"   Creating new collection...")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=Config.BEDROCK_EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f"   ✓ Collection created")
        else:
            print(f"   ✓ Using existing collection")

    except Exception as e:
        print(f"   ❌ Collection setup failed: {e}")
        return False

    # 4. Generate embeddings and upload
    print(f"\n🧮 Generating embeddings for {len(chunks)} chunks...")

    points = []
    batch_size = 10

    for i, chunk in enumerate(tqdm(chunks, desc="   Processing")):
        try:
            # Generate embedding
            embedding = generate_embedding(chunk['text'], bedrock)

            # Create point
            point = PointStruct(
                id=chunk['id'],
                vector=embedding,
                payload={
                    'text': chunk['text'],
                    'page_num': chunk['page_num'],
                    'source': chunk['source'],
                    'chunk_size': chunk['chunk_size']
                }
            )
            points.append(point)

            # Upload in batches
            if len(points) >= batch_size:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=points
                )
                points = []

        except Exception as e:
            print(f"\n   ⚠️  Error processing chunk {i}: {e}")
            continue

    # Upload remaining points
    if points:
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"   ✓ Uploaded {len(chunks)} document chunks")

    # 5. Verify
    print(f"\n✅ Verifying collection...")
    info = qdrant.get_collection(collection_name)
    print(f"   Points in collection: {info.points_count}")
    print(f"   Status: {info.status}")

    # 6. Test search
    print(f"\n🔍 Testing search...")
    test_query = "What is NVIDIA DGX Spark?"
    test_embedding = generate_embedding(test_query, bedrock)

    results = qdrant.query_points(
        collection_name=collection_name,
        query=test_embedding,
        limit=3
    ).points

    print(f"   Query: '{test_query}'")
    print(f"   Top results:")
    for i, result in enumerate(results, 1):
        preview = result.payload['text'][:100] + "..." if len(result.payload['text']) > 100 else result.payload['text']
        print(f"   {i}. Page {result.payload['page_num']} (score: {result.score:.3f})")
        print(f"      {preview}")

    print("\n" + "=" * 70)
    print("✅ Ingestion Complete!")
    print("=" * 70)
    print(f"✓ Processed: {pdf_path}")
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Chunks: {len(chunks)}")
    print(f"✓ Dimensions: {Config.BEDROCK_EMBEDDING_DIMENSION}")

    return True


def main():
    """Main ingestion workflow"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <pdf_path> [collection_name] [--recreate]")
        print("\nExample:")
        print("  python ingest_pdf.py Data/NVIDIA.pdf documents --recreate")
        sys.exit(1)

    pdf_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'documents'
    recreate = '--recreate' in sys.argv

    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    success = ingest_pdf_to_qdrant(
        pdf_path=pdf_path,
        collection_name=collection_name,
        chunk_size=500,
        recreate_collection=recreate
    )

    if success:
        print(f"\n🎉 Ready to search!")
        print(f"   Try: python search_documents.py 'What is NVIDIA DGX Spark?'")
        print(f"   Or: chainlit run document_chat.py -w")
    else:
        print(f"\n❌ Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
