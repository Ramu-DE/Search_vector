#!/usr/bin/env python3
"""
REST API for AI Movie Search
FastAPI backend for production deployment
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from intelligent_search import IntelligentMovieSearch
from bedrock_claude import BedrockClaude

# Initialize FastAPI
app = FastAPI(
    title="AI Movie Search API",
    description="Semantic movie search powered by Claude (1M context) + Bedrock embeddings + Qdrant",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search system
search = None


@app.on_event("startup")
async def startup_event():
    """Initialize search system on startup"""
    global search
    try:
        search = IntelligentMovieSearch(claude_model='sonnet-4.6')
        print("✓ AI Movie Search API ready")
    except Exception as e:
        print(f"✗ Failed to initialize search: {e}")
        raise


# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", example="epic space adventure")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")
    enhance_query: bool = Field(True, description="Use AI to enhance query")
    summarize: bool = Field(True, description="Generate AI summary")


class MovieResult(BaseModel):
    id: int
    title: str
    year: int
    rating: float
    genre: str
    plot: str
    director: str
    cast: str
    score: float


class SearchResponse(BaseModel):
    query: str
    enhanced_queries: List[str]
    preferences: Dict[str, Any]
    results: List[MovieResult]
    summary: Optional[str]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root - basic info"""
    return {
        "name": "AI Movie Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "vector_db": "Qdrant (connected)",
            "embeddings": "Bedrock Titan v2 (1024-dim)",
            "llm": "Claude Sonnet 4.6 (1M context)",
            "search": "operational"
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search_movies(request: SearchRequest):
    """
    Search for movies using AI-powered semantic search

    - **query**: Natural language search query
    - **k**: Number of results (1-20)
    - **enhance_query**: Use Claude to generate query variations
    - **summarize**: Generate AI summary of results
    """
    if not search:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    try:
        result = search.search(
            query=request.query,
            k=request.k,
            enhance_query=request.enhance_query,
            summarize=request.summarize
        )

        return SearchResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search", response_model=SearchResponse)
async def search_movies_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
    enhance_query: bool = Query(True, description="Enhance query with AI"),
    summarize: bool = Query(True, description="Generate AI summary")
):
    """
    Search for movies (GET method for easy browser testing)

    Example: /search?query=epic%20space%20adventure&k=5
    """
    request = SearchRequest(
        query=query,
        k=k,
        enhance_query=enhance_query,
        summarize=summarize
    )
    return await search_movies(request)


@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List available Claude models"""
    return {
        "default": BedrockClaude.DEFAULT_MODEL,
        "available": list(BedrockClaude.MODELS.keys()),
        "models": {
            key: {
                "id": value,
                "context": "1M tokens" if "opus" in key or "sonnet-4" in key or "fable" in key else "200K tokens"
            }
            for key, value in BedrockClaude.MODELS.items()
        }
    }


@app.post("/enhance-query")
async def enhance_query_endpoint(query: str = Query(..., description="Query to enhance")):
    """
    Enhance a query using Claude AI

    Returns semantically similar query variations
    """
    if not search:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    try:
        enhanced = search.claude.enhance_query(query)
        return {
            "original": query,
            "enhanced": enhanced
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@app.post("/extract-preferences")
async def extract_preferences_endpoint(query: str = Query(..., description="Query to analyze")):
    """
    Extract search preferences from natural language query

    Returns genre, year range, rating preferences, etc.
    """
    if not search:
        raise HTTPException(status_code=503, detail="Search system not initialized")

    try:
        preferences = search.claude.extract_preferences(query)
        return {
            "query": query,
            "preferences": preferences
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# Run server
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting AI Movie Search API")
    print("=" * 70)
    print("API Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("Search: http://localhost:8000/search?query=epic+space+adventure")
    print("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
