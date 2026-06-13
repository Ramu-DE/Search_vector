"""
Streamlit application for AI-powered movie search
Provides interactive UI for comparing search methods
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import time

from config import Config
from search import VectorSearchEngine
from embeddings import get_embedding_generator

# Page configuration
st.set_page_config(
    page_title="AI-Powered Movie Search",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def initialize_search_engine():
    """Initialize and cache search engine"""
    try:
        embedding_gen = get_embedding_generator(use_bedrock=False)
        search_engine = VectorSearchEngine(embedding_generator=embedding_gen)
        return search_engine
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        return None


def display_results(results: List[Dict[str, Any]], method_name: str):
    """Display search results in a nice format"""
    if not results:
        st.warning(f"No results found for {method_name}")
        return

    st.subheader(f"🎯 {method_name} Results")

    for i, result in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                # Title and year
                st.markdown(f"### {i}. {result.get('title', 'N/A')} ({result.get('year', 'N/A')})")

                # Plot
                plot = result.get('plot', 'No plot available')
                st.write(plot)

                # Genre
                genres = result.get('genre', [])
                if genres:
                    genre_badges = ' '.join([f"`{g}`" for g in genres])
                    st.markdown(f"**Genres:** {genre_badges}")

            with col2:
                # Rating
                rating = result.get('rating', 0)
                st.metric("Rating", f"⭐ {rating}")

                # Score
                score = result.get('_score', 0)
                st.metric("Relevance", f"{score:.4f}")

            st.divider()


def main():
    """Main application"""

    # Header
    st.title("🎬 AI-Powered Movie Search")
    st.markdown("""
    Compare different search methods: **Keyword** (BM25), **Semantic** (k-NN), and **Hybrid** (Combined)
    """)

    # Initialize search engine
    search_engine = initialize_search_engine()
    if not search_engine:
        st.error("Could not initialize search engine. Please check your OpenSearch configuration.")
        st.info("Set the `AOSS_VECTORSEARCH_ENDPOINT` environment variable")
        return

    # Sidebar for configuration
    st.sidebar.header("⚙️ Search Configuration")

    # Query input
    query = st.sidebar.text_input(
        "Search Query",
        placeholder="e.g., movie to watch with friends",
        help="Enter your search query"
    )

    # Number of results
    k = st.sidebar.slider("Number of Results", min_value=1, max_value=20, value=5)

    # Filters
    st.sidebar.subheader("Filters")
    use_rating_filter = st.sidebar.checkbox("Filter by Rating")
    min_rating = 7.0
    if use_rating_filter:
        min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 7.0, 0.1)

    # Search method
    st.sidebar.subheader("Search Method")
    search_method = st.sidebar.radio(
        "Choose Method",
        ["Keyword (BM25)", "Semantic (k-NN)", "Hybrid", "Compare All"],
        index=3
    )

    # Hybrid search weight
    semantic_weight = 0.6
    if "Hybrid" in search_method:
        semantic_weight = st.sidebar.slider(
            "Semantic Weight",
            0.0, 1.0, 0.6, 0.1,
            help="Weight for semantic search (0=keyword only, 1=semantic only)"
        )

    # Search button
    search_button = st.sidebar.button("🔍 Search", type="primary", use_container_width=True)

    # Main content area
    if search_button and query:
        with st.spinner("Searching..."):
            # Build filters
            filters = {}
            if use_rating_filter:
                filters['rating'] = {'gte': min_rating}

            # Execute search based on method
            if search_method == "Keyword (BM25)":
                results = search_engine.keyword_search(query, k, filters)
                display_results(results, "Keyword Search")

            elif search_method == "Semantic (k-NN)":
                results = search_engine.semantic_search(query, k, filters)
                display_results(results, "Semantic Search")

            elif search_method == "Hybrid":
                results = search_engine.hybrid_search(
                    query, k, semantic_weight=semantic_weight, filters=filters
                )
                display_results(results, "Hybrid Search")

            elif search_method == "Compare All":
                # Create tabs for comparison
                tab1, tab2, tab3 = st.tabs(["Keyword", "Semantic", "Hybrid"])

                with tab1:
                    keyword_results = search_engine.keyword_search(query, k, filters)
                    display_results(keyword_results, "Keyword Search")

                with tab2:
                    semantic_results = search_engine.semantic_search(query, k, filters)
                    display_results(semantic_results, "Semantic Search")

                with tab3:
                    hybrid_results = search_engine.hybrid_search(
                        query, k, semantic_weight=semantic_weight, filters=filters
                    )
                    display_results(hybrid_results, "Hybrid Search")

                # Show comparison table
                st.subheader("📊 Score Comparison")

                # Create comparison dataframe
                comparison_data = []
                for i in range(min(len(keyword_results), len(semantic_results), len(hybrid_results))):
                    comparison_data.append({
                        'Rank': i + 1,
                        'Keyword': keyword_results[i].get('title', 'N/A') if i < len(keyword_results) else '-',
                        'K_Score': f"{keyword_results[i]['_score']:.3f}" if i < len(keyword_results) else '-',
                        'Semantic': semantic_results[i].get('title', 'N/A') if i < len(semantic_results) else '-',
                        'S_Score': f"{semantic_results[i]['_score']:.3f}" if i < len(semantic_results) else '-',
                        'Hybrid': hybrid_results[i].get('title', 'N/A') if i < len(hybrid_results) else '-',
                        'H_Score': f"{hybrid_results[i]['_score']:.3f}" if i < len(hybrid_results) else '-',
                    })

                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

    elif search_button and not query:
        st.warning("Please enter a search query")

    else:
        # Show example queries
        st.info("👈 Enter a search query in the sidebar to get started!")

        st.subheader("💡 Example Queries")
        example_queries = [
            "movie to watch with friends",
            "uplifting underdog story",
            "crime thriller with plot twists",
            "sci-fi action adventure",
            "emotional drama about family",
            "funny comedy for date night"
        ]

        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                st.markdown(f"- `{example}`")

    # Footer with information
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### About
    This application demonstrates AI-powered search using:
    - **OpenSearch Serverless** for vector storage
    - **Sentence Transformers** for embeddings
    - **HNSW algorithm** for fast k-NN search

    ### Search Methods
    - **Keyword**: Traditional BM25 matching
    - **Semantic**: Vector similarity (k-NN)
    - **Hybrid**: Combines both methods
    """)


if __name__ == "__main__":
    main()
