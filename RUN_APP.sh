#!/bin/bash
# Quick start script for the AI Movie Search application

echo "=========================================="
echo "🎬 AI Movie Search Application"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Make sure your environment variables are set:"
    echo "  - AWS_REGION"
    echo "  - QDRANT_URL"
    echo "  - QDRANT_API_KEY"
    echo ""
fi

echo "Select application to run:"
echo "1) Chainlit Chat UI (recommended)"
echo "2) FastAPI REST API"
echo "3) Interactive CLI Demo"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Chainlit Chat UI..."
        echo "   Access at: http://localhost:8000"
        echo ""
        chainlit run chainlit_app.py -w
        ;;
    2)
        echo ""
        echo "🚀 Starting FastAPI Server..."
        echo "   API Docs: http://localhost:8000/docs"
        echo "   Health: http://localhost:8000/health"
        echo ""
        python api.py
        ;;
    3)
        echo ""
        echo "🚀 Starting Interactive Demo..."
        echo ""
        python intelligent_search.py
        ;;
    *)
        echo "Invalid choice. Defaulting to Chainlit..."
        chainlit run chainlit_app.py -w
        ;;
esac
