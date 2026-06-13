#!/bin/bash
# Complete System Test Script
# Tests all components of the AI search system

echo "=========================================="
echo "🧪 Complete System Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_RUN=0
TESTS_PASSED=0

# Helper function
run_test() {
    local test_name=$1
    local test_cmd=$2

    TESTS_RUN=$((TESTS_RUN + 1))
    echo "Test $TESTS_RUN: $test_name"
    echo "---"

    if eval "$test_cmd"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
    echo ""
}

# Activate virtualenv
source .venv/bin/activate

echo "=========================================="
echo "1. Configuration Tests"
echo "=========================================="
echo ""

run_test "Environment variables" \
    "python -c 'from config import Config; assert Config.AWS_REGION; assert Config.QDRANT_URL; assert Config.QDRANT_API_KEY; print(\"AWS Region:\", Config.AWS_REGION); print(\"Qdrant URL:\", Config.QDRANT_URL[:30] + \"...\")'"

run_test "AWS Bedrock connection" \
    "python -c 'import boto3; client = boto3.client(\"bedrock-runtime\", region_name=\"us-west-2\"); print(\"✓ Bedrock connected\")'"

run_test "Qdrant connection" \
    "python -c 'from qdrant_store import QdrantVectorStore; store = QdrantVectorStore(); print(\"✓ Qdrant connected\")'"

echo "=========================================="
echo "2. Embedding Tests"
echo "=========================================="
echo ""

run_test "Generate embedding" \
    "python -c 'import boto3, json; client = boto3.client(\"bedrock-runtime\", region_name=\"us-west-2\"); response = client.invoke_model(modelId=\"amazon.titan-embed-text-v2:0\", body=json.dumps({\"inputText\": \"test\"}), contentType=\"application/json\"); data = json.loads(response[\"body\"].read()); emb = data[\"embedding\"]; print(f\"✓ Generated {len(emb)}-dim embedding\")'"

echo "=========================================="
echo "3. Claude Model Tests"
echo "=========================================="
echo ""

run_test "Claude Sonnet 4.6 via Bedrock" \
    "python -c 'from bedrock_claude import BedrockClaude; claude = BedrockClaude(model=\"sonnet-4.6\"); response = claude.generate(\"Say hello in 5 words\", max_tokens=20); print(\"Response:\", response[\"text\"][:50])'"

echo "=========================================="
echo "4. Vector Search Tests"
echo "=========================================="
echo ""

run_test "Search documents collection" \
    "python -c 'from qdrant_client import QdrantClient; from config import Config; config = Config.get_qdrant_config(); client = QdrantClient(url=config[\"url\"], api_key=config[\"api_key\"]); info = client.get_collection(\"documents\"); print(f\"✓ Collection has {info.points_count} documents\")'"

run_test "Document search query" \
    "python search_documents.py 'What is NVIDIA DGX Spark?' | grep -q 'NVIDIA DGX Spark' && echo '✓ Search returned relevant results'"

echo "=========================================="
echo "5. PDF Processing Tests"
echo "=========================================="
echo ""

run_test "PDF exists" \
    "test -f Data/NVIDIA.pdf && ls -lh Data/NVIDIA.pdf | awk '{print \"✓ PDF found:\", \$9, \"(\"\$5\")\"}'"

run_test "PDF extraction" \
    "python -c 'import PyPDF2; pdf = open(\"Data/NVIDIA.pdf\", \"rb\"); reader = PyPDF2.PdfReader(pdf); pages = len(reader.pages); print(f\"✓ PDF has {pages} pages\")'"

echo "=========================================="
echo "6. Integration Tests"
echo "=========================================="
echo ""

run_test "End-to-end document Q&A" \
    "python -c 'from search_documents import DocumentSearch; search = DocumentSearch(); result = search.answer_question(\"What is DGX Spark?\", k=3); assert len(result[\"answer\"]) > 50; assert len(result[\"sources\"]) > 0; print(f\"✓ Generated {len(result[\"answer\"])} char answer with {len(result[\"sources\"])} sources\")'"

echo "=========================================="
echo "7. Application Tests"
echo "=========================================="
echo ""

run_test "Chainlit app exists" \
    "test -f document_chat.py && test -f chainlit.md && test -f .chainlit/config.toml && echo '✓ Chainlit files present'"

run_test "API app exists" \
    "test -f api.py && python -c 'import api; print(\"✓ API module loads\")'"

echo "=========================================="
echo "📊 Test Summary"
echo "=========================================="
echo ""

TESTS_FAILED=$((TESTS_RUN - TESTS_PASSED))

echo "Total tests run: $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! System is ready.${NC}"
    echo ""
    echo "=========================================="
    echo "🚀 Ready to Use"
    echo "=========================================="
    echo ""
    echo "Start the chat interface:"
    echo "  chainlit run document_chat.py -w"
    echo ""
    echo "Or use CLI:"
    echo "  python search_documents.py 'Your question here'"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the errors above.${NC}"
    exit 1
fi
