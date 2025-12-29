#!/bin/bash
# Run all tests

echo "=========================================="
echo "RAG System Test Suite"
echo "=========================================="
echo ""

# Test 1: Health check
echo "1. Testing health endpoint..."
./test_health.sh
echo ""

# Test 2: Check if we have a test PDF
if [ -z "$1" ]; then
    echo "2. Upload test skipped - no PDF file provided"
    echo "   Usage: ./test_all.sh <path_to_pdf_file>"
    echo ""
else
    echo "2. Testing PDF upload..."
    ./test_upload.sh "$1"
    echo ""
    
    # Extract task_id from upload response
    TASK_ID=$(curl -s -X POST "http://localhost:8000/upload" \
        -F "file=@$1" | jq -r '.task_id // empty')
    
    if [ ! -z "$TASK_ID" ]; then
        echo "3. Testing task status (will monitor until completion)..."
        ./test_task_status.sh "$TASK_ID"
        echo ""
    fi
fi

echo "=========================================="
echo "Tests completed!"
echo "=========================================="

