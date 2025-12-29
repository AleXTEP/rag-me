#!/bin/bash
# Test PDF upload endpoint

if [ -z "$1" ]; then
    echo "Usage: ./test_upload.sh <path_to_pdf_file>"
    echo "Example: ./test_upload.sh sample.pdf"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found"
    exit 1
fi

echo "Uploading PDF: $1"
echo ""

RESPONSE=$(curl -s -X POST "http://localhost:8000/upload" \
    -F "file=@$1")

echo "$RESPONSE" | jq .

# Extract task_id if available
TASK_ID=$(echo "$RESPONSE" | jq -r '.task_id // empty')
FILE_ID=$(echo "$RESPONSE" | jq -r '.file_id // empty')

if [ ! -z "$TASK_ID" ]; then
    echo ""
    echo "Task ID: $TASK_ID"
    echo "File ID: $FILE_ID"
    echo ""
    echo "To check task status, run:"
    echo "  ./test_task_status.sh $TASK_ID"
fi

