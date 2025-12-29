#!/bin/bash
# Test search/query endpoint

if [ -z "$1" ]; then
    echo "Usage: ./test_query_keywords.sh <search_query> [limit]"
    echo "Example: ./test_query_keywords.sh 'machine learning' 5"
    exit 1
fi

QUERY="$1"
LIMIT="${2:-5}"

echo "Searching for: '$QUERY'"
echo "Limit: $LIMIT results"
echo ""

RESPONSE=$(curl -s -X POST "http://localhost:8000/search/keywords" \
    -H "Content-Type: application/json" \
    -d "{\"q\": \"$QUERY\", \"limit\": $LIMIT}")

echo "$RESPONSE" | jq .

# Count results
RESULT_COUNT=$(echo "$RESPONSE" | jq '.objects | length')
echo ""
echo "Found $RESULT_COUNT results"

