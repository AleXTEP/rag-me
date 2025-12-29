#!/bin/bash
# Test health endpoint

echo "Testing API health endpoint..."
curl -s http://localhost:8000/health | jq .
echo ""

