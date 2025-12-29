#!/bin/bash
# Test task status endpoint

if [ -z "$1" ]; then
    echo "Usage: ./test_task_status.sh <task_id>"
    echo "Example: ./test_task_status.sh abc123-def456-..."
    exit 1
fi

TASK_ID=$1

echo "Checking status for task: $TASK_ID"
echo ""

while true; do
    RESPONSE=$(curl -s "http://localhost:8000/task/$TASK_ID")
    STATE=$(echo "$RESPONSE" | jq -r '.state // "UNKNOWN"')
    
    echo "$RESPONSE" | jq .
    echo ""
    
    if [ "$STATE" = "SUCCESS" ] || [ "$STATE" = "FAILURE" ]; then
        echo "Task completed with state: $STATE"
        break
    fi
    
    echo "Waiting 2 seconds before next check..."
    sleep 2
done

