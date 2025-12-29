# Testing Guide

## Quick Start

1. **Make scripts executable:**
   ```bash
   chmod +x test_*.sh
   ```

2. **Check if services are running:**
   ```bash
   ./test_services.sh
   ```

3. **Test health endpoint:**
   ```bash
   ./test_health.sh
   ```

4. **Upload a PDF file:**
   ```bash
   ./test_upload.sh path/to/your/file.pdf
   ```

5. **Check task status:**
   ```bash
   ./test_task_status.sh <task_id>
   ```

6. **Run all tests:**
   ```bash
   ./test_all.sh path/to/your/file.pdf
   ```

## Test Scripts

### `test_services.sh`
Checks if all Docker services (Weaviate, Redis, API) are running and healthy.

### `test_health.sh`
Tests the API health endpoint.

### `test_upload.sh <pdf_file>`
Uploads a PDF file and returns the task ID and file ID.

### `test_task_status.sh <task_id>`
Monitors a task until it completes (SUCCESS or FAILURE).

### `test_all.sh <pdf_file>`
Runs all tests in sequence.

## Example Workflow

```bash
# 1. Start services
docker-compose up -d

# 2. Wait for services to be ready (30-60 seconds)
sleep 30

# 3. Check services
./test_services.sh

# 4. Upload a PDF
./test_upload.sh sample.pdf

# 5. Monitor the task (replace with actual task_id from step 4)
./test_task_status.sh abc123-def456-ghi789
```

## Requirements

- `jq` must be installed for JSON formatting:
  ```bash
  brew install jq  # macOS
  ```

- Services must be running:
  ```bash
  docker-compose up -d
  ```

