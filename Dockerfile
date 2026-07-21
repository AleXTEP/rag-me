FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than CUDA version)
# This prevents sentence-transformers from pulling the large CUDA dependencies
# PyTorch >= 2.6 required for torch.distributed.tensor.DTensor, imported by the
# transformers version docling pulls in
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose.yml)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

