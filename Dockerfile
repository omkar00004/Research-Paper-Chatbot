# Use a lightweight Python 3.11 base image
FROM python:3.11-slim

# HuggingFace Spaces runs as a non-root user (uid 1000)
# Create a working directory
WORKDIR /app

# Install system-level dependencies needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p logs uploads chroma_db "Research Paper"

# HuggingFace Spaces exposes port 7860 by default
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
