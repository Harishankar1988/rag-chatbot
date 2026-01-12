# Multi-stage build for PDF Chatbot RAG System

FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploaded data/vectors

# Expose ports
EXPOSE 5000 8501

# Default command
CMD ["python", "backend/app.py"]
