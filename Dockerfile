# Dockerfile for H100 GPU Deployment
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements
COPY requirements.txt .
COPY requirements_api.txt .

# Install Python packages
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements_api.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 8001

# Set environment variables for optimal H100 performance
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the API server    `
CMD ["python3", "api_server.py"]

