#!/bin/bash
# H100 API Deployment Script
# Deploy trained models as API with Docker

set -e

echo "🐳 H100 API Deployment"
echo "======================"

# Stop existing container if running
echo "🛑 Stopping existing containers..."
docker stop llm-api-h100 2>/dev/null || true
docker rm llm-api-h100 2>/dev/null || true

# Build H100 optimized Docker image
echo "🔨 Building H100 Docker image..."
docker build -f Dockerfile.h100 -t llm-api-h100:latest .

# Run H100 API container
echo "🚀 Starting H100 API container..."
docker run -d \
    --name llm-api-h100 \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/datasets:/app/datasets \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_ALLOC_CONF=max_split_size_mb:512 \
    -e TOKENIZERS_PARALLELISM=false \
    --restart unless-stopped \
    llm-api-h100:latest

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 10

# Check container status
echo "📊 Container status:"
docker ps | grep llm-api-h100

# Test API health
echo "🧪 Testing API health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed"
    echo "📝 Container logs:"
    docker logs llm-api-h100 --tail 20
    exit 1
fi

echo ""
echo "🎉 H100 API Deployment Complete!"
echo "================================"
echo ""
echo "🌐 API Endpoints:"
echo "  Health: http://localhost:8000/health"
echo "  Docs: http://localhost:8000/docs"
echo "  Models: http://localhost:8000/api/models"
echo ""
echo "🧪 Test with: ./test_all_models.sh"
echo ""
echo "📊 Monitor with:"
echo "  docker logs -f llm-api-h100"
echo "  nvidia-smi -l 1"
