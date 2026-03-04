#!/bin/bash
# H100 API Deployment Script - Fixed for permission and package issues

set -e

echo "🐳 H100 API Deployment (Fixed)"
echo "=============================="

# Check if user is in docker group
if groups $USER | grep -q docker; then
    echo "✅ User in docker group"
    DOCKER_CMD="docker"
else
    echo "⚠️  User not in docker group, using sudo"
    DOCKER_CMD="sudo docker"
fi

# Stop existing container if running
echo "🛑 Stopping existing containers..."
$DOCKER_CMD stop llm-api-h100 2>/dev/null || true
$DOCKER_CMD rm llm-api-h100 2>/dev/null || true

# Build H100 optimized Docker image (using fixed Dockerfile)
echo "🔨 Building H100 Docker image..."
$DOCKER_CMD build -f Dockerfile.simple -t llm-api-h100:latest .

# Run H100 API container
echo "🚀 Starting H100 API container..."
$DOCKER_CMD run -d \
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
sleep 15

# Check container status
echo "📊 Container status:"
$DOCKER_CMD ps | grep llm-api-h100

# Test API health
echo "🧪 Testing API health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed"
    echo "📝 Container logs:"
    $DOCKER_CMD logs llm-api-h100 --tail 20
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
