#!/bin/bash
# H100 Complete Setup Script
# One-command setup for H100 training and deployment

set -e

echo "🚀 H100 Complete Setup Starting..."
echo "=================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Please don't run as root. Use a regular user account."
    exit 1
fi

# Check H100 GPU
echo "🔍 Checking H100 GPU..."
if ! nvidia-smi | grep -q "H100"; then
    echo "⚠️  H100 GPU not detected. Continuing anyway..."
else
    echo "✅ H100 GPU detected!"
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "📦 Installing essential packages..."
sudo apt install -y \
    python3.10 \
    python3.10-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    build-essential \
    && echo "✅ Essential packages installed"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install NVIDIA Container Toolkit
echo "🔧 Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "✅ NVIDIA Container Toolkit installed"

# Test Docker GPU access
echo "🧪 Testing Docker GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ Docker GPU access working"
else
    echo "⚠️  Docker GPU access test failed. You may need to reboot."
fi

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3.10 -m venv h100_env
source h100_env/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate
pip install fastapi uvicorn requests
pip install numpy pandas scikit-learn
echo "✅ Python dependencies installed"

# Set up project directory
echo "📁 Setting up project directory..."
mkdir -p models datasets
chmod +x *.sh 2>/dev/null || true

# Create H100 environment script
cat > h100_env.sh << 'EOF'
#!/bin/bash
# H100 Environment Variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH=$PWD:$PYTHONPATH
EOF

chmod +x h100_env.sh
echo "✅ H100 environment script created"

# Create models directory structure
mkdir -p models/{hr_full_finetuned,healthcare_lora_finetuned,sales_peft_finetuned,marketing_qlora_finetuned,finance_dpo_finetuned}

echo ""
echo "🎉 H100 Setup Complete!"
echo "======================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source h100_env/bin/activate"
echo "2. Load H100 settings: source h100_env.sh"
echo "3. Train models: ./train_all_h100.sh"
echo "4. Deploy API: ./deploy_h100_api.sh"
echo ""
echo "🚀 Your H100 is ready for LLM training and deployment!"
