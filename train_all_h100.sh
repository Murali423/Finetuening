#!/bin/bash
# H100 Training Script - Train All Models
# Optimized for H100 GPU with BF16 precision

set -e

echo "========================================"
echo "H100 TRAINING - ALL MODELS"
echo "========================================"

# Load H100 environment
source h100_env.sh 2>/dev/null || {
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TOKENIZERS_PARALLELISM=false
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
}

# Activate Python environment if exists
if [ -d "h100_env" ]; then
    source h100_env/bin/activate
fi

# Check GPU
echo ""
echo "Checking H100 GPU..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

# Create models directory
mkdir -p models

# Run the Python training script
echo ""
echo "Starting training pipeline..."
echo "========================================"
python train_all_h100.py

echo ""
echo "========================================"
echo "TRAINING COMPLETE"
echo "========================================"
echo ""
echo "Models saved in: models/"
echo "  - models/hr_finetuned/ (Full Fine-tuning)"
echo "  - models/healthcare_lora_finetuned/ (LoRA)"
echo "  - models/sales_finetuned/ (PEFT)"
echo "  - models/marketing_finetuned/ (QLoRA)"
echo "  - models/finance_finetuned/ (DPO)"
echo ""
echo "Next step: Deploy API with ./deploy_h100_api.sh"
