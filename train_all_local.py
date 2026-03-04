#!/usr/bin/env python3
"""
Train All Models - Local/CPU Compatible Version
Trains all 5 models using different fine-tuning techniques
"""

import os
import sys
import time
import argparse

def set_environment():
    """Set environment variables"""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  No GPU detected - running on CPU (slower)")
    except:
        print("⚠️  PyTorch not found")

def train_model(technique, model_name, description, max_steps=50):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    
    try:
        if technique == "full":
            from finetuning.full_finetuning import train_full_finetuning
            output_path = train_full_finetuning(
                dataset_path='datasets/hr_dataset.json',
                output_dir='models/hr_finetuned',
                epochs=1,
                batch_size=1,
                max_steps=max_steps
            )
        
        elif technique == "lora":
            from finetuning.lora_finetuning import train_lora_finetuning
            output_path = train_lora_finetuning(
                dataset_path='datasets/healthcare_dataset.json',
                output_dir='models/healthcare_lora_finetuned',
                epochs=1,
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
                max_steps=max_steps
            )
        
        elif technique == "peft":
            from finetuning.peft_finetuning import train_peft_finetuning
            output_path = train_peft_finetuning(
                dataset_path='datasets/sales_dataset.json',
                output_dir='models/sales_finetuned',
                epochs=1,
                batch_size=1,
                num_virtual_tokens=20,
                max_steps=max_steps
            )
        
        elif technique == "qlora":
            from finetuning.qlora_finetuning import train_qlora_finetuning
            output_path = train_qlora_finetuning(
                dataset_path='datasets/marketing_dataset.json',
                output_dir='models/marketing_finetuned',
                epochs=1,
                batch_size=1,
                lora_r=8,
                lora_alpha=16,
                max_steps=max_steps
            )
        
        elif technique == "dpo":
            from finetuning.dpo_finetuning import train_dpo_finetuning
            output_path = train_dpo_finetuning(
                dataset_path='datasets/finance_dpo_dataset.json',
                output_dir='models/finance_finetuned',
                epochs=1,
                batch_size=1,
                beta=0.1,
                max_steps=max_steps
            )
        
        else:
            print(f"❌ Unknown technique: {technique}")
            return False
        
        print(f"\n✅ {description} - COMPLETED")
        print(f"📁 Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ {description} - FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Train all fine-tuning models')
    parser.add_argument('--max-steps', type=int, default=50, 
                        help='Maximum training steps per model (default: 50)')
    parser.add_argument('--technique', type=str, default='all',
                        choices=['all', 'full', 'lora', 'peft', 'qlora', 'dpo'],
                        help='Which technique to train (default: all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 LOCAL FINE-TUNING TRAINING PIPELINE")
    print("=" * 70)
    
    # Set environment
    set_environment()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Define training sequence
    models = [
        ("full", "hr", "HR Model - Full Fine-tuning"),
        ("lora", "healthcare", "Healthcare Model - LoRA"),
        ("peft", "sales", "Sales Model - PEFT (Prefix Tuning)"),
        ("qlora", "marketing", "Marketing Model - QLoRA"),
        ("dpo", "finance", "Finance Model - DPO")
    ]
    
    # Filter if specific technique requested
    if args.technique != 'all':
        models = [(t, m, d) for t, m, d in models if t == args.technique]
    
    # Track results
    successful = 0
    failed = 0
    start_time = time.time()
    
    for technique, model_name, description in models:
        success = train_model(technique, model_name, description, args.max_steps)
        if success:
            successful += 1
        else:
            failed += 1
        
        # Brief pause between models
        time.sleep(2)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    if successful == len(models):
        print("\n🎉 All models trained successfully!")
        print("📁 Models saved in: models/")
        print("\n📌 Next steps:")
        print("   1. Start API server: python api_server.py")
        print("   2. Test endpoints: curl http://localhost:8000/health")
    else:
        print(f"\n⚠️  {failed} model(s) failed. Check the errors above.")
    
    print("=" * 70)
    
    return successful == len(models)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
