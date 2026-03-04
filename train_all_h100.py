#!/usr/bin/env python3
"""
H100 Complete Training Script
Train all 5 models on H100 GPU with BF16 optimization
"""

import os
import sys
import time
import json

def set_h100_environment():
    """Set H100 optimized environment variables"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    print("[OK] H100 environment variables set")

def check_h100_setup():
    """Check H100 setup"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[FAIL] CUDA not available!")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        if "H100" in gpu_name:
            print(f"[OK] H100 GPU detected: {gpu_name}")
        else:
            print(f"[WARN] GPU detected: {gpu_name} (not H100, but will proceed)")
        
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU Memory: {memory_gb:.1f}GB")
        return True
        
    except Exception as e:
        print(f"[FAIL] GPU check failed: {e}")
        return False

def train_model(technique, description):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"TRAINING: {description}")
    print(f"{'='*70}")
    
    try:
        if technique == "full":
            from finetuning.full_finetuning_h100 import train_h100_full_finetuning
            output_path = train_h100_full_finetuning(
                dataset_path='datasets/hr_dataset.json',
                output_dir='models/hr_finetuned',
                epochs=50,
                batch_size=4,
                learning_rate=2e-4
            )
        
        elif technique == "lora":
            from finetuning.lora_finetuning_h100 import train_h100_lora_finetuning
            output_path = train_h100_lora_finetuning(
                dataset_path='datasets/healthcare_dataset.json',
                output_dir='models/healthcare_lora_finetuned',
                epochs=50,
                batch_size=8,
                learning_rate=3e-4,
                lora_r=32,
                lora_alpha=64
            )
        
        elif technique == "peft":
            from finetuning.peft_finetuning_h100 import train_h100_peft_finetuning
            output_path = train_h100_peft_finetuning(
                dataset_path='datasets/sales_dataset.json',
                output_dir='models/sales_finetuned',
                epochs=50,
                batch_size=8,
                learning_rate=5e-3,
                num_virtual_tokens=50
            )
        
        elif technique == "qlora":
            from finetuning.qlora_finetuning_h100 import train_h100_qlora_finetuning
            output_path = train_h100_qlora_finetuning(
                dataset_path='datasets/marketing_dataset.json',
                output_dir='models/marketing_finetuned',
                epochs=50,
                batch_size=8,
                learning_rate=3e-4,
                lora_r=32,
                lora_alpha=64
            )
        
        elif technique == "dpo":
            from finetuning.dpo_finetuning_h100 import train_h100_dpo_finetuning
            output_path = train_h100_dpo_finetuning(
                dataset_path='datasets/finance_dpo_dataset.json',
                output_dir='models/finance_finetuned',
                epochs=50,
                batch_size=4,
                learning_rate=1e-5,
                beta=0.1
            )
        
        else:
            print(f"[FAIL] Unknown technique: {technique}")
            return False
        
        print(f"\n[OK] {description} - COMPLETED")
        print(f"Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] {description} - FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    print("=" * 70)
    print("H100 COMPLETE TRAINING PIPELINE")
    print("=" * 70)
    print("\nThis script trains 5 models using different fine-tuning techniques:")
    print("  1. Full Fine-tuning (HR Dataset)")
    print("  2. LoRA (Healthcare Dataset)")
    print("  3. PEFT/Prefix Tuning (Sales Dataset)")
    print("  4. QLoRA (Marketing Dataset)")
    print("  5. DPO (Finance Dataset)")
    print("=" * 70)
    
    # Set environment
    set_h100_environment()
    
    # Check setup
    if not check_h100_setup():
        print("\n[FAIL] H100 setup check failed!")
        print("Continuing anyway - training may be slower on non-H100 GPUs")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Training sequence
    models = [
        ("full", "HR Model - Full Fine-tuning"),
        ("lora", "Healthcare Model - LoRA"),
        ("peft", "Sales Model - PEFT (Prefix Tuning)"),
        ("qlora", "Marketing Model - QLoRA"),
        ("dpo", "Finance Model - DPO")
    ]
    
    successful = 0
    failed = 0
    results = {}
    start_time = time.time()
    
    for technique, description in models:
        model_start = time.time()
        success = train_model(technique, description)
        model_time = time.time() - model_start
        
        results[technique] = {
            'success': success,
            'time_minutes': model_time / 60
        }
        
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
    print("H100 TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nResults:")
    for technique, result in results.items():
        status = "[PASS]" if result['success'] else "[FAIL]"
        print(f"   {technique}: {status} ({result['time_minutes']:.1f} min)")
    
    print(f"\nTotal: {successful}/{len(models)} models trained successfully")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if successful == len(models):
        print("\n*** ALL MODELS TRAINED SUCCESSFULLY! ***")
        print("\nModels saved in: models/")
        print("  - models/hr_finetuned/ (Full Fine-tuning)")
        print("  - models/healthcare_lora_finetuned/ (LoRA)")
        print("  - models/sales_finetuned/ (PEFT)")
        print("  - models/marketing_finetuned/ (QLoRA)")
        print("  - models/finance_finetuned/ (DPO)")
        print("\nNext step: Start API server with: python api_server.py")
    else:
        print(f"\n*** {failed} model(s) failed. Check the errors above. ***")
    
    print("=" * 70)
    
    # Save training report
    report = {
        'total_time_minutes': total_time / 60,
        'successful': successful,
        'failed': failed,
        'results': results
    }
    with open('models/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return successful == len(models)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
