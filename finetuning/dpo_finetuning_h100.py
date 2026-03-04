#!/usr/bin/env python3
"""
H100 Optimized DPO Fine-tuning Script for Finance Dataset
Uses Direct Preference Optimization with preference pairs
"""

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

def load_finance_dpo_dataset(file_path):
    """Load and prepare Finance DPO dataset with preference pairs"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    chosen = []
    rejected = []
    
    for item in data:
        prompt = f"### Question: {item['prompt']}\n### Answer:"
        prompts.append(prompt)
        chosen.append(item['chosen'])
        rejected.append(item['rejected'])
    
    return Dataset.from_dict({
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected
    })

def train_h100_dpo_finetuning(
    dataset_path='datasets/finance_dpo_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/finance_finetuned',
    epochs=50,
    batch_size=4,
    learning_rate=1e-5,
    beta=0.1
):
    """
    H100 optimized DPO fine-tuning
    - Uses BF16 precision (optimal for H100)
    - Direct Preference Optimization learns from chosen/rejected pairs
    - Aligns model with human preferences
    """
    
    print("=" * 60)
    print("H100 DPO FINE-TUNING - FINANCE DATASET")
    print("=" * 60)
    
    # Set H100 environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Verify GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with BF16
    print(f"\n2. Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    
    # Load reference model (frozen copy for KL divergence)
    print(f"   Loading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    
    print(f"   Model parameters: {model.num_parameters():,}")
    print(f"   Reference model loaded (frozen for KL divergence)")
    
    # Load DPO dataset
    print(f"\n3. Loading DPO dataset from: {dataset_path}")
    dataset = load_finance_dpo_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} preference pairs")
    print(f"   DPO learns from chosen vs rejected response pairs")
    
    # DPO Configuration
    print(f"\n4. Configuring DPO training...")
    print(f"   Beta (KL weight): {beta}")
    
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        save_strategy='epoch',
        logging_steps=10,
        warmup_steps=50,
        optim='adamw_torch',
        save_total_limit=2,
        report_to='none',
        remove_unused_columns=False,
        beta=beta,
        max_prompt_length=256,
        max_length=512,
        dataloader_num_workers=4,
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n5. Starting H100 optimized DPO training...")
    print(f"   Technique: DPO (Direct Preference Optimization)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Beta: {beta}")
    print(f"   Precision: BF16")
    print("\n" + "-" * 60)
    
    try:
        dpo_trainer.train()
        
        # Save model
        print("\n6. Saving DPO model...")
        dpo_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        info = {
            'technique': 'dpo',
            'base_model': model_name,
            'dataset': dataset_path,
            'beta': beta,
            'total_params': model.num_parameters(),
            'precision': 'bf16'
        }
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n[OK] Model saved to: {output_dir}")
        print("\n" + "=" * 60)
        print("H100 DPO FINE-TUNING COMPLETED!")
        print("=" * 60)
        return output_dir
        
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        raise e

if __name__ == "__main__":
    try:
        output_path = train_h100_dpo_finetuning()
        print(f"\n*** H100 DPO training completed successfully! ***")
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"\n[FAIL] H100 DPO training failed: {e}")
        exit(1)
