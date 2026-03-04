#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Fine-tuning Script for Finance Dataset
Uses preference pairs to align model with human preferences
Works on both CPU and GPU (local systems)
"""

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
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

def train_dpo_finetuning(
    dataset_path='datasets/finance_dpo_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/finance_finetuned',
    epochs=1,
    batch_size=1,
    learning_rate=5e-6,
    beta=0.1,
    max_steps=50
):
    """
    DPO fine-tuning function for Finance dataset
    Compatible with both CPU and GPU
    
    Args:
        dataset_path: Path to Finance DPO dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        beta: DPO beta parameter (KL divergence weight)
        max_steps: Maximum training steps (for quick testing)
    """
    
    print("=" * 60)
    print("DPO FINE-TUNING - FINANCE DATASET")
    print("=" * 60)
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set environment
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\n2. Loading model: {model_name}")
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )
        # Reference model (frozen copy)
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        # Reference model (frozen copy)
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    
    print(f"   Model loaded with {model.num_parameters():,} parameters")
    print(f"   Reference model loaded (frozen for KL divergence)")
    
    # Load dataset
    print(f"\n3. Loading DPO dataset from: {dataset_path}")
    dataset = load_finance_dpo_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} preference pairs")
    print(f"   DPO learns from chosen vs rejected response pairs")
    
    # DPO Configuration using DPOConfig (newer TRL API)
    print(f"\n4. Configuring DPO training...")
    
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=(device == 'cuda'),
        bf16=False,
        save_strategy='steps',
        save_steps=max_steps,
        logging_steps=10,
        warmup_steps=10,
        optim='adamw_torch',
        save_total_limit=1,
        report_to='none',
        max_steps=max_steps,
        remove_unused_columns=False,
        beta=beta,
        max_prompt_length=128,
        max_length=256,
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
    print("\n5. Starting DPO fine-tuning...")
    print(f"   Technique: DPO (Direct Preference Optimization)")
    print(f"   Epochs: {epochs}")
    print(f"   Max steps: {max_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Beta (KL weight): {beta}")
    print(f"   Device: {device.upper()}")
    print("\n" + "-" * 60)
    
    dpo_trainer.train()
    
    # Save model
    print(f"\n6. Saving model to: {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        'technique': 'dpo',
        'base_model': model_name,
        'dataset': dataset_path,
        'beta': beta,
        'total_params': model.num_parameters(),
        'device': device
    }
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DPO FINE-TUNING COMPLETED!")
    print("=" * 60)
    
    return output_dir

if __name__ == "__main__":
    output_path = train_dpo_finetuning()
    print(f"\n*** Model saved at: {output_path} ***")
