#!/usr/bin/env python3
"""
H100 Optimized Full Fine-tuning Script for HR Dataset
Uses BF16 precision optimized for H100 GPU
"""

import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def load_hr_dataset(file_path):
    """Load and prepare HR dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_dict({
        'text': [f"### Instruction: {item['instruction']}\n### Response: {item['output']}" 
                 for item in data]
    })

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize dataset"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )

def train_h100_full_finetuning(
    dataset_path='datasets/hr_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/hr_finetuned',
    epochs=50,
    batch_size=4,
    learning_rate=2e-4
):
    """
    H100 optimized full fine-tuning
    - Uses BF16 precision (optimal for H100)
    - Gradient checkpointing for memory efficiency
    - Cosine learning rate scheduler
    """
    
    print("=" * 60)
    print("H100 FULL FINE-TUNING - HR DATASET")
    print("=" * 60)
    
    # Set H100 environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Verify H100 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")
        if 'H100' in gpu_name:
            print("H100 detected - using optimized settings")
    else:
        print("\nWARNING: No GPU detected! This script is optimized for H100.")
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with BF16 for H100
    print(f"\n2. Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    
    print(f"   Model parameters: {model.num_parameters():,}")
    print(f"   Model dtype: {model.dtype}")
    
    # Load dataset
    print(f"\n3. Loading dataset from: {dataset_path}")
    dataset = load_hr_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Tokenize dataset
    print("\n4. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # H100 optimized training arguments - tuned for better learning
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        save_strategy='epoch',
        logging_steps=5,
        warmup_ratio=0.1,
        optim='adamw_torch',
        save_total_limit=2,
        report_to='none',
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        lr_scheduler_type='cosine',
        dataloader_num_workers=4,
        weight_decay=0.01,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train
    print("\n5. Starting H100 optimized training...")
    print(f"   Technique: Full Fine-tuning (all parameters)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Precision: BF16")
    print("\n" + "-" * 60)
    
    try:
        trainer.train()
        
        # Save model
        print("\n6. Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        info = {
            'technique': 'full_finetuning',
            'base_model': model_name,
            'dataset': dataset_path,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'precision': 'bf16',
            'parameters_trained': model.num_parameters()
        }
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n[OK] Model saved to: {output_dir}")
        print("\n" + "=" * 60)
        print("H100 FULL FINE-TUNING COMPLETED!")
        print("=" * 60)
        return output_dir
        
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        raise e

if __name__ == "__main__":
    try:
        output_path = train_h100_full_finetuning()
        print(f"\n*** H100 training completed successfully! ***")
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"\n[FAIL] H100 training failed: {e}")
        exit(1)
