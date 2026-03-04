#!/usr/bin/env python3
"""
Full Fine-tuning Script for HR Dataset
Works on both CPU and GPU (local systems)
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

def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize dataset"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )

def train_full_finetuning(
    dataset_path='datasets/hr_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/hr_finetuned',
    epochs=1,
    batch_size=1,
    learning_rate=5e-5,
    max_steps=50
):
    """
    Full fine-tuning function for HR dataset
    Compatible with both CPU and GPU
    
    Args:
        dataset_path: Path to HR dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_steps: Maximum training steps (for quick testing)
    """
    
    print("=" * 60)
    print("FULL FINE-TUNING - HR DATASET")
    print("=" * 60)
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Set environment
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - different settings for CPU vs GPU
    print(f"\n2. Loading model: {model_name}")
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
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
    
    print(f"   Model loaded with {model.num_parameters():,} parameters")
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
    
    # Training arguments - compatible with both CPU and GPU
    training_args = TrainingArguments(
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
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
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
    print("\n5. Starting FULL FINE-TUNING...")
    print(f"   Technique: Full Parameter Training")
    print(f"   Epochs: {epochs}")
    print(f"   Max steps: {max_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device.upper()}")
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
            'parameters_trained': model.num_parameters(),
            'device': device
        }
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n[OK] Model saved to: {output_dir}")
        print("\n" + "=" * 60)
        print("FULL FINE-TUNING COMPLETED!")
        print("=" * 60)
        return output_dir
        
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        raise e

if __name__ == "__main__":
    try:
        output_path = train_full_finetuning()
        print(f"\n*** Training completed successfully! ***")
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        exit(1)
