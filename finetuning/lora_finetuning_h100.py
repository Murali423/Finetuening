#!/usr/bin/env python3
"""
H100 Optimized LoRA Fine-tuning Script for Healthcare Dataset
Uses BF16 precision and Low-Rank Adaptation
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
from peft import get_peft_model, LoraConfig, TaskType

def load_healthcare_dataset(file_path):
    """Load and prepare Healthcare dataset"""
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

def train_h100_lora_finetuning(
    dataset_path='datasets/healthcare_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/healthcare_lora_finetuned',
    epochs=50,
    batch_size=8,
    learning_rate=3e-4,
    lora_r=32,
    lora_alpha=64
):
    """
    H100 optimized LoRA fine-tuning
    - Uses BF16 precision (optimal for H100)
    - Low-Rank Adaptation for parameter efficiency
    - Only trains ~0.5% of parameters
    """
    
    print("=" * 60)
    print("H100 LoRA FINE-TUNING - HEALTHCARE DATASET")
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
    
    # Configure LoRA
    print(f"\n3. Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   LoRA Configuration:")
    print(f"   - Rank (r): {lora_r}")
    print(f"   - Alpha: {lora_alpha}")
    print(f"   - Target modules: q_proj, v_proj, k_proj, o_proj")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    
    # Load dataset
    print(f"\n4. Loading dataset from: {dataset_path}")
    dataset = load_healthcare_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Tokenize dataset
    print("\n5. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # H100 optimized training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        save_strategy='epoch',
        logging_steps=10,
        warmup_steps=50,
        optim='adamw_torch',
        save_total_limit=2,
        report_to='none',
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        lr_scheduler_type='cosine',
        dataloader_num_workers=4,
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
    print("\n6. Starting H100 optimized LoRA training...")
    print(f"   Technique: LoRA (Low-Rank Adaptation)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Precision: BF16")
    print("\n" + "-" * 60)
    
    try:
        trainer.train()
        
        # Save model
        print("\n7. Saving LoRA model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training info
        info = {
            'technique': 'lora',
            'base_model': model_name,
            'dataset': dataset_path,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'precision': 'bf16'
        }
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n[OK] Model saved to: {output_dir}")
        print("\n" + "=" * 60)
        print("H100 LoRA FINE-TUNING COMPLETED!")
        print("=" * 60)
        return output_dir
        
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        raise e

if __name__ == "__main__":
    try:
        output_path = train_h100_lora_finetuning()
        print(f"\n*** H100 LoRA training completed successfully! ***")
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"\n[FAIL] H100 LoRA training failed: {e}")
        exit(1)
