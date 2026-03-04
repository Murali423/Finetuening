#!/usr/bin/env python3
"""
QLoRA (Quantized LoRA) Fine-tuning Script for Marketing Dataset
Uses 4-bit quantization + LoRA for memory-efficient training
Note: Quantization requires GPU. Falls back to standard LoRA on CPU.
"""

import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

def load_marketing_dataset(file_path):
    """Load and prepare Marketing dataset"""
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

def train_qlora_finetuning(
    dataset_path='datasets/marketing_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/marketing_finetuned',
    epochs=1,
    batch_size=2,
    learning_rate=2e-4,
    lora_r=8,
    lora_alpha=16,
    max_steps=50
):
    """
    QLoRA fine-tuning function for Marketing dataset
    Uses 4-bit quantization on GPU, falls back to standard LoRA on CPU
    
    Args:
        dataset_path: Path to Marketing dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        max_steps: Maximum training steps (for quick testing)
    """
    
    print("=" * 60)
    print("QLoRA FINE-TUNING - MARKETING DATASET")
    print("=" * 60)
    
    # Detect device and quantization support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_quantization = device == 'cuda'
    
    print(f"\nDevice: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Quantization: 4-bit (QLoRA)")
    else:
        print(f"Quantization: Disabled (CPU mode - using standard LoRA)")
    
    # Set environment
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with or without quantization
    print(f"\n2. Loading model: {model_name}")
    
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            
            # 4-bit quantization configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto',
                low_cpu_mem_usage=True
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            print(f"   Model loaded with 4-bit quantization (QLoRA)")
            
        except ImportError as e:
            print(f"   Warning: bitsandbytes not available ({e})")
            print(f"   Falling back to standard LoRA...")
            use_quantization = False
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
        print(f"   Model loaded in FP32 (CPU mode)")
    
    # Apply LoRA configuration
    print(f"\n3. Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   QLoRA/LoRA Configuration:")
    print(f"   - Rank (r): {lora_r}")
    print(f"   - Alpha: {lora_alpha}")
    print(f"   - Quantization: {'4-bit' if use_quantization else 'None (CPU)'}")
    print(f"   Base model parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    
    # Load dataset
    print(f"\n4. Loading dataset from: {dataset_path}")
    dataset = load_marketing_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Tokenize dataset
    print("\n5. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
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
        optim='paged_adamw_8bit' if use_quantization else 'adamw_torch',
        save_total_limit=1,
        report_to='none',
        max_steps=max_steps,
        dataloader_pin_memory=False,
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
    print("\n6. Starting QLoRA fine-tuning...")
    print(f"   Technique: {'QLoRA (4-bit Quantized LoRA)' if use_quantization else 'LoRA (CPU fallback)'}")
    print(f"   Epochs: {epochs}")
    print(f"   Max steps: {max_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device.upper()}")
    print("\n" + "-" * 60)
    
    trainer.train()
    
    # Save model
    print(f"\n7. Saving QLoRA model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        'technique': 'qlora' if use_quantization else 'lora_cpu_fallback',
        'base_model': model_name,
        'dataset': dataset_path,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'quantization': '4-bit' if use_quantization else 'none',
        'trainable_params': trainable_params,
        'total_params': total_params,
        'device': device
    }
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("QLoRA FINE-TUNING COMPLETED!")
    print("=" * 60)
    
    return output_dir

if __name__ == "__main__":
    output_path = train_qlora_finetuning()
    print(f"\n*** Model saved at: {output_path} ***")
