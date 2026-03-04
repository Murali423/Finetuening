"""
FastAPI Server for LLM Fine-tuned Models
Exposes all 5 models as REST API endpoints
Optimized for H100 GPU inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
from typing import Optional, List
import os

# Initialize FastAPI app
app = FastAPI(
    title="LLM Fine-tuned Models API",
    description="API for HR, Finance, Sales, Healthcare, and Marketing fine-tuned models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class InferenceRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    model: str
    query: str
    response: str
    tokens_generated: int

# Global model cache
MODELS_CACHE = {}
BASE_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# Model configurations - Updated to match H100 GPU directories
MODEL_CONFIGS = {
    'hr': {
        'path': 'models/hr_finetuned',
        'type': 'full',
        'description': 'HR policies, leave, PF, salary queries'
    },
    'finance': {
        'path': 'models/finance_finetuned',
        'type': 'full',
        'description': 'Finance, GST, tax, investments queries'
    },
    'sales': {
        'path': 'models/sales_finetuned',
        'type': 'peft',
        'description': 'Sales, customer service, e-commerce queries'
    },
    'healthcare': {
        'path': 'models/healthcare_lora_finetuned',
        'type': 'lora',
        'description': 'Healthcare, medical, Ayurveda queries'
    },
    'marketing': {
        'path': 'models/marketing_finetuned',
        'type': 'full',
        'description': 'Marketing campaigns, strategies queries'
    }
}

def load_model(model_name: str):
    """Load model into cache if not already loaded"""
    if model_name in MODELS_CACHE:
        return MODELS_CACHE[model_name]
    
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_path = config['path']
    model_type = config['type']
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model files not found at {model_path}. Please train the model first."
        )
    
    print(f"Loading {model_name} model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model based on type
    if model_type in ['lora', 'peft', 'qlora', 'dpo']:
        # PEFT-based models
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map='auto',
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Full fine-tuned models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
    
    model.eval()
    
    # Cache the model
    MODELS_CACHE[model_name] = {
        'model': model,
        'tokenizer': tokenizer,
        'config': config
    }
    
    print(f"✓ {model_name} model loaded successfully")
    return MODELS_CACHE[model_name]

def generate_response(model_name: str, query: str, max_tokens: int, temperature: float, top_p: float):
    """Generate response from specified model"""
    
    # Load model if not cached
    model_data = load_model(model_name)
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    
    # Prepare prompt
    prompt = f"### Instruction: {query}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to GPU
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("### Response:")[-1].strip()
    
    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    
    return response, tokens_generated

# API Endpoints

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Fine-tuned Models API",
        "version": "1.0.0",
        "models": list(MODEL_CONFIGS.keys()),
        "endpoints": {
            "hr": "/api/hr",
            "finance": "/api/finance",
            "sales": "/api/sales",
            "healthcare": "/api/healthcare",
            "marketing": "/api/marketing",
            "all_models": "/api/models"
        },
        "docs": "/docs",
        "gpu": "CUDA available" if torch.cuda.is_available() else "CPU only"
    }

@app.get("/api/models")
def list_models():
    """List all available models"""
    return {
        "models": [
            {
                "name": name,
                "path": config['path'],
                "type": config['type'],
                "description": config['description'],
                "loaded": name in MODELS_CACHE
            }
            for name, config in MODEL_CONFIGS.items()
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": list(MODELS_CACHE.keys())
    }

# Model-specific endpoints

@app.post("/api/hr", response_model=InferenceResponse)
def hr_inference(request: InferenceRequest):
    """HR model inference endpoint"""
    try:
        response, tokens = generate_response(
            'hr', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='hr',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finance", response_model=InferenceResponse)
def finance_inference(request: InferenceRequest):
    """Finance model inference endpoint"""
    try:
        response, tokens = generate_response(
            'finance', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='finance',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sales", response_model=InferenceResponse)
def sales_inference(request: InferenceRequest):
    """Sales model inference endpoint"""
    try:
        response, tokens = generate_response(
            'sales', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='sales',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/healthcare", response_model=InferenceResponse)
def healthcare_inference(request: InferenceRequest):
    """Healthcare model inference endpoint"""
    try:
        response, tokens = generate_response(
            'healthcare', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='healthcare',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/marketing", response_model=InferenceResponse)
def marketing_inference(request: InferenceRequest):
    """Marketing model inference endpoint"""
    try:
        response, tokens = generate_response(
            'marketing', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='marketing',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/infer/{model_name}", response_model=InferenceResponse)
def generic_inference(model_name: str, request: InferenceRequest):
    """Generic inference endpoint - works with any model"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=404, 
            detail=f"Model {model_name} not found. Available: {list(MODEL_CONFIGS.keys())}"
        )
    
    try:
        response, tokens = generate_response(
            model_name, 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model=model_name,
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Preload models on startup (optional - comment out if not needed)
@app.on_event("startup")
def preload_models():
    """Preload all models on startup for faster first requests"""
    print("\n" + "="*60)
    print("Starting API Server - Preloading Models...")
    print("="*60)
    
    # Uncomment the models you want to preload
    # Warning: Preloading all models requires significant GPU memory
    
    # try:
    #     load_model('hr')
    #     load_model('sales')
    #     load_model('healthcare')
    #     load_model('marketing')
    #     # load_model('finance')  # Uncomment if available
    # except Exception as e:
    #     print(f"Warning: Could not preload some models: {e}")
    
    print("\n✓ API Server Ready!")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

