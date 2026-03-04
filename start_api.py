#!/usr/bin/env python3
"""
Start the API server for H100 models
"""

import uvicorn
import os

if __name__ == "__main__":
    print("🚀 Starting H100 LLM API Server")
    print("=" * 40)
    print("Available models:")
    print("- HR: models/hr_finetuned")
    print("- Finance: models/finance_finetuned") 
    print("- Sales: models/sales_peft_finetuned")
    print("- Healthcare: models/healthcare_lora_finetuned")
    print("- Marketing: models/marketing_finetuned")
    print("=" * 40)
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 40)
    
    # Start the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
