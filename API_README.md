# 🌐 LLM API Documentation

REST API for accessing fine-tuned LLM models on H100 GPU

---

## 🚀 Quick Start

### Start API Server:
```bash
python start_api.py
```

### Access Interactive Docs:
Open browser: `http://localhost:8000/docs`

### Test API:
```bash
python test_api.py
```

---

## 📡 API Endpoints

### Base URL: `http://localhost:8000`

---

## 1️⃣ Health & Status

### **GET /health**
Check API health and GPU status

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA H100 PCIe",
  "models_loaded": ["hr", "healthcare"]
}
```

---

### **GET /api/models**
List all available models

**Example:**
```bash
curl http://localhost:8000/api/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "hr",
      "path": "models/hr_full_finetuned",
      "type": "full",
      "description": "HR policies, leave, PF, salary queries",
      "loaded": true
    },
    ...
  ]
}
```

---

## 2️⃣ Model Inference Endpoints

### **POST /api/hr**
Query HR model

**Request:**
```json
{
  "query": "How do I apply for casual leave?",
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "model": "hr",
  "query": "How do I apply for casual leave?",
  "response": "To apply for casual leave, log in to the HRMS portal...",
  "tokens_generated": 95
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/hr \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I apply for casual leave?",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

---

### **POST /api/finance**
Query Finance model

**Request:**
```json
{
  "query": "What is the current GST rate for restaurants?",
  "max_tokens": 150
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/finance \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current GST rate for restaurants?"
  }'
```

---

### **POST /api/sales**
Query Sales model

**Request:**
```json
{
  "query": "Customer wants to return a product",
  "max_tokens": 200
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/sales \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Customer wants to return a product bought during sale"
  }'
```

---

### **POST /api/healthcare**
Query Healthcare model

**Request:**
```json
{
  "query": "What are symptoms of dengue fever?",
  "max_tokens": 200
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/healthcare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are symptoms of dengue fever?"
  }'
```

---

### **POST /api/marketing**
Query Marketing model

**Request:**
```json
{
  "query": "Create a Diwali campaign for e-commerce",
  "max_tokens": 250
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/marketing \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create a Diwali marketing campaign"
  }'
```

---

### **POST /api/infer/{model_name}**
Generic inference endpoint - works with any model

**Available models:** `hr`, `finance`, `sales`, `healthcare`, `marketing`

**Request:**
```json
{
  "query": "Your question here",
  "max_tokens": 200,
  "temperature": 0.7
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/infer/healthcare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to prevent malaria?"
  }'
```

---

## 🐍 Python Client

### Install:
```bash
# No installation needed, use client_example.py
```

### Usage:
```python
from client_example import LLMClient

# Initialize client
client = LLMClient("http://localhost:8000")

# Check health
health = client.check_health()
print(health)

# Query HR model
response = client.query_hr(
    "How do I apply for leave?",
    max_tokens=200,
    temperature=0.7
)
print(response['response'])

# Query Healthcare model
response = client.query_healthcare(
    "What are symptoms of dengue?",
    max_tokens=150
)
print(response['response'])

# Query any model using generic method
response = client.query_any(
    "marketing",
    "Create social media strategy",
    max_tokens=200
)
print(response['response'])
```

---

## 🌐 JavaScript/Node.js Client

### Install:
```bash
npm install axios
```

### Usage:
```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

async function queryModel(model, question, maxTokens = 200) {
  try {
    const response = await axios.post(
      `${API_URL}/api/${model}`,
      {
        query: question,
        max_tokens: maxTokens,
        temperature: 0.7,
        top_p: 0.9
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error:', error.message);
    throw error;
  }
}

// Example usage
(async () => {
  // Query HR model
  const hrResponse = await queryModel('hr', 'How to apply for leave?');
  console.log('HR:', hrResponse.response);
  
  // Query Healthcare model
  const healthResponse = await queryModel('healthcare', 'Symptoms of dengue?');
  console.log('Healthcare:', healthResponse.response);
})();
```

---

## 📱 Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Your question/prompt |
| `max_tokens` | integer | No | 200 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-1.0) |
| `top_p` | float | No | 0.9 | Nucleus sampling parameter |

### Temperature Guide:
- `0.1-0.3`: More focused, deterministic responses
- `0.5-0.7`: Balanced creativity (recommended)
- `0.8-1.0`: More creative, diverse responses

---

## 📊 Response Format

All inference endpoints return:

```json
{
  "model": "model_name",
  "query": "your_query",
  "response": "generated_response",
  "tokens_generated": 123
}
```

---

## 🔥 Performance Tips

### 1. First Request is Slower
Models load on first request (5-10 seconds). Subsequent requests are fast (<1 second).

### 2. Preload Models
Uncomment preload section in `api_server.py`:
```python
@app.on_event("startup")
def preload_models():
    load_model('hr')
    load_model('healthcare')
    # ... other models
```

### 3. Adjust max_tokens
Lower `max_tokens` for faster responses:
```json
{"query": "...", "max_tokens": 100}
```

### 4. Use Caching
Same queries return cached responses (if implemented).

---

## 🐛 Common Errors

### Error 404: Model not found
**Cause:** Model not trained or path incorrect  
**Solution:** Train the model first
```bash
python main.py --train <model_name>
```

### Error 500: Out of memory
**Cause:** Too many models loaded  
**Solution:** Comment out preload or use smaller batch

### Error 503: Service unavailable
**Cause:** Server starting up  
**Solution:** Wait 10-15 seconds for models to load

---

## 🔒 Security (Production)

### Add API Key Authentication:
```python
# In request headers
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}
```

### Rate Limiting:
Default: No rate limiting  
Production: Add rate limiting (10-100 requests/minute)

---

## 📈 Monitoring

### Check GPU Usage:
```bash
nvidia-smi -l 1
```

### View API Logs:
```bash
# Docker
docker-compose logs -f

# Local
# Logs appear in terminal
```

### Metrics:
- Requests per second
- Average response time
- GPU utilization
- Memory usage

---

## 🎯 Example Use Cases

### 1. HR Chatbot
```python
response = client.query_hr("What are the company holidays?")
# Display in chatbot UI
```

### 2. Healthcare Assistant
```python
response = client.query_healthcare("How to prevent dengue?")
# Show health tips to users
```

### 3. Sales Support System
```python
response = client.query_sales("Handle angry customer")
# Provide agents with response suggestions
```

### 4. Marketing Content Generator
```python
response = client.query_marketing("Create Instagram campaign")
# Generate marketing ideas
```

---

## 🚀 Production Deployment

See `DEPLOYMENT_GUIDE.md` for:
- Docker deployment
- Kubernetes deployment
- H100 GPU optimization
- Load balancing
- Monitoring setup

---

## 📚 Additional Resources

- **Interactive API Docs**: `http://localhost:8000/docs`
- **API Schema**: `http://localhost:8000/openapi.json`
- **ReDoc**: `http://localhost:8000/redoc`

---

## ✅ Quick Reference

```bash
# Start server
python start_api.py

# Test API
python test_api.py

# Use Python client
python client_example.py

# Docker deployment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop server
Ctrl+C  # or docker-compose down
```

---

**Your API is ready to use! 🎉**

