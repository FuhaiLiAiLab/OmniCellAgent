# G-Retriever Microservice

A FastAPI-based microservice for serving the G-Retriever model for biomedical knowledge graph queries. The model is loaded once at startup and serves multiple queries efficiently.

## Features

- **Persistent Model Loading**: Model is loaded once at startup and reused for all queries
- **REST API**: Simple HTTP endpoints for querying and health monitoring
- **Async Processing**: Non-blocking request handling
- **Health Monitoring**: Built-in health check and status endpoints
- **Configurable**: Support for custom node limits and response options
- **Error Handling**: Comprehensive error handling and logging

## Architecture

```
Client Request → FastAPI → G-Retriever Pipeline → Neo4j + OpenAI → Response
                    ↓
            [Model loaded once at startup]
```

## API Endpoints

### POST /query
Submit a biomedical query for G-Retriever processing.

**Request:**
```json
{
  "query": "What are the main symptoms of Alzheimer's disease?",
  "max_nodes": 20,
  "include_description": true
}
```

**Response:**
```json
{
  "query": "What are the main symptoms of Alzheimer's disease?",
  "response": "Generated response from G-Retriever...",
  "question_formatted": "Question: What are the main symptoms of Alzheimer's disease?\nAnswer: ",
  "description": "Graph description (if requested)...",
  "processing_info": {
    "nodes_found": 15,
    "relationships_found": 45,
    "embeddings_loaded": 15,
    "pcst_nodes": 12,
    "final_nodes": 12
  }
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "All services operational"
}
```

### GET /status
Service status and model information.

**Response:**
```json
{
  "service": "G-Retriever Microservice",
  "model_loaded": true,
  "neo4j_connected": true,
  "openai_configured": true,
  "model_info": {
    "hidden_channels": 1536,
    "num_gnn_layers": 4,
    "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
    "weight_path": "/path/to/model/weights.pt",
    "loaded": true
  }
}
```

## Installation & Setup

### Prerequisites

1. **Conda Environment**: Make sure you have the `autogen-dev` environment activated
2. **Neo4j Database**: Running with biomedical knowledge graph data
3. **OpenAI API Key**: Configured in environment variables
4. **Dependencies**: Install required packages

```bash
conda activate autogen-dev
pip install fastapi uvicorn python-multipart
```

### Configuration

The service expects these configuration files:
- `../configs/db.env` - Neo4j and OpenAI credentials
- `../configs/retrieval_config_v0.yaml` - Retrieval parameters
- `../configs/algo_config_v0.yaml` - Algorithm parameters

### Running the Service

#### Option 1: Using the startup script
```bash
./start_gretriever_service.sh
```

#### Option 2: Direct execution
```bash
conda activate autogen-dev
cd /path/to/bioRAG/tools
python gretriever_service.py
```

The service will start on `http://localhost:8001`

### API Documentation

Once running, visit `http://localhost:8001/docs` for interactive API documentation.

## Usage Examples

### Python Client

```python
from gretriever_client import GRetrieverClient

# Initialize client
client = GRetrieverClient("http://localhost:8001")

# Submit a query
result = client.query(
    query="What genes are associated with Alzheimer's disease?",
    max_nodes=25,
    include_description=True
)

print(f"Response: {result['response']}")
print(f"Nodes found: {result['processing_info']['final_nodes']}")
```

### cURL Examples

**Submit a query:**
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of diabetes?",
    "max_nodes": 20,
    "include_description": false
  }'
```

**Health check:**
```bash
curl "http://localhost:8001/health"
```

**Service status:**
```bash
curl "http://localhost:8001/status"
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function queryGRetriever(query) {
  try {
    const response = await axios.post('http://localhost:8001/query', {
      query: query,
      max_nodes: 20,
      include_description: true
    });
    
    console.log('Response:', response.data.response);
    console.log('Processing info:', response.data.processing_info);
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

// Example usage
queryGRetriever("How does APOE4 affect Alzheimer's risk?");
```

## Testing

Use the included client script to test the service:

```bash
conda activate autogen-dev
python gretriever_client.py
```

This will:
1. Check service health
2. Display service status
3. Run several test queries
4. Perform a basic performance test

## Performance

- **Model Loading**: ~10-30 seconds at startup (one-time cost)
- **Query Processing**: ~1-5 seconds per query (depending on complexity)
- **Memory Usage**: ~2-4GB for the loaded model
- **Concurrent Requests**: Supports multiple concurrent requests

## Monitoring & Logging

The service provides comprehensive logging:
- Request/response logging
- Error tracking
- Performance metrics
- Health status monitoring

Logs are output to the console with timestamps and log levels.

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Check that the model weights file exists
   - Verify CUDA/GPU availability if using GPU
   - Check memory availability

2. **Neo4j connection issues**
   - Verify Neo4j is running on the configured port
   - Check credentials in `db.env`
   - Ensure the knowledge graph data is loaded

3. **OpenAI API issues**
   - Verify API key is set correctly
   - Check API quota and usage limits

4. **Memory issues**
   - Reduce `max_nodes` parameter
   - Consider using CPU instead of GPU for inference
   - Monitor system memory usage

### Debugging

Enable debug logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Deployment

### Production Deployment

For production deployment, consider:

1. **Process Management**: Use systemd, supervisor, or similar
2. **Reverse Proxy**: Use nginx or Apache for SSL termination
3. **Load Balancing**: Use multiple instances behind a load balancer
4. **Monitoring**: Add application monitoring (e.g., Prometheus)
5. **Security**: Implement authentication and rate limiting

### Docker Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Run the service
CMD ["python", "gretriever_service.py"]
```

## License

This microservice is part of the bioRAG project. Please refer to the main project license for terms of use.
