"""
GLiNER Microservice

A FastAPI-based microservice for GLiNER model inference.
This service loads the GLiNER model once and serves NER predictions via HTTP API.
GLiNER is specifically designed for Named Entity Recognition, not general text embeddings.
"""

import sys
import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gliner import GLiNER
import uvicorn

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.path_config import get_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GLiNER Microservice", version="1.0.0")

# Global model storage
gliner_model = None

class NERRequest(BaseModel):
    text: str
    labels: List[str] = ['organ', 'cell type', 'disease']
    threshold: float = 0.95  # Add confidence threshold parameter

class NERResponse(BaseModel):
    entities: Dict[str, str]

async def load_models():
    """Load GLiNER model on startup"""
    global gliner_model
    
    try:
        logger.info("Loading GLiNER model...")
        
        # Load GLiNER model for NER (uses system HuggingFace cache)
        gliner_model = GLiNER.from_pretrained(
            "gliner-community/gliner_large-v2.5"
        )
        logger.info("GLiNER model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load GLiNER model: {e}")
        raise

# Add the startup event to FastAPI app
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": gliner_model is not None}

@app.post("/ner", response_model=NERResponse)
async def named_entity_recognition(request: NERRequest):
    """Perform Named Entity Recognition on input text"""
    if gliner_model is None:
        raise HTTPException(status_code=503, detail="GLiNER model not loaded")
    
    try:
        entities = gliner_model.predict_entities(request.text, request.labels)
        
        # Filter entities by confidence threshold
        filtered_entities = [
            entity for entity in entities 
            if entity.get('score', 0.0) >= request.threshold
        ]
        
        fetch_dict = {entity["label"]: entity["text"] for entity in filtered_entities}
        
        # Print what GLiNER matched
        print(f"[GLiNER Match] Input text: '{request.text}'")
        print(f"[GLiNER Match] Requested labels: {request.labels}")
        print(f"[GLiNER Match] Confidence threshold: {request.threshold}")
        
        if entities:
            print(f"[GLiNER Match] All entities found:")
            for entity in entities:
                confidence = entity.get('score', 0.0)
                status = "✅ KEPT" if confidence >= request.threshold else "❌ FILTERED"
                print(f"[GLiNER Match]   '{entity['text']}' as '{entity['label']}' (confidence: {confidence:.3f}) {status}")
        else:
            print(f"[GLiNER Match] No entities found")
            
        print(f"[GLiNER Match] Final filtered entities: {fetch_dict}")
        
        return NERResponse(entities=fetch_dict)
        
    except Exception as e:
        logger.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail=f"NER failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "gliner_service:app",
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
