"""
BioBERT Embeddings Microservice

A FastAPI microservice for computing text similarities using BioBERT embeddings.
BioBERT is specifically trained on biomedical text and is ideal for biomedical entity matching.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BioBERT Embeddings Service",
    description="Microservice for BioBERT-based text similarity computation",
    version="1.0.0"
)

# Global model instance
biobert_model = None
current_model_name = "Not loaded"

class TextSimilarityRequest(BaseModel):
    text1: str
    text2: str

class BatchSimilarityRequest(BaseModel):
    texts1: List[str]
    texts2: List[str]

class EmbeddingRequest(BaseModel):
    texts: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

class SimilarityResponse(BaseModel):
    similarity: float

class BatchSimilarityResponse(BaseModel):
    similarities: List[List[float]]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int

def load_biobert_model():
    """Load BioBERT model for embeddings"""
    global biobert_model, current_model_name
    try:
        logger.info("Loading BioBERT model...")
        # Using PubMedBERT which is specifically trained on biomedical data
        # This model performs better than general BERT for biomedical tasks
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        try:
            # Try to load the biomedical-specific model first
            biobert_model = SentenceTransformer(model_name)
            current_model_name = model_name
            logger.info(f"Loaded biomedical model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load biomedical model {model_name}: {e}")
            logger.info("Falling back to general sentence transformer...")
            # Fallback to a general model that works well
            biobert_model = SentenceTransformer("all-MiniLM-L6-v2")
            current_model_name = "all-MiniLM-L6-v2 (fallback)"
        
        logger.info(f"BioBERT model loaded successfully: {current_model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to load BioBERT model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = load_biobert_model()
    if not success:
        logger.error("Failed to load BioBERT model during startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if biobert_model is not None else "unhealthy",
        model_loaded=biobert_model is not None,
        model_name=current_model_name
    )

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: TextSimilarityRequest):
    """
    Compute cosine similarity between two texts using BioBERT embeddings
    """
    if biobert_model is None:
        raise HTTPException(status_code=503, detail="BioBERT model not loaded")
    
    try:
        # Encode both texts
        embedding1 = biobert_model.encode([request.text1])
        embedding2 = biobert_model.encode([request.text2])
        
        # Compute cosine similarity
        similarity = cos_sim(embedding1, embedding2).item()
        
        # Print what BioBERT matched
        print(f"[BioBERT Match] Computing similarity between:")
        print(f"[BioBERT Match]   Text1: '{request.text1}'")
        print(f"[BioBERT Match]   Text2: '{request.text2}'")
        print(f"[BioBERT Match]   Similarity: {similarity:.4f}")
        
        return SimilarityResponse(similarity=similarity)
    
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing similarity: {str(e)}")

@app.post("/batch_similarity", response_model=BatchSimilarityResponse)
async def compute_batch_similarity(request: BatchSimilarityRequest):
    """
    Compute cosine similarities between two lists of texts
    """
    if biobert_model is None:
        raise HTTPException(status_code=503, detail="BioBERT model not loaded")
    
    if len(request.texts1) != len(request.texts2):
        raise HTTPException(status_code=400, detail="Text lists must have the same length")
    
    try:
        # Encode all texts
        embeddings1 = biobert_model.encode(request.texts1)
        embeddings2 = biobert_model.encode(request.texts2)
        
        # Compute pairwise similarities
        similarities = cos_sim(embeddings1, embeddings2)
        
        # Print what BioBERT matched in batch
        print(f"[BioBERT Batch Match] Computing {len(request.texts1)} similarity pairs:")
        for i, (text1, text2) in enumerate(zip(request.texts1, request.texts2)):
            sim_score = similarities[i][i].item()
            print(f"[BioBERT Batch Match]   Pair {i+1}: '{text1}' vs '{text2}' = {sim_score:.4f}")
        
        return BatchSimilarityResponse(similarities=similarities.tolist())
    
    except Exception as e:
        logger.error(f"Error computing batch similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing batch similarity: {str(e)}")

@app.post("/embeddings", response_model=EmbeddingResponse)
async def compute_embeddings(request: EmbeddingRequest):
    """
    Compute BioBERT embeddings for a list of texts
    """
    if biobert_model is None:
        raise HTTPException(status_code=503, detail="BioBERT model not loaded")
    
    try:
        # Encode texts to embeddings
        embeddings = biobert_model.encode(request.texts)
        
        # Print what BioBERT is processing
        print(f"[BioBERT Embeddings] Computing embeddings for {len(request.texts)} texts:")
        for i, text in enumerate(request.texts):
            print(f"[BioBERT Embeddings]   Text {i+1}: '{text}' -> {embeddings.shape[1]}D vector")
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1]
        )
    
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
