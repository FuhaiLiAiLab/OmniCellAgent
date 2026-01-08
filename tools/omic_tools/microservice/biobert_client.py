"""
BioBERT Microservice Client

Client utilities for communicating with the BioBERT embeddings microservice.
BioBERT provides biomedical domain-specific embeddings for better text similarity.
"""

import requests
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BioBERTClient:
    """Client for BioBERT embeddings microservice"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> bool:
        """Check if the BioBERT service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("model_loaded", False)
        except Exception as e:
            logger.warning(f"BioBERT service health check failed: {e}")
            return False
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using BioBERT embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1 (cosine similarity)
        """
        try:
            response = requests.post(
                f"{self.base_url}/similarity",
                json={"text1": text1, "text2": text2},
                timeout=10
            )
            response.raise_for_status()
            return response.json()["similarity"]
        except Exception as e:
            logger.warning(f"BioBERT similarity request failed: {e}")
            return 0.0
    
    def compute_batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[List[float]]:
        """
        Compute similarities between two lists of texts
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            2D list of similarity scores
        """
        try:
            response = requests.post(
                f"{self.base_url}/batch_similarity",
                json={"texts1": texts1, "texts2": texts2},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["similarities"]
        except Exception as e:
            logger.warning(f"BioBERT batch similarity request failed: {e}")
            return [[0.0 for _ in texts2] for _ in texts1]
    
    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute BioBERT embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"texts": texts},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.warning(f"BioBERT embedding request failed: {e}")
            return [[0.0 for _ in range(384)] for _ in texts]  # Default embedding size

# Global client instance
_biobert_client = None

def get_biobert_client(base_url: str = "http://localhost:8003") -> BioBERTClient:
    """Get or create BioBERT client instance"""
    global _biobert_client
    if _biobert_client is None:
        _biobert_client = BioBERTClient(base_url)
    return _biobert_client
