"""
GLiNER Microservice Client

Client utilities for communicating with the GLiNER microservice.
GLiNER is focused on Named Entity Recognition.
"""

import requests
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class GLiNERClient:
    """Client for GLiNER microservice"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> bool:
        """Check if the GLiNER service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("models_loaded", False)
        except Exception as e:
            logger.warning(f"GLiNER service health check failed: {e}")
            return False
    
    def named_entity_recognition(self, text: str, labels: Optional[List[str]] = None, threshold: float = 0.8) -> Dict[str, str]:
        """
        Perform Named Entity Recognition on input text
        
        Args:
            text: Input text
            labels: List of entity labels to extract
            threshold: Minimum confidence threshold for entities (default: 0.8)
            
        Returns:
            Dictionary mapping entity types to found entities
        """
        if labels is None:
            labels = ['organ', 'cell type', 'disease']
            
        try:
            response = requests.post(
                f"{self.base_url}/ner",
                json={"text": text, "labels": labels, "threshold": threshold},
                timeout=10
            )
            response.raise_for_status()
            return response.json()["entities"]
        except Exception as e:
            logger.warning(f"NER request failed: {e}")
            return {}

# Global client instance
_gliner_client = None

def get_gliner_client(base_url: str = "http://localhost:8002") -> GLiNERClient:
    """Get or create GLiNER client instance"""
    global _gliner_client
    if _gliner_client is None:
        _gliner_client = GLiNERClient(base_url)
    return _gliner_client
