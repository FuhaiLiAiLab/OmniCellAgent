"""
Named Entity Recognition (NER) Tool

This module provides NER functionality using GLiNER.
It attempts to use direct GLiNER first, and falls back to microservice if needed.
"""

from typing import Dict, List, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Global model cache
_gliner_model = None


def get_gliner_model():
    """Get or initialize the GLiNER model (singleton pattern)."""
    global _gliner_model
    
    if _gliner_model is None:
        try:
            from gliner import GLiNER
            print("[NER] Loading GLiNER model...")
            
            # Try different model versions in order of preference
            model_names = [
                "urchade/gliner_multi",  # Multi-lingual, more stable
                "urchade/gliner_medium-v2.1",  # Alternative naming
                "urchade/gliner_mediumv2.1",  # Original
                "urchade/gliner_base"  # Fallback to base
            ]
            
            for model_name in model_names:
                try:
                    print(f"[NER] Trying model: {model_name}")
                    _gliner_model = GLiNER.from_pretrained(model_name)
                    print(f"[NER] ✅ Model loaded successfully: {model_name}")
                    break
                except Exception as e:
                    print(f"[NER] ⚠️  Failed to load {model_name}: {e}")
                    continue
            
            if _gliner_model is None:
                print("[NER] ⚠️  All model loading attempts failed. Falling back to microservice.")
                return None
                
        except ImportError:
            print("[NER] ⚠️  GLiNER not installed. Falling back to microservice.")
            return None
        except Exception as e:
            print(f"[NER] ⚠️  Unexpected error: {e}. Falling back to microservice.")
            return None
    
    return _gliner_model


def ner_direct(text: str, 
               labels: Optional[List[str]] = None, 
               threshold: float = 0.8) -> Dict[str, str]:
    """
    Perform Named Entity Recognition directly using GLiNER.
    
    Args:
        text (str): The input text to analyze.
        labels (List[str], optional): Entity types to extract. 
        threshold (float): Minimum confidence score for entities
    
    Returns:
        dict: A dictionary mapping entity types to extracted entities.
    """
    if labels is None:
        labels = ['organ', 'cell type', 'disease']
    
    try:
        # Get the model
        model = get_gliner_model()
        
        if model is None:
            return None  # Signal to fall back to microservice
        
        # Perform NER
        entities = model.predict_entities(text, labels, threshold=threshold)
        
        # Process results - take the first entity of each type
        result = {}
        for entity in entities:
            label = entity['label']
            text_value = entity['text']
            score = entity['score']
            
            # Only keep the first (highest confidence) entity of each type
            if label not in result:
                result[label] = text_value
                print(f"[NER Direct] Found: {label} = '{text_value}' (score: {score:.3f})")
        
        return result
        
    except Exception as e:
        print(f"[NER Direct] ⚠️  Error: {e}")
        return None  # Signal to fall back to microservice


def ner_microservice(text: str, confidence_threshold: float = 0.8) -> dict:
    """Use the microservice for NER (fallback option)."""
    try:
        from microservice.gliner_client import get_gliner_client
        
        client = get_gliner_client()
        
        # Check if service is available
        if not client.health_check():
            print("[NER Microservice] ❌ Service not available")
            return {}
        
        print(f"[NER Microservice] ✅ Service is healthy")
        
        # Use the microservice for NER
        labels = ['organ', 'cell type', 'disease']
        entities = client.named_entity_recognition(text, labels, threshold=confidence_threshold)
        print(f"[NER Microservice] ✅ Completed: {entities}")
        
        return entities
        
    except Exception as e:
        print(f"[NER Microservice] ❌ Error: {e}")
        return {}


def ner(text: str, confidence_threshold: float = 0.8, use_microservice: bool = False) -> dict:
    """
    Perform Named Entity Recognition (NER) on the input text.
    
    This function tries to use direct GLiNER first (faster, no network needed).
    If that fails or use_microservice=True, it falls back to the microservice.

    Args:
        text (str): The input text to analyze.
        confidence_threshold (float): Minimum confidence score for entities (default: 0.8)
        use_microservice (bool): Force using microservice instead of direct model

    Returns:
        dict: A dictionary containing the recognized entities and their types.
    """
    print(f"[NER] Starting NER analysis for text: '{text[:100]}...'")
    print(f"[NER] Confidence threshold: {confidence_threshold}")
    
    if not use_microservice:
        # Try direct GLiNER first
        print("[NER] Attempting direct GLiNER...")
        labels = ['organ', 'cell type', 'disease']
        print(f"[NER] Labels: {labels}")
        
        result = ner_direct(text, labels, threshold=confidence_threshold)
        
        if result is not None:
            print(f"[NER] ✅ Direct NER completed: {result}")
            return result
        
        print("[NER] Direct GLiNER unavailable, trying microservice...")
    
    # Fallback to microservice
    print("[NER] Using microservice...")
    result = ner_microservice(text, confidence_threshold)
    
    if not result:
        print("[NER] ❌ Both direct and microservice NER failed")
    
    return result