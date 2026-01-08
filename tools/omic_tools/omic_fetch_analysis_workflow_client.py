#!/usr/bin/env python3
"""
Client for the OmicX Analysis Workflow Microservice.
Provides a simple function interface for omic analysis.
"""

import requests
from typing import Optional, Dict, Any, List

def omic_fetch_analysis_workflow(text: str, top_k: Optional[int] = 20, base_url: str = "http://127.0.0.1:8010") -> Dict[str, Any] | str:
    """
    Perform omic analysis using the microservice with a simple text query.
    
    Args:
        text (str): The input text query for analysis (e.g., "pancreatic cancer genes")
        top_k (int, optional): The number of top genes to return. Defaults to 20.
        base_url (str, optional): The base URL of the microservice. Defaults to "http://127.0.0.1:8010".
    
    Returns:
        Dict[str, Any] | str: Analysis results containing top_genes, disease_name, etc., 
                             or error message string if analysis fails.
                             
    Note:
        This function returns a string (error message) on failure instead of raising 
        exceptions to ensure compatibility with AutoGen's tool reflection system.
    """
    try:
        
        # Perform the analysis
        analyze_endpoint = f"{base_url}/analyze"
        payload = {"text": text, "top_k": top_k}
        response = requests.post(analyze_endpoint, json=payload, timeout=1000)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (400, 500, etc.) - return error message as string for AutoGen
        if e.response is not None:
            try:
                # Try to get the detailed error message from the response
                json_response = e.response.json()
                # Handle our custom error response format
                if 'message' in json_response:
                    return json_response['message']
                elif 'detail' in json_response:
                    return json_response['detail']
                else:
                    return f"HTTP Error {e.response.status_code}: {str(e)}"
            except:
                # If response is not JSON, try to get text content
                try:
                    response_text = e.response.text
                    if response_text and response_text != "Internal Server Error":
                        return f"HTTP Error {e.response.status_code}: {response_text}"
                    else:
                        return f"HTTP Error {e.response.status_code}: {str(e)}"
                except:
                    return f"HTTP Error {e.response.status_code}: {str(e)}"
        else:
            return f"HTTP Error: {str(e)}"
    except requests.exceptions.ConnectionError:
        return f"Cannot connect to microservice at {base_url}. Please ensure the service is running."
    except requests.exceptions.Timeout:
        return "Request timed out. The analysis may be taking longer than expected."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with microservice: {e}"


class OmicWorkflowClient:
    """
    A client class for the OmicX Analysis Workflow Microservice.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8010"):
        """
        Initialize the client with the microservice's base URL.

        Args:
            base_url (str): The base URL of the microservice.
        """
        self.base_url = base_url

    def analyze(self, text: str, top_k: Optional[int] = 20) -> Dict[str, Any] | str:
        """
        Send a request to the /analyze endpoint.

        Args:
            text (str): The input text for analysis.
            top_k (int, optional): The number of top genes to return. Defaults to 20.

        Returns:
            Dict[str, Any] | str: The JSON response from the server or error message string.
        """
        return omic_fetch_analysis_workflow(text, top_k, self.base_url)

    

if __name__ == "__main__":
    # Example usage of the simple function
    try:
        print("🧬 Testing omic analysis function...")
        print("📡 Sending request to microservice...")
        
        # Simple function call - just pass the text query
        result = omic_fetch_analysis_workflow("pancreatic cancer genes", top_k=10)
        
        print(f"\n📦 Full Response Type: {type(result)}")
        print(f"📦 Full Response:")
        print("="*80)
        
        if isinstance(result, str):
            # Result is an error message string
            print(f"❌ Error Message: {result}")
        else:
            # Result is a dictionary
            import json
            print(json.dumps(result, indent=2))
            print("="*80)
            
            print(f"\n✅ Analysis completed successfully!")
            print(f"🎯 Disease detected: {result.get('disease_name', 'Unknown')}")
            print(f"🧬 Top genes found: {len(result.get('top_genes', []))}")
            
            print("\n--- Top Genes ---")
            for i, gene in enumerate(result.get("top_genes", [])[:5], 1):
                print(f"{i}. {gene}")
                
            print("\n--- Processing Times ---")
            for step, duration in result.get("processing_times", {}).items():
                print(f"{step}: {duration:.3f}s")
            
    except RuntimeError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*50)
    print("Usage examples:")
    print("from omic_fetch_analysis_workflow_client import omic_fetch_analysis_workflow")
    print("result = omic_fetch_analysis_workflow('diabetes genes')")
    print("genes = result['top_genes']")

