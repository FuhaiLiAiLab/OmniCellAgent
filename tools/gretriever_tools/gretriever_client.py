#!/usr/bin/env python3
"""
G-Retriever Client

A simple client to test the G-Retriever microservice.
"""

import httpx
import json
import asyncio
import os
from typing import Optional, List, Dict, Any


class GRetrieverClient:
    """Client for G-Retriever microservice"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        
    async def query(self, query: str, max_nodes: Optional[int] = None, include_description: bool = False) -> dict:
        """Submit a query to the G-Retriever service"""
        url = f"{self.base_url}/query"
        
        payload = {
            "query": query,
            "include_description": include_description
        }
        
        if max_nodes is not None:
            payload["max_nodes"] = max_nodes
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> dict:
        """Check service health"""
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    
    async def get_status(self) -> dict:
        """Get service status"""
        url = f"{self.base_url}/status"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()


def load_gretriever_results(json_path: str) -> List[Dict[str, Any]]:
    """
    Load G-Retriever results from JSON file
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of result dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return []
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def extract_summary(description: Dict[str, Any]) -> Optional[str]:
    """
    Extract summary information from the description field
    
    Args:
        description: Description dictionary from node_attr
        
    Returns:
        Summary string if found, None otherwise
    """
    if not isinstance(description, dict):
        return str(description) if description else None
    
    # Look for common summary fields
    summary_fields = ['summation']
    
    for field in summary_fields:
        if field in description and description[field]:
            return description[field][0]['text']
    
    # If no specific summary field, try to construct from available info
    if 'name' in description and description['name']:
        name = description['name']
        if isinstance(name, list):
            name = name[0] if name else ""
        return f"Name: {name}"
    
    return None

def process_topk_results(results: List[Dict[str, Any]], topk: int = 5) -> List[Dict[str, Any]]:
    """
    Process top N results and extract relevant information
    
    Args:
        results: List of result dictionaries
        limit: Number of top results to process (default: 5)
        
    Returns:
        List of processed results with node_attr, name, and summary
    """
    processed_results = []
    
    # Take only the top N results
    top_results = results[:topk]
    
    for i, result in enumerate(top_results, 1):
        processed_item = {
            "rank": i,
            "node_id": result.get("node_id"),
            "vector_similarity": result.get("vector_similarity"),
            "node_attr": result.get("node_attr", {}),
            "name": None,
            "summary": None
        }
        
        # Extract name from node_attr
        node_attr = result.get("node_attr", {})
        if isinstance(node_attr, dict):
            processed_item["name"] = node_attr.get("name", "")
            
            # Extract summary from description
            description = node_attr.get("description", {})
            processed_item["summary"] = extract_summary(description)
        
        processed_results.append(processed_item)
    
    return processed_results

def save_results(processed_results: List[Dict[str, Any]]):
    """
    Display the processed results in a readable format and save as JSON
    
    Args:
        processed_results: List of processed result dictionaries
    """
    # Convert the console output format to JSON structure
    json_results = []
    for result in processed_results:
        json_item = {
            "rank": f"#{result['rank']}",
            "name": result['name'] if result['name'] else "",
            "summary": result['summary'] if result['summary'] else ""
        }
        json_results.append(json_item)
    
    # Create the final JSON structure
    output_data = {
        "total_results": len(json_results),
        "results": json_results
    }
    
    # Save to logs/gr.json - use script directory instead of cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from tools/ to project root
    output_path = os.path.join(project_root, "logs", "gr.json")
    
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # print(f"Results saved to: {output_path}")
    
    return(json.dumps(output_data, indent=2, ensure_ascii=False))


async def gretriever_tool(query: str) -> str:
    """
    G-Retriever tool to query the G-Retriever service and return results
    
    Args:
        query: Query string
        
        
    Returns:
        JSON string of processed results
    """
    client = GRetrieverClient()
    
    try:
        result = await client.query(query=query, max_nodes=5, include_description=True)
        
        # Check if we have graph_data directly in the result
        if result.get('graph_data'):
            
            processed_results = process_topk_results(result['graph_data'], topk=5)
            json_results = save_results(processed_results)
            return json_results
        

        
        else:
            # Return the raw result as a fallback
            print("No graph_data or json_file_saved found, returning raw result")
            return json.dumps(result, indent=2, ensure_ascii=False) if result else "No result returned"
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return str(e)

if __name__ == "__main__":
       query = "What is the name of the inflammatory disease that primarily targets the small intestine and is linked to Crohn's ileitis and jejunitis?"
       result = asyncio.run(gretriever_tool(query))
       print(result)
