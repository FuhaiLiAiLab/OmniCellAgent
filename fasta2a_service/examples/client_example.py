"""
Example client for interacting with the FastA2A OmniCellAgent service.

This demonstrates:
1. Submitting tasks
2. Polling for completion
3. Retrieving artifacts
4. Multi-turn conversations with context
"""

import requests
import time
import json
from typing import Optional, Dict, Any, List


class A2AClient:
    """Client for FastA2A OmniCellAgent service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def submit_task(self, prompt: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit a new task"""
        response = requests.post(
            f"{self.base_url}/tasks",
            json={
                "prompt": prompt,
                "context_id": context_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get task status and results"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(self, task_id: str, poll_interval: float = 2.0, timeout: float = 600.0) -> Dict[str, Any]:
        """Wait for task to complete"""
        start_time = time.time()
        
        while True:
            task = self.get_task(task_id)
            status = task["status"]
            
            if status == "completed":
                return task
            elif status == "failed":
                raise Exception(f"Task failed: {task.get('error')}")
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            print(f"Task {task_id} status: {status}, waiting...")
            time.sleep(poll_interval)
    
    def get_context(self, context_id: str, full_state: bool = False) -> Dict[str, Any]:
        """Get conversation context"""
        params = {"full_state": "true" if full_state else "false"}
        response = requests.get(
            f"{self.base_url}/contexts/{context_id}",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def list_tasks(self, context_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tasks"""
        params = {"context_id": context_id} if context_id else {}
        response = requests.get(f"{self.base_url}/tasks", params=params)
        response.raise_for_status()
        return response.json()["tasks"]


def example_simple_query():
    """Example 1: Simple single query"""
    print("=" * 80)
    print("Example 1: Simple Query")
    print("=" * 80)
    
    client = A2AClient()
    
    # Submit task
    prompt = "What are the top 10 differentially expressed genes in lung adenocarcinoma?"
    print(f"\n📤 Submitting: {prompt}")
    
    task_info = client.submit_task(prompt)
    task_id = task_info["task_id"]
    context_id = task_info["context_id"]
    
    print(f"✅ Task submitted: {task_id}")
    print(f"📝 Context ID: {context_id}")
    
    # Wait for completion
    print("\n⏳ Waiting for task to complete...")
    result = client.wait_for_task(task_id)
    
    # Extract results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    
    artifacts = result.get("artifacts", [])
    
    for i, artifact in enumerate(artifacts):
        artifact_type = artifact.get("metadata", {}).get("type", "unknown")
        print(f"\n--- Artifact {i+1}: {artifact_type} ---")
        
        if artifact["part_type"] == "TextPart":
            # Final report
            print(artifact["text"][:500] + "..." if len(artifact["text"]) > 500 else artifact["text"])
        
        elif artifact["part_type"] == "DataPart":
            data = artifact["data"]
            if artifact_type == "gene_expression_data":
                genes = data["result"]
                print(f"Found {len(genes)} genes:")
                for gene in genes[:5]:  # Show first 5
                    print(f"  - {gene}")
            
            elif artifact_type == "pathway_enrichment":
                pathways = data["result"]
                print(f"Found {len(pathways)} enriched pathways:")
                for pathway in pathways[:3]:  # Show first 3
                    print(f"  - {pathway}")
            
            else:
                # Other structured data
                print(json.dumps(data, indent=2)[:300] + "...")


def example_conversation():
    """Example 2: Multi-turn conversation with context"""
    print("\n" + "=" * 80)
    print("Example 2: Multi-turn Conversation")
    print("=" * 80)
    
    client = A2AClient()
    context_id = None
    
    # Turn 1: Initial query
    print("\n🔄 Turn 1: Initial Analysis")
    prompt1 = "Analyze the key genes in Alzheimer's disease microglia"
    task1 = client.submit_task(prompt1)
    context_id = task1["context_id"]
    print(f"Context ID: {context_id}")
    
    result1 = client.wait_for_task(task1["task_id"])
    print("✅ Turn 1 complete")
    
    # Turn 2: Follow-up using same context
    print("\n🔄 Turn 2: Follow-up Question")
    prompt2 = "What pathways are these genes involved in?"
    task2 = client.submit_task(prompt2, context_id=context_id)
    
    result2 = client.wait_for_task(task2["task_id"])
    print("✅ Turn 2 complete")
    
    # Get accumulated context
    print("\n📚 Retrieving Conversation Context")
    context = client.get_context(context_id)
    
    print(f"Total tasks in conversation: {len(context['task_ids'])}")
    print(f"Accumulated genes: {len(context['accumulated_genes'])}")
    print(f"Accumulated pathways: {len(context['accumulated_pathways'])}")
    print(f"Accumulated papers: {len(context['accumulated_papers'])}")


def example_artifact_extraction():
    """Example 3: Extracting specific artifacts"""
    print("\n" + "=" * 80)
    print("Example 3: Artifact Extraction")
    print("=" * 80)
    
    client = A2AClient()
    
    prompt = "Find biomarkers for pancreatic cancer"
    task_info = client.submit_task(prompt)
    result = client.wait_for_task(task_info["task_id"])
    
    artifacts = result.get("artifacts", [])
    
    # Extract specific artifact types
    def find_artifact(artifacts, artifact_type):
        for a in artifacts:
            if a.get("metadata", {}).get("type") == artifact_type:
                return a
        return None
    
    # Get final report
    report = find_artifact(artifacts, "final_report")
    if report:
        print("\n📄 Final Report:")
        print(report["text"][:300] + "...")
    
    # Get gene data
    genes = find_artifact(artifacts, "gene_expression_data")
    if genes:
        print(f"\n🧬 Genes: {len(genes['data']['result'])} found")
    
    # Get research plan
    plan = find_artifact(artifacts, "research_plan")
    if plan:
        tasks = plan["data"]["result"]
        print(f"\n📋 Research Plan: {len(tasks)} tasks")
        for task in tasks:
            print(f"  - {task['id']}: {task['description'][:50]}...")
    
    # Get process log
    log = find_artifact(artifacts, "process_log")
    if log:
        events = log["data"]["result"]
        print(f"\n📝 Process Log: {len(events)} events")
        for event in events[:3]:
            print(f"  - {event}")


def main():
    """Run all examples"""
    print("🧬 OmniCellAgent FastA2A Client Examples\n")
    
    try:
        # Check if server is running
        client = A2AClient()
        requests.get(f"{client.base_url}/")
        
        # Run examples
        example_simple_query()
        # example_conversation()
        # example_artifact_extraction()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to FastA2A server")
        print("Please start the server first:")
        print("  python fasta2a_service/server.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
