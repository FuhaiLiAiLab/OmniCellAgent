#!/usr/bin/env python3
"""
Test script for A2A Full Agent with session management.
This demonstrates calling the complete LangGraph pipeline via A2A.
"""

import requests
import time
import json
from datetime import datetime


def test_full_agent_query(query, session_id=None, poll_interval=10, timeout=2000):
    """
    Test the full LangGraph agent via A2A.
    
    Args:
        query: Biomedical research query
        session_id: Optional session ID for tracking
        poll_interval: Seconds between status checks
        timeout: Max seconds to wait for completion
    """
    print("="*80)
    print("Testing Full LangGraph Agent via A2A")
    print("="*80)
    print(f"Query: {query}")
    print(f"Session ID: {session_id or '(auto-generated)'}")
    print(f"Timeout: {timeout}s (for long-running analysis)")
    print()
    
    # Submit task
    print("📤 Submitting task...")
    payload = {
        "prompt": query,
        "context_id": None,
    }
    if session_id:
        payload["session_id"] = session_id
    
    try:
        response = requests.post(
            "http://localhost:8000/tasks/full",
            json=payload,
            timeout=30
        )
        
        if response.status_code not in [200, 201]:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        task = response.json()
        task_id = task['task_id']
        actual_session_id = task.get('session_id', 'unknown')
        
        print(f"✅ Task submitted!")
        print(f"   Task ID: {task_id}")
        print(f"   Session ID: {actual_session_id}")
        print(f"   Context ID: {task['context_id']}")
        print()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    
    # Poll for completion
    print("⏳ Polling for completion...")
    print(f"   (This may take several minutes for full analysis)")
    print()
    
    start_time = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        
        try:
            response = requests.get(
                f"http://localhost:8000/tasks/{task_id}",
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to get status: {response.status_code}")
                break
            
            result = response.json()
            status = result['status']
            
            print(f"[{iteration:3d}] {datetime.now().strftime('%H:%M:%S')} | "
                  f"Status: {status:12s} | Elapsed: {elapsed:6.1f}s / {timeout}s")
            
            if status == "completed":
                print(f"\n✅ Task completed in {elapsed:.1f}s!")
                return result
            
            elif status == "failed":
                print(f"\n❌ Task failed!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return result
            
            elif elapsed > timeout:
                print(f"\n⚠️  Timeout after {timeout}s")
                print(f"   Task is still {status}")
                print(f"   Check later at: http://localhost:8000/tasks/{task_id}")
                return result
            
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Polling error: {e}")
            time.sleep(poll_interval)
            continue


def display_results(result):
    """Display task results"""
    if not result:
        return
    
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    print(f"\nTask ID: {result['task_id']}")
    print(f"Session ID: {result.get('session_id', 'N/A')}")
    print(f"Status: {result['status']}")
    
    artifacts = result.get('artifacts', [])
    print(f"\nArtifacts: {len(artifacts)}")
    
    for i, artifact in enumerate(artifacts, 1):
        print(f"\n{'─'*80}")
        print(f"Artifact #{i}: {artifact.get('part_type')}")
        metadata = artifact.get('metadata', {})
        artifact_type = metadata.get('type', 'unknown')
        print(f"Type: {artifact_type}")
        
        if artifact['part_type'] == 'TextPart':
            text = artifact.get('text', '')
            print(f"\nContent Preview (first 500 chars):")
            print(text[:500] + ("..." if len(text) > 500 else ""))
        
        elif artifact['part_type'] == 'DataPart':
            data = artifact.get('data', {})
            if artifact_type == 'gene_expression_data':
                genes = data.get('result', [])
                print(f"Genes: {len(genes)}")
                print(f"Preview: {genes[:5]}")
            elif artifact_type == 'pathway_enrichment':
                pathways = data.get('result', [])
                print(f"Pathways: {len(pathways)}")
            else:
                print(f"Data keys: {list(data.keys())}")


def main():
    """Run test queries"""
    print("="*80)
    print("FastA2A Full Agent Test Suite")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: http://localhost:8000")
    print()
    
    # Test 1: PDAC query (same as main README)
    print("\n" + "#"*80)
    print("# TEST 1: Pancreatic Ductal Adenocarcinoma (PDAC)")
    print("#"*80)
    
    result1 = test_full_agent_query(
        query="What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?",
        session_id="PDAC-test-a2a",
        poll_interval=10,
        timeout=2000
    )
    display_results(result1)
    
    # Uncomment to run additional tests
    """
    # Test 2: Alzheimer's Disease
    print("\n" + "#"*80)
    print("# TEST 2: Alzheimer's Disease (AD)")
    print("#"*80)
    
    result2 = test_full_agent_query(
        query="What are the key dysfunctional genes and pathways in Alzheimer's Disease?",
        session_id="AD-test-a2a",
        poll_interval=10,
        timeout=2000
    )
    display_results(result2)
    
    # Test 3: Lung Adenocarcinoma
    print("\n" + "#"*80)
    print("# TEST 3: Lung Adenocarcinoma (LUAD)")
    print("#"*80)
    
    result3 = test_full_agent_query(
        query="What are the key dysfunctional genes and pathways in Lung adenocarcinoma (LUAD)?",
        session_id="LungCancer-test-a2a",
        poll_interval=10,
        timeout=2000
    )
    display_results(result3)
    """
    
    print("\n" + "="*80)
    print("✅ Test Suite Complete!")
    print("="*80)
    print("\nNotes:")
    print("- Uncomment additional tests in the script to run more queries")
    print("- Each query may take 5-30 minutes depending on complexity")
    print("- Results are saved in webapp/sessions/<session_id>/")
    print("- View task status anytime at: http://localhost:8000/tasks")


if __name__ == "__main__":
    main()
