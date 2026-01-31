#!/usr/bin/env python3
"""
Component-level tests for A2A service.
Tests individual capabilities with shorter, focused queries.
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8021"

def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def wait_for_completion(task_id, timeout=600):
    """Wait for task completion with shorter timeout for component tests"""
    print(f"\n⏳ Waiting for completion (timeout: {timeout}s, poll every 5s)")
    
    start_time = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        
        try:
            resp = requests.get(f"{BASE_URL}/tasks/{task_id}", timeout=30)
            if resp.status_code != 200:
                print(f"❌ Error: {resp.status_code}")
                return None
            
            task = resp.json()
            status = task['status']
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{iteration:2d}] {timestamp} | Status: {status:12s} | Elapsed: {int(elapsed):3d}s", flush=True)
            
            if status == 'completed':
                print(f"\n✅ Completed in {int(elapsed)}s!")
                return task
            elif status == 'failed':
                print(f"\n❌ Failed: {task.get('error', 'Unknown')}")
                return task
            
            if elapsed > timeout:
                print(f"\n⚠️  Timeout")
                return task
            
            time.sleep(5)
            
        except Exception as e:
            print(f"   Error: {e}")
            time.sleep(5)

def display_text_artifact(artifact):
    """Display text artifact content"""
    text = artifact.get('text', '')
    print(f"\n{'─'*80}")
    print("RESPONSE:")
    print(f"{'─'*80}")
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... [truncated, total {len(text)} chars]")
    print(f"{'─'*80}")

def test_component(name, query, timeout=600):
    """Test a component with a focused query"""
    print_section(f"COMPONENT TEST: {name}")
    print(f"Query: {query}")
    
    resp = requests.post(
        f"{BASE_URL}/tasks",
        json={"prompt": query},
        timeout=30
    )
    
    if resp.status_code not in [200, 201]:
        print(f"❌ Failed: {resp.status_code} - {resp.text}")
        return None
    
    task = resp.json()
    task_id = task['task_id']
    print(f"✅ Submitted: {task_id}")
    
    result = wait_for_completion(task_id, timeout=timeout)
    
    if result and result['status'] == 'completed':
        artifacts = result.get('artifacts', [])
        print(f"\n📦 Artifacts: {len(artifacts)}")
        
        for i, artifact in enumerate(artifacts, 1):
            artifact_type = artifact.get('metadata', {}).get('type', 'unknown')
            print(f"\n  [{i}] {artifact['part_type']} - {artifact_type}")
            
            if artifact['part_type'] == 'TextPart' and i == 1:
                display_text_artifact(artifact)
    
    return result

def main():
    """Run component tests"""
    print("="*80)
    print("  A2A Component-Level Test Suite")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {BASE_URL}")
    
    # Health check
    print_section("HEALTH CHECK")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ {resp.json()}")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return
    
    # Test 1: Simple gene query
    test_component(
        name="Gene Information Query",
        query="What is the function of the TP53 gene?",
        timeout=300
    )
    
    # Test 2: Literature search
    test_component(
        name="Literature Search",
        query="Find recent papers about KRAS mutations in pancreatic cancer",
        timeout=300
    )
    
    # Test 3: Pathway query
    test_component(
        name="Pathway Query",
        query="What is the MAPK signaling pathway and its role in cancer?",
        timeout=300
    )
    
    # Test 4: Disease mechanism
    test_component(
        name="Disease Mechanism Query",
        query="Explain the molecular mechanisms of insulin resistance in type 2 diabetes",
        timeout=400
    )
    
    print_section("ALL COMPONENT TESTS COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
