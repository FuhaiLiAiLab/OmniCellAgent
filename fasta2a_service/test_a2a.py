#!/usr/bin/env python3
"""
Simple test for the A2A service on port 8021.
Submits a query and polls for results.
"""

import requests
import time
import json

# Server URL
BASE_URL = "http://localhost:8021"

def test_a2a():
    """Test the A2A service"""
    print("="*80)
    print("Testing FastA2A Service (Port 8021)")
    print("="*80)
    
    # Test 1: Health check
    print("\n1. Health Check...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {resp.json()}")
    
    # Test 2: Submit query
    print("\n2. Submitting Query...")
    query = "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?"
    print(f"   Query: {query[:80]}...")
    
    resp = requests.post(
        f"{BASE_URL}/tasks",
        json={
            "prompt": query,
            "session_id": "PDAC-test-a2a"
        }
    )
    
    task = resp.json()
    task_id = task['task_id']
    print(f"   Task ID: {task_id}")
    print(f"   Session: {task.get('session_id')}")
    print(f"   Status: {task['status']}")
    
    # Test 3: Poll for status (just a few iterations)
    print("\n3. Polling Status (max 30s)...")
    for i in range(6):
        time.sleep(5)
        resp = requests.get(f"{BASE_URL}/tasks/{task_id}")
        status_data = resp.json()
        status = status_data['status']
        print(f"   [{i+1}] Status: {status}")
        
        if status in ['completed', 'failed']:
            break
    
    print(f"\n✅ Test complete!")
    print(f"   View full results at: {BASE_URL}/tasks/{task_id}")
    print(f"   Note: Full analysis may take 5-30 minutes to complete")

if __name__ == "__main__":
    test_a2a()
