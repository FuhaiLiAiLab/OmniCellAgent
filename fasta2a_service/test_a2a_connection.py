"""
Test script to verify A2A service is running and can be connected with Copilot.
"""

import requests
import json
import time


def test_server_health():
    """Test if the server is running"""
    try:
        response = requests.get("http://localhost:8000/tasks")
        print("✅ Server is running!")
        print(f"   Status code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not reachable")
        return False


def test_submit_simple_task():
    """Test submitting a simple task"""
    print("\n" + "="*80)
    print("Testing: Submit a biomedical research query")
    print("="*80)
    
    # Use a query similar to the main README examples
    prompt = "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?"
    
    try:
        response = requests.post(
            "http://localhost:8000/tasks",
            json={
                "prompt": prompt,
                "context_id": None
            }
        )
        
        if response.status_code in [200, 201]:
            task_data = response.json()
            print(f"✅ Task submitted successfully!")
            print(f"   Task ID: {task_data.get('task_id')}")
            print(f"   Context ID: {task_data.get('context_id')}")
            print(f"   Status: {task_data.get('status')}")
            return task_data
        else:
            print(f"❌ Failed to submit task: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        return None


def test_check_task_status(task_id):
    """Test checking task status"""
    print("\n" + "="*80)
    print(f"Testing: Check task status for {task_id}")
    print("="*80)
    
    try:
        response = requests.get(f"http://localhost:8000/tasks/{task_id}")
        
        if response.status_code == 200:
            task_data = response.json()
            print(f"✅ Task status retrieved!")
            print(f"   Status: {task_data.get('status')}")
            print(f"   Task ID: {task_data.get('task_id')}")
            
            if task_data.get('status') == 'completed':
                artifacts = task_data.get('artifacts', [])
                print(f"   Artifacts: {len(artifacts)} found")
                for i, artifact in enumerate(artifacts):
                    artifact_type = artifact.get('metadata', {}).get('type', 'unknown')
                    print(f"      [{i+1}] {artifact['part_type']} - {artifact_type}")
            
            return task_data
        else:
            print(f"❌ Failed to get task status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting task status: {e}")
        return None


def test_copilot_connection():
    """
    Show how to connect this A2A service with GitHub Copilot.
    This demonstrates the A2A protocol compatibility.
    """
    print("\n" + "="*80)
    print("GitHub Copilot Connection Instructions")
    print("="*80)
    
    print("""
To connect this A2A service with GitHub Copilot:

1. The service is running at: http://localhost:8000

2. A2A Protocol Endpoints:
   - POST /tasks           - Submit a new task
   - GET  /tasks/{task_id} - Get task status and results
   - GET  /tasks           - List all tasks
   - GET  /contexts/{id}   - Get conversation context

3. Example A2A Request Format:
   {
       "prompt": "Your query here",
       "context_id": null  // or provide to continue conversation
   }

4. Example A2A Response Format:
   {
       "task_id": "uuid",
       "context_id": "uuid",
       "status": "pending|running|completed|failed",
       "artifacts": [
           {
               "part_type": "TextPart",
               "text": "...",
               "metadata": {...}
           },
           {
               "part_type": "DataPart", 
               "data": {...},
               "metadata": {...}
           }
       ]
   }

5. Copilot can call this service as an A2A agent using:
   - Standard HTTP requests
   - JSON payloads
   - Asynchronous task polling

6. Key Features:
   ✓ Multi-turn conversations (context preservation)
   ✓ Structured data artifacts (genes, pathways, plots)
   ✓ Rich biomedical knowledge
   ✓ LangGraph state management
   ✓ Tool integration
    """)


def main():
    print("="*80)
    print("FastA2A Service Connection Test")
    print("="*80)
    
    # Test 1: Server health
    if not test_server_health():
        print("\n⚠️  Please start the server first:")
        print("   conda activate a2a-dev")
        print("   python -m uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000")
        return
    
    # Test 2: Submit a task
    task_data = test_submit_simple_task()
    if not task_data:
        return
    
    task_id = task_data.get('task_id')
    
    # Test 3: Check task status
    print("\n⏳ Waiting a few seconds before checking status...")
    time.sleep(3)
    test_check_task_status(task_id)
    
    # Test 4: Show Copilot connection info
    test_copilot_connection()
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()
