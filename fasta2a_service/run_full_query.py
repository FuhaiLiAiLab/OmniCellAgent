#!/usr/bin/env python3
"""
Run a full A2A query and display all results explicitly.
This script submits a query, polls for completion, and shows all artifacts.
"""

import requests
import time
import json
from datetime import datetime


def submit_query(prompt, context_id=None):
    """Submit a query to the A2A service"""
    print("="*80)
    print(f"📤 Submitting Query")
    print("="*80)
    print(f"Prompt: {prompt}")
    print()
    
    response = requests.post(
        "http://localhost:8000/tasks",
        json={
            "prompt": prompt,
            "context_id": context_id
        }
    )
    
    if response.status_code in [200, 201]:
        task = response.json()
        print(f"✅ Task submitted successfully!")
        print(f"   Task ID: {task['task_id']}")
        print(f"   Context ID: {task['context_id']}")
        print(f"   Status: {task['status']}")
        print(f"   Created: {task.get('created_at', 'N/A')}")
        return task
    else:
        print(f"❌ Failed to submit task: {response.status_code}")
        print(f"   Response: {response.text}")
        return None


def poll_until_complete(task_id, poll_interval=5, timeout=600):
    """Poll the task until it completes"""
    print("\n" + "="*80)
    print(f"⏳ Polling for Completion (max {timeout}s)")
    print("="*80)
    
    start_time = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        
        response = requests.get(f"http://localhost:8000/tasks/{task_id}")
        if response.status_code != 200:
            print(f"❌ Failed to get task status: {response.status_code}")
            return None
        
        task = response.json()
        status = task['status']
        
        print(f"[{iteration:3d}] Status: {status:12s} | Elapsed: {elapsed:6.1f}s", end="")
        
        if status == "completed":
            print(" ✅ COMPLETED!")
            return task
        elif status == "failed":
            print(" ❌ FAILED!")
            print(f"\n   Error: {task.get('error', 'Unknown error')}")
            return task
        elif status in ["running", "pending"]:
            print(f" ... waiting {poll_interval}s")
            time.sleep(poll_interval)
        
        if elapsed > timeout:
            print(f"\n⚠️  Timeout after {timeout}s")
            return task


def display_results(task):
    """Display all task results and artifacts"""
    print("\n" + "="*80)
    print("📊 TASK RESULTS")
    print("="*80)
    
    print(f"\nTask ID: {task['task_id']}")
    print(f"Context ID: {task['context_id']}")
    print(f"Status: {task['status']}")
    print(f"Created: {task.get('created_at', 'N/A')}")
    print(f"Updated: {task.get('updated_at', 'N/A')}")
    
    if task['status'] == 'failed':
        print(f"\n❌ Error: {task.get('error', 'Unknown error')}")
        return
    
    artifacts = task.get('artifacts', [])
    print(f"\n📦 Artifacts: {len(artifacts)} found")
    
    if not artifacts:
        print("   (No artifacts available yet)")
        return
    
    print("\n" + "="*80)
    print("ARTIFACT DETAILS")
    print("="*80)
    
    for i, artifact in enumerate(artifacts, 1):
        print(f"\n{'─'*80}")
        print(f"Artifact #{i}")
        print(f"{'─'*80}")
        
        part_type = artifact.get('part_type', 'unknown')
        metadata = artifact.get('metadata', {})
        artifact_type = metadata.get('type', 'unknown')
        
        print(f"Part Type: {part_type}")
        print(f"Type: {artifact_type}")
        
        if part_type == "TextPart":
            text = artifact.get('text', '')
            print(f"\n{'-'*80}")
            print("TEXT CONTENT:")
            print(f"{'-'*80}")
            print(text)
            print(f"{'-'*80}")
        
        elif part_type == "DataPart":
            data = artifact.get('data', {})
            
            if artifact_type == "gene_expression_data":
                genes = data.get('result', [])
                count = metadata.get('count', len(genes))
                print(f"Gene Count: {count}")
                print(f"\nFirst 10 genes:")
                for j, gene in enumerate(genes[:10], 1):
                    print(f"  {j:2d}. {gene}")
                if len(genes) > 10:
                    print(f"  ... and {len(genes) - 10} more")
            
            elif artifact_type == "pathway_enrichment":
                pathways = data.get('result', [])
                count = metadata.get('count', len(pathways))
                print(f"Pathway Count: {count}")
                print(f"\nTop pathways:")
                for j, pathway in enumerate(pathways[:5], 1):
                    print(f"  {j}. {pathway}")
                if len(pathways) > 5:
                    print(f"  ... and {len(pathways) - 5} more")
            
            elif artifact_type == "research_plan":
                plan = data.get('result', [])
                print(f"Plan Steps: {len(plan)}")
                for j, step in enumerate(plan, 1):
                    print(f"\n  Step {j}:")
                    print(f"    ID: {step.get('id')}")
                    print(f"    Description: {step.get('description')}")
                    print(f"    Agent: {step.get('assigned_agent')}")
                    print(f"    Status: {step.get('status')}")
            
            elif artifact_type == "visualization":
                viz = data.get('result', {})
                print(f"Filename: {viz.get('filename')}")
                print(f"Format: {viz.get('format')}")
                print(f"Encoding: {viz.get('encoding')}")
                data_size = len(viz.get('data', ''))
                print(f"Data Size: {data_size} characters (base64)")
            
            elif artifact_type == "langgraph_state":
                state = data.get('result', {})
                print(f"State Keys: {list(state.keys())}")
                for key, value in state.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"  {key}: {len(value)} keys")
                    else:
                        print(f"  {key}: {type(value).__name__}")
            
            else:
                # Generic data display
                print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
                print(f"\nData Preview:")
                print(json.dumps(data, indent=2)[:500] + "...")
        
        print()


def main():
    """Main execution"""
    print("="*80)
    print("FastA2A Full Query Test")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Use the same query from the main README
    query = "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?"
    
    # Submit the query
    task = submit_query(query)
    if not task:
        return
    
    task_id = task['task_id']
    
    # Poll until complete
    completed_task = poll_until_complete(task_id, poll_interval=5, timeout=600)
    if not completed_task:
        return
    
    # Display full results
    display_results(completed_task)
    
    print("\n" + "="*80)
    print("✅ Test Complete!")
    print("="*80)
    print(f"\nTask ID: {task_id}")
    print(f"View full results at: http://localhost:8000/tasks/{task_id}")
    print()


if __name__ == "__main__":
    main()
