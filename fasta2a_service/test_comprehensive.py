#!/usr/bin/env python3
"""
Comprehensive A2A Test Suite
Tests both individual components and the full agent, waiting for completion.
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

def wait_for_completion(task_id, poll_interval=10, timeout=2000):
    """
    Wait for task to complete and return full results.
    
    Args:
        task_id: Task ID to monitor
        poll_interval: Seconds between polls
        timeout: Max seconds to wait
    """
    print(f"\n⏳ Waiting for completion (timeout: {timeout}s, poll every {poll_interval}s)")
    print(f"   This may take 5-30 minutes for full biomedical analysis...")
    
    start_time = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        
        try:
            resp = requests.get(f"{BASE_URL}/tasks/{task_id}", timeout=30)
            if resp.status_code != 200:
                print(f"❌ Error getting status: {resp.status_code}")
                return None
            
            task = resp.json()
            status = task['status']
            
            # Show progress
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{iteration:3d}] {timestamp} | Status: {status:12s} | "
                  f"Elapsed: {int(elapsed):4d}s / {timeout}s", flush=True)
            
            if status == 'completed':
                print(f"\n✅ Task completed in {int(elapsed)}s ({elapsed/60:.1f} minutes)!")
                return task
            
            elif status == 'failed':
                print(f"\n❌ Task failed after {int(elapsed)}s")
                print(f"   Error: {task.get('error', 'Unknown error')}")
                return task
            
            if elapsed > timeout:
                print(f"\n⚠️  Timeout after {timeout}s")
                return task
            
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"   Error polling: {e}")
            time.sleep(poll_interval)

def display_artifacts(task):
    """Display all artifacts from completed task"""
    if not task or task['status'] != 'completed':
        return
    
    artifacts = task.get('artifacts', [])
    
    print_section(f"ARTIFACTS ({len(artifacts)} found)")
    
    for i, artifact in enumerate(artifacts, 1):
        print(f"\n{'─'*80}")
        print(f"Artifact #{i}: {artifact['part_type']}")
        
        metadata = artifact.get('metadata', {})
        artifact_type = metadata.get('type', 'unknown')
        print(f"Type: {artifact_type}")
        
        if artifact['part_type'] == 'TextPart':
            text = artifact.get('text', '')
            print(f"\n{'─'*40} CONTENT {'─'*40}")
            print(text)
            print('─'*80)
        
        elif artifact['part_type'] == 'DataPart':
            data = artifact.get('data', {})
            
            if artifact_type == 'research_plan':
                plan = data.get('result', [])
                print(f"\nResearch Plan ({len(plan)} steps):")
                for j, step in enumerate(plan, 1):
                    print(f"\n  Step {j}: {step.get('description')}")
                    print(f"    Agent: {step.get('assigned_agent')}")
                    print(f"    Status: {step.get('status')}")
                    result = step.get('result', '')
                    if result:
                        preview = result[:200] + ('...' if len(result) > 200 else '')
                        print(f"    Result: {preview}")
            
            elif artifact_type == 'gene_expression_data':
                genes = data.get('result', [])
                count = metadata.get('count', len(genes))
                print(f"\nGene Expression Data ({count} genes):")
                print(f"First 20 genes:")
                for gene in genes[:20]:
                    print(f"  - {gene}")
                if len(genes) > 20:
                    print(f"  ... and {len(genes) - 20} more")
            
            elif artifact_type == 'pathway_enrichment':
                pathways = data.get('result', [])
                count = metadata.get('count', len(pathways))
                print(f"\nPathway Enrichment ({count} pathways):")
                for j, pathway in enumerate(pathways[:10], 1):
                    print(f"  {j}. {pathway}")
                if len(pathways) > 10:
                    print(f"  ... and {len(pathways) - 10} more")
            
            elif artifact_type == 'literature_references':
                papers = data.get('result', [])
                count = metadata.get('count', len(papers))
                print(f"\nLiterature References ({count} papers):")
                for j, paper in enumerate(papers[:10], 1):
                    print(f"  {j}. {paper}")
                if len(papers) > 10:
                    print(f"  ... and {len(papers) - 10} more")
            
            elif artifact_type == 'visualization':
                viz = data.get('result', {})
                print(f"\nVisualization:")
                print(f"  File: {viz.get('filename')}")
                print(f"  Format: {viz.get('format')}")
                print(f"  Size: {len(viz.get('data', ''))} bytes (base64)")
            
            elif artifact_type == 'langgraph_state':
                state = data.get('result', {})
                print(f"\nLangGraph State:")
                for key in state.keys():
                    value = state[key]
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"  {key}: {len(value)} keys")
                    else:
                        print(f"  {key}: {type(value).__name__}")
            
            elif artifact_type == 'process_log':
                logs = data.get('result', [])
                print(f"\nProcess Log ({len(logs)} entries):")
                for j, log in enumerate(logs[:5], 1):
                    print(f"  {j}. {log}")
                if len(logs) > 5:
                    print(f"  ... and {len(logs) - 5} more")
            
            else:
                print(f"\nMetadata:")
                print(json.dumps(metadata, indent=2))
                print(f"\nData keys: {list(data.keys())}")

def test_component(name, query, session_id, timeout=600):
    """
    Test an individual component.
    For now, all go through /tasks endpoint (future: separate endpoints)
    """
    print_section(f"TEST: {name}")
    print(f"Query: {query}")
    print(f"Session: {session_id}")
    
    resp = requests.post(
        f"{BASE_URL}/tasks",
        json={
            "prompt": query,
            "session_id": session_id
        },
        timeout=30
    )
    
    if resp.status_code not in [200, 201]:
        print(f"❌ Failed to submit: {resp.status_code}")
        print(f"   Response: {resp.text}")
        return None
    
    task = resp.json()
    task_id = task['task_id']
    print(f"\n✅ Task submitted: {task_id}")
    
    # Wait for completion
    result = wait_for_completion(task_id, poll_interval=10, timeout=timeout)
    
    if result:
        display_artifacts(result)
    
    return result

def test_full_agent(query, session_id, timeout=2000):
    """Test the complete LangGraph agent pipeline"""
    print_section(f"TEST: FULL AGENT PIPELINE")
    print(f"Query: {query}")
    print(f"Session: {session_id}")
    print(f"Timeout: {timeout}s (~33 minutes)")
    
    resp = requests.post(
        f"{BASE_URL}/tasks",
        json={
            "prompt": query,
            "session_id": session_id
        },
        timeout=30
    )
    
    if resp.status_code not in [200, 201]:
        print(f"❌ Failed to submit: {resp.status_code}")
        print(f"   Response: {resp.text}")
        return None
    
    task = resp.json()
    task_id = task['task_id']
    print(f"\n✅ Task submitted: {task_id}")
    print(f"   Context: {task['context_id']}")
    print(f"   Session: {task.get('session_id', 'N/A')}")
    
    # Wait for completion with full timeout
    result = wait_for_completion(task_id, poll_interval=15, timeout=timeout)
    
    if result:
        display_artifacts(result)
        
        # Show where results are saved
        session_id = result.get('session_id')
        if session_id:
            print_section("OUTPUT LOCATION")
            print(f"Session results saved in:")
            print(f"  webapp/sessions/{session_id}/")
            print(f"\nIncludes:")
            print(f"  - Differential expression analysis")
            print(f"  - Volcano plots")
            print(f"  - Enrichment results")
            print(f"  - KEGG pathway plots")
            print(f"  - Gene lists")
    
    return result

def main():
    """Run comprehensive test suite"""
    print("="*80)
    print("  FastA2A Comprehensive Test Suite")
    print("  Testing Components & Full Agent Pipeline")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {BASE_URL}")
    
    # Health check
    print_section("HEALTH CHECK")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        health = resp.json()
        print(f"Status: {health}")
        if not health.get('broker_running'):
            print("⚠️  Warning: Broker not running!")
            return
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return
    
    # Component Tests (shorter timeout for simpler queries)
    # Uncomment to test individual components
    """
    print("\n" + "#"*80)
    print("# COMPONENT TESTS")
    print("#"*80)
    
    test_component(
        name="Literature Search",
        query="Find recent papers about KRAS mutations in pancreatic cancer",
        session_id="lit-test-a2a",
        timeout=300
    )
    
    test_component(
        name="Gene Query",
        query="What is the function of the TP53 gene?",
        session_id="gene-test-a2a",
        timeout=300
    )
    """
    
    # Full Agent Test (long timeout for complete analysis)
    print("\n" + "#"*80)
    print("# FULL AGENT PIPELINE TEST")
    print("#"*80)
    
    test_full_agent(
        query="What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?",
        session_id="PDAC-full-a2a",
        timeout=2000  # 33+ minutes
    )
    
    print_section("TEST SUITE COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
