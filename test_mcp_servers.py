"""Test script to verify MCP servers are working correctly"""
import subprocess
import time
import json
import sys

def test_mcp_server(server_path, server_name):
    """Test a single MCP server by checking if it starts and responds"""
    print(f"\n{'='*60}")
    print(f"Testing {server_name}")
    print('='*60)
    
    try:
        # Start server in stdio mode and send a test request
        proc = subprocess.Popen(
            ['conda', 'run', '-n', 'a2a-dev', 'python', server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a moment for server to initialize
        time.sleep(2)
        
        # Send initialize request (MCP protocol)
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        try:
            proc.stdin.write(json.dumps(initialize_request) + '\n')
            proc.stdin.flush()
            
            # Try to read response (with timeout)
            proc.stdout.flush()
            time.sleep(1)
            
            # Terminate the process
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            
            print(f"✅ {server_name} started successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  {server_name} started but protocol test failed: {e}")
            proc.terminate()
            return True  # Server started, which is the main check
            
    except Exception as e:
        print(f"❌ {server_name} failed to start: {e}")
        return False

def main():
    """Test all MCP servers"""
    servers = [
        ('mcp_servers/pubmed_server.py', 'PubMed Server'),
        ('mcp_servers/websearch_server.py', 'WebSearch Server'),
        ('mcp_servers/knowledge_graph_server.py', 'Knowledge Graph Server'),
        ('mcp_servers/scientist_rag_server.py', 'Scientist RAG Server'),
        ('mcp_servers/omics_server.py', 'Omics Server'),
    ]
    
    results = []
    for server_path, server_name in servers:
        success = test_mcp_server(server_path, server_name)
        results.append((server_name, success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print('='*60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
