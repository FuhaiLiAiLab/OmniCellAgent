#!/bin/bash

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Testing All Microservices"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test a service
test_service() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    local data=${4:-}
    local timeout=${5:-10}  # Default 10 seconds, can be overridden
    
    echo -n "Testing $name... "
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" \
            --connect-timeout 10 \
            --max-time $timeout 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" \
            --connect-timeout 10 \
            --max-time $timeout 2>&1)
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC} (HTTP $http_code)"
        if [ -n "$body" ]; then
            echo "  Response: ${body:0:100}..."
        fi
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $http_code)"
        if [ -n "$body" ]; then
            echo "  Error: ${body:0:100}..."
        fi
        return 1
    fi
}

# Wait a bit for services to be ready
echo -e "${YELLOW}Waiting 3 seconds for services to be ready...${NC}"
sleep 3
echo ""

# Test each service
total=0
passed=0

# 1. Neo4j (HTTP endpoint)
echo "1. Neo4j Database"
((total++))
if test_service "Neo4j HTTP" "http://localhost:7474"; then
    ((passed++))
fi
echo ""

# 2. Scientist RAG Tool (port 8000)
echo "2. Scientist RAG Tool (port 8000)"
((total++))
if test_service "Health Check" "http://localhost:8000/health"; then
    ((passed++))
fi
((total++))
if test_service "Authors List" "http://localhost:8000/authors"; then
    ((passed++))
fi
((total++))
test_query='{"author": "Scientist 1", "question": "What are the main research topics?"}'
if test_service "Query Endpoint" "http://localhost:8000/query" "POST" "$test_query" 30; then
    ((passed++))
fi
echo ""

# 3. GRetriever Service (port 8001)
echo "3. GRetriever Service (port 8001)"
((total++))
if test_service "Health Check" "http://localhost:8001/health"; then
    ((passed++))
fi
((total++))
echo "  (Note: GRetriever loads large models and may take 60+ seconds on first query)"
gretriever_query='{"query": "Alzheimer disease treatment", "max_nodes": 10, "include_description": false}'
if test_service "Query Endpoint" "http://localhost:8001/query" "POST" "$gretriever_query" 120; then
    ((passed++))
fi
echo ""

# 4. Omic Fetch Analysis Workflow (commented out - not started by default)
# echo "4. Omic Fetch Analysis Workflow"
# ((total++))
# if ps aux | grep -E "python.*omic_fetch_analysis_workflow_microservice" | grep -v grep > /dev/null; then
#     echo -e "${GREEN}✓ Process Running${NC}"
#     ((passed++))
# else
#     echo -e "${RED}✗ Process Not Running${NC}"
# fi
# echo ""

# 4. GLiNER Service
echo "4. GLiNER Service"
((total++))
if ps aux | grep -E "python.*gliner_service" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✓ Process Running${NC}"
    ((passed++))
else
    echo -e "${RED}✗ Process Not Running${NC}"
fi
echo ""

# 5. BioBERT Service
echo "5. BioBERT Service"
((total++))
if ps aux | grep -E "python.*biobert_service" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✓ Process Running${NC}"
    ((passed++))
else
    echo -e "${RED}✗ Process Not Running${NC}"
fi
echo ""

# 6. Webapp (port 8050)
echo "6. Webapp (port 8050)"
((total++))
if test_service "Webapp" "http://localhost:8050"; then
    ((passed++))
fi
echo ""

# 7. Ngrok Tunnel
echo "7. Ngrok Tunnel"
((total++))
if ps aux | grep -E "ngrok.*8050" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✓ Process Running${NC}"
    ((passed++))
    # Try to get ngrok status
    ngrok_status=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -n "$ngrok_status" ]; then
        echo "  Public URL: $ngrok_status"
    fi
else
    echo -e "${RED}✗ Process Not Running${NC}"
fi
echo ""

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $total"
echo -e "Passed: ${GREEN}$passed${NC}"
echo -e "Failed: ${RED}$((total - passed))${NC}"
echo ""

if [ $passed -eq $total ]; then
    echo -e "${GREEN}All services are running correctly!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some services failed or are not running.${NC}"
    echo "Check the logs in: $PROJECT_ROOT/logs/service-logs/"
    exit 1
fi
