#!/bin/bash

# Start FastA2A Server for OmniCellAgent
# This script starts the A2A service in the background using nohup

eval "$(conda shell.bash hook)"
conda activate a2a-dev

cd /home/hao/BioProtocol/OmniCellAgent

echo "🚀 Starting FastA2A Server..."
echo "================================"

# Check if server is already running
if lsof -i:8000 -t >/dev/null 2>&1; then
    echo "⚠️  Server is already running on port 8000"
    echo "   To stop it, run: pkill -f 'uvicorn fasta2a_service.server'"
    exit 1
fi

# Start server in background
nohup python -m uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000 > fasta2a_service/server.log 2>&1 &

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if lsof -i:8000 -t >/dev/null 2>&1; then
    echo "✅ Server started successfully!"
    echo "   URL: http://localhost:8000"
    echo "   Logs: fasta2a_service/server.log"
    echo ""
    echo "To test the server, run:"
    echo "   ./fasta2a_service/run_test.sh"
    echo ""
    echo "To stop the server, run:"
    echo "   pkill -f 'uvicorn fasta2a_service.server'"
else
    echo "❌ Server failed to start. Check logs:"
    echo "   tail -f fasta2a_service/server.log"
    exit 1
fi
