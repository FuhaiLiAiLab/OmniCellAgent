#!/bin/bash

# FastA2A Service Setup Script for OmniCellAgent
# This script installs dependencies and sets up the service

echo "🚀 Setting up FastA2A Service for OmniCellAgent"
echo "================================================"

# Check if conda environment exists
CONDA_ENV="fasta2a-dev"

if [ ! -d "$CONDA_ENV" ]; then
    echo "❌ Error: Conda environment not found at $CONDA_ENV"
    echo "Please create the environment first"
    exit 1
fi

echo "✅ Found conda environment: $CONDA_ENV"

# Install core dependencies
echo ""
echo "📦 Installing core dependencies..."
conda run -n "$CONDA_ENV" pip install -r fasta2a_service/requirements.txt
# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Core dependencies installed"
else
    echo "❌ Failed to install core dependencies"
    exit 1
fi

# Optional: Install official fasta2a package
echo ""
read -p "Install official fasta2a package? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing fasta2a..."
    conda run -n "$CONDA_ENV" pip install fasta2a
    
    if [ $? -eq 0 ]; then
        echo "✅ fasta2a installed"
    else
        echo "⚠️  fasta2a installation failed (optional, service will still work)"
    fi
fi

# Create storage directory
echo ""
echo "📁 Creating storage directory..."
mkdir -p ./fasta2a_storage/tasks
mkdir -p ./fasta2a_storage/contexts
echo "✅ Storage directory created: ./fasta2a_storage/"

# Test imports
echo ""
echo "🧪 Testing imports..."
conda run -n "$CONDA_ENV" python -c "
import sys
sys.path.insert(0, '.')
from fasta2a_service.storage import FileStorage
from fasta2a_service.worker import LangGraphWorker
from fasta2a_service.broker import InMemoryBroker
from fasta2a_service.server import app
print('✅ All imports successful')
" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Service is ready"
else
    echo "❌ Import test failed - please check dependencies"
    exit 1
fi

# Print usage instructions
echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "To start the service:"
echo "  conda run -n $CONDA_ENV python fasta2a_service/server.py --port 8000"
echo ""
echo "Or with uvicorn:"
echo "  conda run -n $CONDA_ENV uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000"
echo ""
echo "To test the service:"
echo "  conda run -n $CONDA_ENV python fasta2a_service/examples/client_example.py"
echo ""
echo "API will be available at: http://localhost:8000"
echo ""
