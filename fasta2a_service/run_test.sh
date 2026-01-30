#!/bin/bash

# Test script for A2A service
# Run this in a separate terminal while the server is running

eval "$(conda shell.bash hook)"
conda activate a2a-dev

cd /home/hao/BioProtocol/OmniCellAgent

echo "Testing A2A Service Connection..."
echo "=================================="

python fasta2a_service/test_a2a_connection.py
