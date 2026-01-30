# FastA2A OmniCellAgent Service - Quick Start Guide

## What is This?

This service exposes the OmniCellAgent LangGraph agent through the FastA2A (Agent-to-Agent) protocol, enabling:

- **Standard A2A communication** for interoperability with other agents
- **Structured outputs** beyond simple text (genes, pathways, plots, research plans)
- **Multi-turn conversations** with full context preservation
- **Asynchronous execution** for long-running biomedical analyses
- **Rich artifacts** with type information and JSON schemas

## Quick Start

### 1. Install Dependencies

```bash
# Run the setup script
./fasta2a_service/setup.sh

# Or manually:
conda run -n fasta2a-dev pip install -r fasta2a_service/requirements.txt
```

### 2. Start the Server

```bash
# Option 1: Using the server script
conda run -n fasta2a-dev python fasta2a_service/server.py --port 8000

# Option 2: Using uvicorn directly
conda run -n fasta2a-dev uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000

# Option 3: With auto-reload for development
conda run -n fasta2a-dev uvicorn fasta2a_service.server:app --reload
```

The service will be available at `http://localhost:8000`

### 3. Test with Example Client

```bash
conda run -n fasta2a-dev python fasta2a_service/examples/client_example.py
```

## API Usage Examples

See the full client example and documentation in the files.
