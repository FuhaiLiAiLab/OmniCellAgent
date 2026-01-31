"""
FastA2A Server for OmniCellAgent

Simple A2A-compliant HTTP server exposing the LangGraph agent.
Run with: python -m uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000
"""

import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from fasta2a_service.utils import (
    FileStorage, LangGraphWorker, InMemoryBroker,
    Task, TaskStatus
)


# Global broker instance
broker: Optional[InMemoryBroker] = None


@asynccontextmanager
async def lifespan(app: Starlette):
    """Application lifespan manager"""
    global broker
    
    print("🚀 Starting FastA2A OmniCellAgent service...")
    
    # Initialize components
    storage = FileStorage(base_dir="./fasta2a_storage")
    worker = LangGraphWorker(storage=storage)  # Use agent's default model
    broker = InMemoryBroker(storage=storage, worker=worker, max_concurrent_tasks=2)
    
    # Start broker
    await broker.start()
    print("✅ Service ready")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    if broker:
        await broker.stop()
    print("👋 Service stopped")


# ============================================================================
# API Endpoints
# ============================================================================

async def create_task(request: Request):
    """POST /tasks - Create a new task"""
    data = await request.json()
    
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(
            {"error": "Missing 'prompt' field"},
            status_code=400
        )
    
    # Get or create context
    context_id = data.get("context_id") or str(uuid.uuid4())
    timeout = data.get("timeout", 2000)  # Default 2000 seconds
    
    # Create task
    task = Task(
        task_id=str(uuid.uuid4()),
        context_id=context_id,
        prompt=prompt,
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
        artifacts=[]
    )
    
    # Save and submit
    broker.storage.save_task(task)
    
    # Load context state if exists
    context_state = broker.storage.load_context(context_id)
    
    await broker.submit_task(task, metadata={
        'timeout': timeout,
        'context_state': context_state
    })
    
    return JSONResponse(task.to_dict(), status_code=201)


async def get_task(request: Request):
    """GET /tasks/{task_id} - Get task status and results"""
    task_id = request.path_params['task_id']
    
    task = broker.get_task_status(task_id)
    if not task:
        return JSONResponse(
            {"error": f"Task {task_id} not found"},
            status_code=404
        )
    
    return JSONResponse(task.to_dict())


async def list_tasks(request: Request):
    """GET /tasks - List all tasks"""
    context_id = request.query_params.get("context_id")
    
    tasks = broker.storage.list_tasks(context_id=context_id)
    
    return JSONResponse({
        "tasks": [task.to_dict() for task in tasks]
    })


async def get_context(request: Request):
    """GET /contexts/{context_id} - Get conversation context"""
    context_id = request.path_params['context_id']
    full_state = request.query_params.get("full_state", "false").lower() == "true"
    
    context = broker.storage.load_context(context_id)
    if not context:
        return JSONResponse(
            {"error": f"Context {context_id} not found"},
            status_code=404
        )
    
    if full_state:
        return JSONResponse(context)
    else:
        # Return summary
        return JSONResponse({
            "context_id": context_id,
            "has_state": True
        })


async def health_check(request: Request):
    """GET /health - Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "FastA2A OmniCellAgent",
        "broker_running": broker._running if broker else False
    })


# ============================================================================
# Application Setup
# ============================================================================

routes = [
    Route("/tasks", create_task, methods=["POST"]),
    Route("/tasks", list_tasks, methods=["GET"]),
    Route("/tasks/{task_id}", get_task, methods=["GET"]),
    Route("/contexts/{context_id}", get_context, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
]

app = Starlette(
    debug=True,
    routes=routes,
    lifespan=lifespan,
    middleware=[
        Middleware(CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
