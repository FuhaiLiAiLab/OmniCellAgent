"""
Enhanced A2A server with multiple agent endpoints.

This server exposes:
1. Full LangGraph agent (complete pipeline)
2. Individual subagents (OmicMiningAgent, LiteratureAgent, etc.)

Each endpoint allows testing specific functionality independently.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from fasta2a_service.storage import FileStorage
from fasta2a_service.worker import LangGraphWorker
from fasta2a_service.broker import InMemoryBroker


# Global broker instances
full_agent_broker: Optional[InMemoryBroker] = None


@asynccontextmanager
async def lifespan(app: Starlette):
    """Application lifespan manager"""
    global full_agent_broker
    
    # Startup
    print("🚀 Starting Enhanced FastA2A OmniCellAgent service...")
    print("   Multiple agent endpoints available:")
    print("   - /tasks/full        - Complete LangGraph pipeline")
    print("   - /tasks/omic        - OmicMiningAgent only (future)")
    print("   - /tasks/literature  - LiteratureAgent only (future)")
    
    # Initialize storage and workers
    storage = FileStorage(base_dir="./fasta2a_storage")
    
    # Full agent worker (uses default LangGraph configuration)
    full_worker = LangGraphWorker(storage=storage, timeout=2000)  # 2000 sec timeout
    full_agent_broker = InMemoryBroker(
        storage=storage, 
        worker=full_worker, 
        max_concurrent_tasks=2
    )
    
    # Start brokers
    await full_agent_broker.start()
    
    print("✅ Service ready")
    print(f"   Timeout: 2000 seconds for long-running queries")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    if full_agent_broker:
        await full_agent_broker.stop()
    print("👋 Service stopped")


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "FastA2A OmniCellAgent",
        "endpoints": {
            "full_agent": "/tasks/full",
            "tasks_status": "/tasks/{task_id}",
            "list_tasks": "/tasks",
            "contexts": "/contexts/{context_id}"
        }
    })


async def submit_full_agent_task(request: Request) -> JSONResponse:
    """Submit task to full LangGraph agent"""
    try:
        data = await request.json()
        prompt = data.get("prompt")
        context_id = data.get("context_id")
        session_id = data.get("session_id")  # Optional session ID
        
        if not prompt:
            return JSONResponse(
                {"error": "Missing 'prompt' field"},
                status_code=400
            )
        
        # Generate session_id if not provided
        if not session_id:
            session_id = f"a2a_session_{uuid.uuid4().hex[:8]}"
        
        # Create task with session_id
        task = await full_agent_broker.submit_task(
            prompt=prompt,
            context_id=context_id,
            metadata={"session_id": session_id}
        )
        
        return JSONResponse(
            {
                "task_id": task.task_id,
                "context_id": task.context_id,
                "session_id": session_id,
                "status": task.status.value,
                "created_at": task.created_at.isoformat()
            },
            status_code=201
        )
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def get_task_status(request: Request) -> JSONResponse:
    """Get task status and results"""
    task_id = request.path_params["task_id"]
    
    try:
        task = full_agent_broker.storage.get_task(task_id)
        if not task:
            return JSONResponse(
                {"error": f"Task {task_id} not found"},
                status_code=404
            )
        
        response = {
            "task_id": task.task_id,
            "context_id": task.context_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }
        
        # Add session_id if available
        if task.metadata and "session_id" in task.metadata:
            response["session_id"] = task.metadata["session_id"]
        
        # Add error if failed
        if task.error:
            response["error"] = task.error
        
        # Add artifacts if completed
        if task.artifacts:
            response["artifacts"] = [
                {
                    "part_type": a.part_type,
                    **({{"text": a.data}} if a.part_type == "TextPart" else {"data": a.data}),
                    "metadata": a.metadata
                }
                for a in task.artifacts
            ]
        
        return JSONResponse(response)
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def list_tasks(request: Request) -> JSONResponse:
    """List all tasks"""
    try:
        context_id = request.query_params.get("context_id")
        tasks = full_agent_broker.storage.list_tasks(context_id=context_id)
        
        return JSONResponse({
            "tasks": [
                {
                    "task_id": t.task_id,
                    "context_id": t.context_id,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat(),
                    "session_id": t.metadata.get("session_id") if t.metadata else None
                }
                for t in tasks
            ]
        })
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def get_context(request: Request) -> JSONResponse:
    """Get conversation context"""
    context_id = request.path_params["context_id"]
    full_state = request.query_params.get("full_state", "false").lower() == "true"
    
    try:
        context = full_agent_broker.storage.get_context(context_id)
        if not context:
            return JSONResponse(
                {"error": f"Context {context_id} not found"},
                status_code=404
            )
        
        response = {
            "context_id": context.context_id,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        }
        
        if full_state and context.langgraph_state:
            response["langgraph_state"] = context.langgraph_state
        
        return JSONResponse(response)
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# Routes
routes = [
    Route("/", health_check, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/tasks/full", submit_full_agent_task, methods=["POST"]),
    Route("/tasks/{task_id}", get_task_status, methods=["GET"]),
    Route("/tasks", list_tasks, methods=["GET"]),
    Route("/contexts/{context_id}", get_context, methods=["GET"]),
]

# Middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

# Create app
app = Starlette(
    routes=routes,
    middleware=middleware,
    lifespan=lifespan
)


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="FastA2A OmniCellAgent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
