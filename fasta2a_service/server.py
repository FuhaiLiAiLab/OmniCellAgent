"""
FastA2A server implementation for OmniCellAgent.

This module provides an A2A-compliant HTTP server using Starlette.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from fasta2a_service.storage import FileStorage
from fasta2a_service.worker import LangGraphWorker
from fasta2a_service.broker import InMemoryBroker


# Global broker instance
broker: Optional[InMemoryBroker] = None


@asynccontextmanager
async def lifespan(app: Starlette):
    """Application lifespan manager"""
    global broker
    
    # Startup
    print("🚀 Starting FastA2A OmniCellAgent service...")
    
    # Initialize components
    storage = FileStorage(base_dir="./fasta2a_storage")
    worker = LangGraphWorker(storage=storage, model_name="gemini-2.0-flash-exp")
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


async def create_task(request: Request) -> JSONResponse:
    """
    POST /tasks
    Create and submit a new task.
    
    Request body:
    {
        "prompt": "What are the key genes in Alzheimer's disease?",
        "context_id": "optional-context-id"
    }
    """
    try:
        data = await request.json()
        prompt = data.get("prompt")
        context_id = data.get("context_id")
        
        if not prompt:
            return JSONResponse(
                {"error": "Missing required field: prompt"},
                status_code=400
            )
        
        # Submit task
        task = await broker.submit_task(prompt, context_id)
        
        return JSONResponse({
            "task_id": task.task_id,
            "context_id": task.context_id,
            "status": task.status.value,
            "created_at": task.created_at
        }, status_code=201)
        
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to create task: {str(e)}"},
            status_code=500
        )


async def get_task(request: Request) -> JSONResponse:
    """
    GET /tasks/{task_id}
    Get task status and results.
    """
    try:
        task_id = request.path_params["task_id"]
        task = await broker.get_task_status(task_id)
        
        if not task:
            return JSONResponse(
                {"error": "Task not found"},
                status_code=404
            )
        
        # Build response
        response_data = {
            "task_id": task.task_id,
            "context_id": task.context_id,
            "status": task.status.value,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }
        
        # Add completion info
        if task.status.value == "completed":
            response_data["completed_at"] = task.completed_at
            response_data["artifacts"] = [a.to_dict() for a in task.artifacts]
        elif task.status.value == "failed":
            response_data["error"] = task.error
        
        return JSONResponse(response_data)
        
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to get task: {str(e)}"},
            status_code=500
        )


async def list_tasks(request: Request) -> JSONResponse:
    """
    GET /tasks?context_id=optional
    List all tasks, optionally filtered by context_id.
    """
    try:
        context_id = request.query_params.get("context_id")
        tasks = broker.storage.list_tasks(context_id)
        
        return JSONResponse({
            "tasks": [
                {
                    "task_id": t.task_id,
                    "context_id": t.context_id,
                    "status": t.status.value,
                    "created_at": t.created_at,
                    "prompt": t.prompt[:100] + "..." if len(t.prompt) > 100 else t.prompt
                }
                for t in tasks
            ],
            "count": len(tasks)
        })
        
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to list tasks: {str(e)}"},
            status_code=500
        )


async def get_context(request: Request) -> JSONResponse:
    """
    GET /contexts/{context_id}
    Get conversation context including accumulated data.
    """
    try:
        context_id = request.path_params["context_id"]
        context = broker.storage.load_context(context_id)
        
        if not context:
            return JSONResponse(
                {"error": "Context not found"},
                status_code=404
            )
        
        # Return context (optionally exclude full state to reduce size)
        include_full_state = request.query_params.get("full_state", "false").lower() == "true"
        
        response_data = {
            "context_id": context.context_id,
            "created_at": context.created_at,
            "updated_at": context.updated_at,
            "task_ids": context.task_ids,
            "accumulated_genes": context.accumulated_genes,
            "accumulated_pathways": context.accumulated_pathways,
            "accumulated_papers": context.accumulated_papers,
        }
        
        if include_full_state:
            response_data["langgraph_state"] = context.langgraph_state
        
        return JSONResponse(response_data)
        
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to get context: {str(e)}"},
            status_code=500
        )


async def health_check(request: Request) -> JSONResponse:
    """GET / - Health check endpoint"""
    return JSONResponse({
        "service": "OmniCellAgent FastA2A",
        "status": "healthy",
        "version": "0.1.0"
    })


# Routes
routes = [
    Route("/", endpoint=health_check, methods=["GET"]),
    Route("/tasks", endpoint=create_task, methods=["POST"]),
    Route("/tasks", endpoint=list_tasks, methods=["GET"]),
    Route("/tasks/{task_id}", endpoint=get_task, methods=["GET"]),
    Route("/contexts/{context_id}", endpoint=get_context, methods=["GET"]),
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


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="FastA2A OmniCellAgent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(
        "fasta2a_service.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
