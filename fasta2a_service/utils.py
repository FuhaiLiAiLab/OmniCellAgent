"""
Utilities for FastA2A service: storage, worker, and broker.
Consolidates all backend logic for task execution and management.
"""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.langgraph_agent import LangGraphOmniCellAgent


# ============================================================================
# Storage Classes
# ============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Artifact:
    """A2A artifact (TextPart or DataPart)"""
    part_type: str  # "TextPart" or "DataPart"
    data: Any
    metadata: Dict[str, Any]


@dataclass
class Task:
    """A2A task representation"""
    task_id: str
    context_id: str
    prompt: str
    status: TaskStatus
    created_at: str
    updated_at: str
    artifacts: List[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self):
        result = asdict(self)
        result['status'] = self.status.value
        if self.artifacts is None:
            result['artifacts'] = []
        return result


class FileStorage:
    """Simple file-based storage for tasks and contexts"""
    
    def __init__(self, base_dir: str = "./fasta2a_storage"):
        self.base_dir = Path(base_dir)
        self.tasks_dir = self.base_dir / "tasks"
        self.contexts_dir = self.base_dir / "contexts"
        
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.contexts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_task(self, task: Task):
        """Save task to disk"""
        task_file = self.tasks_dir / f"{task.task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)
    
    def load_task(self, task_id: str) -> Optional[Task]:
        """Load task from disk"""
        task_file = self.tasks_dir / f"{task_id}.json"
        if not task_file.exists():
            return None
        
        with open(task_file, 'r') as f:
            data = json.load(f)
        
        return Task(
            task_id=data['task_id'],
            context_id=data['context_id'],
            prompt=data['prompt'],
            status=TaskStatus(data['status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            artifacts=data.get('artifacts', []),
            error=data.get('error')
        )
    
    def list_tasks(self, context_id: Optional[str] = None) -> List[Task]:
        """List all tasks, optionally filtered by context"""
        tasks = []
        for task_file in self.tasks_dir.glob("*.json"):
            task = self.load_task(task_file.stem)
            if task and (context_id is None or task.context_id == context_id):
                tasks.append(task)
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def save_context(self, context_id: str, state: Dict[str, Any]):
        """Save conversation context"""
        context_file = self.contexts_dir / f"{context_id}.json"
        with open(context_file, 'w') as f:
            json.dump({
                'context_id': context_id,
                'state': state,
                'updated_at': datetime.utcnow().isoformat()
            }, f, indent=2)
    
    def load_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation context"""
        context_file = self.contexts_dir / f"{context_id}.json"
        if not context_file.exists():
            return None
        
        with open(context_file, 'r') as f:
            data = json.load(f)
        return data.get('state')


# ============================================================================
# Worker
# ============================================================================

class LangGraphWorker:
    """Executes LangGraph agent tasks"""
    
    def __init__(self, storage: FileStorage, model_name: Optional[str] = None):
        self.storage = storage
        self.model_name = model_name
    
    async def execute_task(self, task: Task, context_state: Optional[Dict[str, Any]] = None, 
                          timeout: int = 2000) -> Task:
        """Execute a task with the LangGraph agent"""
        try:
            task.status = TaskStatus.RUNNING
            task.updated_at = datetime.utcnow().isoformat()
            self.storage.save_task(task)
            
            # Create agent
            if self.model_name:
                agent = LangGraphOmniCellAgent(
                    model_name=self.model_name,
                    session_id=task.context_id
                )
            else:
                agent = LangGraphOmniCellAgent(session_id=task.context_id)
            
            # Run with timeout
            try:
                report = await asyncio.wait_for(
                    agent.run(task.prompt),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"Task execution timed out after {timeout} seconds")
            
            # Create artifacts
            artifacts = []
            
            # Add text report
            if report:
                artifacts.append({
                    "part_type": "TextPart",
                    "text": report,
                    "metadata": {
                        "type": "final_report",
                        "format": "markdown"
                    }
                })
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.artifacts = artifacts
            task.updated_at = datetime.utcnow().isoformat()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.utcnow().isoformat()
        
        self.storage.save_task(task)
        return task


# ============================================================================
# Broker
# ============================================================================

class InMemoryBroker:
    """Manages task queue and execution"""
    
    def __init__(self, storage: FileStorage, worker: LangGraphWorker, 
                 max_concurrent_tasks: int = 2):
        self.storage = storage
        self.worker = worker
        self.max_concurrent_tasks = max_concurrent_tasks
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._worker_task = None
    
    async def start(self):
        """Start the broker worker"""
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        print("🚀 Broker started")
    
    async def stop(self):
        """Stop the broker worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        print("🛑 Broker stopped")
    
    async def submit_task(self, task: Task, metadata: Optional[Dict[str, Any]] = None):
        """Submit a task to the queue"""
        await self.queue.put((task, metadata or {}))
        print(f"📥 Task {task.task_id} submitted to queue")
    
    async def _process_queue(self):
        """Process tasks from the queue"""
        while self._running:
            try:
                # Clean up completed tasks
                completed = [tid for tid, t in self.running_tasks.items() if t.done()]
                for tid in completed:
                    del self.running_tasks[tid]
                
                # Start new tasks if under limit
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    try:
                        task, metadata = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=1.0
                        )
                        
                        # Start task execution
                        timeout = metadata.get('timeout', 2000)
                        context_state = metadata.get('context_state')
                        
                        task_coro = self.worker.execute_task(
                            task, 
                            context_state=context_state,
                            timeout=timeout
                        )
                        
                        self.running_tasks[task.task_id] = asyncio.create_task(task_coro)
                        print(f"🚀 Started executing task {task.task_id}")
                        
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                print(f"❌ Broker error: {e}")
                await asyncio.sleep(1.0)
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get current task status"""
        return self.storage.load_task(task_id)
