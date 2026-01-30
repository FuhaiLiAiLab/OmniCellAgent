"""
Broker implementation for task scheduling and execution.

This module manages the task queue and coordinates worker execution.
"""

import asyncio
from typing import Dict, Optional
from datetime import datetime
import uuid

from fasta2a_service.storage import FileStorage, Task, TaskStatus
from fasta2a_service.worker import LangGraphWorker


class InMemoryBroker:
    """
    Simple in-memory broker for task scheduling.
    
    For production, consider using:
    - Celery with Redis/RabbitMQ
    - RQ (Redis Queue)
    - Dramatiq
    - Or integrate with existing task queue system
    """
    
    def __init__(self, storage: FileStorage, worker: LangGraphWorker, max_concurrent_tasks: int = 3):
        self.storage = storage
        self.worker = worker
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Active tasks being processed
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Task queue
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Background processor
        self.processor_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start the broker background processor"""
        if self.running:
            return
        
        self.running = True
        self.processor_task = asyncio.create_task(self._process_queue())
        print("🚀 Broker started")
    
    async def stop(self):
        """Stop the broker and wait for active tasks"""
        self.running = False
        
        # Wait for queue to empty
        await self.task_queue.join()
        
        # Cancel processor
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        print("🛑 Broker stopped")
    
    async def submit_task(self, prompt: str, context_id: Optional[str] = None) -> Task:
        """
        Submit a new task for execution.
        
        Args:
            prompt: User prompt/query
            context_id: Optional context ID for conversation continuity
        
        Returns:
            Created task
        """
        # Generate IDs
        task_id = str(uuid.uuid4())
        if context_id is None:
            context_id = str(uuid.uuid4())
        
        # Create task
        task = Task(
            task_id=task_id,
            context_id=context_id,
            prompt=prompt,
            status=TaskStatus.PENDING
        )
        
        # Save task
        self.storage.save_task(task)
        
        # Add to queue
        await self.task_queue.put(task_id)
        
        print(f"📥 Task {task_id} submitted to queue")
        return task
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get current status of a task"""
        return self.storage.load_task(task_id)
    
    async def _process_queue(self):
        """Background task processor"""
        while self.running:
            try:
                # Wait for task with timeout
                try:
                    task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if we can process (concurrent limit)
                while len(self.active_tasks) >= self.max_concurrent_tasks:
                    # Wait for a task to complete
                    done, pending = await asyncio.wait(
                        self.active_tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    # Clean up completed tasks
                    completed_ids = [
                        tid for tid, task in self.active_tasks.items()
                        if task in done
                    ]
                    for tid in completed_ids:
                        del self.active_tasks[tid]
                
                # Start processing task
                task = self.storage.load_task(task_id)
                if task:
                    exec_task = asyncio.create_task(self._execute_task(task))
                    self.active_tasks[task_id] = exec_task
                
                # Mark queue item as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in broker processor: {e}")
    
    async def _execute_task(self, task: Task):
        """Execute a single task"""
        try:
            print(f"🔄 Processing task {task.task_id}")
            
            # Load context if exists
            context = self.storage.load_context(task.context_id)
            context_state = context.langgraph_state if context else None
            
            # Execute
            updated_task = await self.worker.execute_task(task, context_state)
            
            if updated_task.status == TaskStatus.COMPLETED:
                print(f"✅ Task {task.task_id} completed")
            else:
                print(f"❌ Task {task.task_id} failed: {updated_task.error}")
            
        except Exception as e:
            print(f"💥 Task {task.task_id} exception: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.storage.save_task(task)
        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
