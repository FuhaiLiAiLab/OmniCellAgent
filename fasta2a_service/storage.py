"""
Storage implementation for FastA2A service.

This module provides storage for:
1. Tasks in A2A protocol format
2. Conversation context with full LangGraph state
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class TaskStatus(str, Enum):
    """A2A Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Artifact:
    """A2A Artifact"""
    part_type: str  # "TextPart" or "DataPart"
    data: Any
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "part_type": self.part_type,
        }
        if self.part_type == "TextPart":
            result["text"] = self.data
        else:  # DataPart
            result["data"] = self.data
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class Task:
    """A2A Task"""
    task_id: str
    context_id: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[Artifact] = field(default_factory=list)
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "context_id": self.context_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "message_history": self.message_history
        }


@dataclass
class Context:
    """Conversation context with full LangGraph state"""
    context_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    task_ids: List[str] = field(default_factory=list)
    
    # Full LangGraph state for context continuity
    langgraph_state: Optional[Dict[str, Any]] = None
    
    # Accumulated data across multiple tasks in this conversation
    accumulated_genes: List[Dict[str, Any]] = field(default_factory=list)
    accumulated_pathways: List[Dict[str, Any]] = field(default_factory=list)
    accumulated_papers: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "task_ids": self.task_ids,
            "langgraph_state": self.langgraph_state,
            "accumulated_genes": self.accumulated_genes,
            "accumulated_pathways": self.accumulated_pathways,
            "accumulated_papers": self.accumulated_papers
        }


class FileStorage:
    """File-based storage for tasks and contexts"""
    
    def __init__(self, base_dir: str = "./fasta2a_storage"):
        self.base_dir = Path(base_dir)
        self.tasks_dir = self.base_dir / "tasks"
        self.contexts_dir = self.base_dir / "contexts"
        
        # Create directories
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.contexts_dir.mkdir(parents=True, exist_ok=True)
    
    # Task operations
    def save_task(self, task: Task) -> None:
        """Save a task to storage"""
        task.updated_at = datetime.now().isoformat()
        task_file = self.tasks_dir / f"{task.task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task.to_dict(), f, indent=2, default=str)
    
    def load_task(self, task_id: str) -> Optional[Task]:
        """Load a task from storage"""
        task_file = self.tasks_dir / f"{task_id}.json"
        if not task_file.exists():
            return None
        
        with open(task_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct artifacts
        artifacts = []
        for a_data in data.get("artifacts", []):
            artifacts.append(Artifact(
                part_type=a_data["part_type"],
                data=a_data.get("text") if a_data["part_type"] == "TextPart" else a_data.get("data"),
                metadata=a_data.get("metadata")
            ))
        
        return Task(
            task_id=data["task_id"],
            context_id=data["context_id"],
            prompt=data["prompt"],
            status=TaskStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            artifacts=artifacts,
            message_history=data.get("message_history", [])
        )
    
    def list_tasks(self, context_id: Optional[str] = None) -> List[Task]:
        """List all tasks, optionally filtered by context_id"""
        tasks = []
        for task_file in self.tasks_dir.glob("*.json"):
            task = self.load_task(task_file.stem)
            if task and (context_id is None or task.context_id == context_id):
                tasks.append(task)
        return sorted(tasks, key=lambda t: t.created_at)
    
    # Context operations
    def save_context(self, context: Context) -> None:
        """Save a context to storage"""
        context.updated_at = datetime.now().isoformat()
        context_file = self.contexts_dir / f"{context.context_id}.json"
        with open(context_file, 'w') as f:
            json.dump(context.to_dict(), f, indent=2, default=str)
    
    def load_context(self, context_id: str) -> Optional[Context]:
        """Load a context from storage"""
        context_file = self.contexts_dir / f"{context_id}.json"
        if not context_file.exists():
            return None
        
        with open(context_file, 'r') as f:
            data = json.load(f)
        
        return Context(
            context_id=data["context_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            task_ids=data.get("task_ids", []),
            langgraph_state=data.get("langgraph_state"),
            accumulated_genes=data.get("accumulated_genes", []),
            accumulated_pathways=data.get("accumulated_pathways", []),
            accumulated_papers=data.get("accumulated_papers", [])
        )
    
    def create_context(self, context_id: str) -> Context:
        """Create a new context"""
        context = Context(context_id=context_id)
        self.save_context(context)
        return context
    
    def update_context_with_task(self, context_id: str, task_id: str, langgraph_state: Dict[str, Any]) -> None:
        """Update context with task results"""
        context = self.load_context(context_id)
        if not context:
            context = self.create_context(context_id)
        
        # Add task to context
        if task_id not in context.task_ids:
            context.task_ids.append(task_id)
        
        # Update LangGraph state
        context.langgraph_state = langgraph_state
        
        # Accumulate structured data from shared_data
        shared_data = langgraph_state.get("shared_data", {})
        
        # Accumulate genes
        if shared_data.get("top_genes"):
            for gene in shared_data["top_genes"]:
                if gene not in context.accumulated_genes:
                    context.accumulated_genes.append(gene)
        
        # Accumulate pathways
        if shared_data.get("pathways"):
            for pathway in shared_data["pathways"]:
                if pathway not in context.accumulated_pathways:
                    context.accumulated_pathways.append(pathway)
        
        # Accumulate papers
        if shared_data.get("paper_dois"):
            for doi in shared_data["paper_dois"]:
                if doi not in context.accumulated_papers:
                    context.accumulated_papers.append(doi)
        
        self.save_context(context)
