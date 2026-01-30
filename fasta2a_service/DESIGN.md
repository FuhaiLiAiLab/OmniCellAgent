# FastA2A Service Design Considerations

## Beyond Simple Text Responses

When serving a complex LangGraph agent like OmniCellAgent through the FastA2A protocol, there are several important considerations beyond returning simple text responses. This document outlines the key design decisions and implementation details.

## 1. Structured State Management

### Challenge
The LangGraph agent maintains complex internal state including:
- Multi-step research plans with task breakdowns
- Intermediate results from multiple specialized agents
- Tool call history and responses
- Shared data structures (genes, pathways, papers)
- Process execution logs

### Solution: Dual Storage Architecture

**Task Storage (A2A Format)**
- Stores tasks in A2A protocol-compliant format
- Maintains message history in A2A message structure
- Provides artifacts in TextPart/DataPart format
- Ensures protocol compliance for interoperability

**Context Storage (Agent-Specific)**
- Stores complete LangGraph state for conversation continuity
- Preserves tool calls, intermediate results, and agent outputs
- Maintains accumulated data across multiple turns
- Enables true multi-turn conversations

### Implementation Details

```python
class Context:
    context_id: str
    langgraph_state: Dict[str, Any]  # Full agent state
    accumulated_genes: List[Dict]     # Cross-turn gene data
    accumulated_pathways: List[Dict]  # Cross-turn pathway data
    accumulated_papers: List[Dict]    # Cross-turn literature
```

## 2. Artifact Types and Structures

### TextPart Artifacts
**Final Report**
- Comprehensive markdown report
- Includes citations and references
- Contains analysis summary
- Metadata: `{"type": "final_report", "format": "markdown"}`

### DataPart Artifacts

**1. Research Plan**
```json
{
  "part_type": "DataPart",
  "data": {
    "result": [
      {
        "id": "task_1",
        "description": "Analyze gene expression",
        "assigned_agent": "OmicMiningAgent",
        "status": "completed",
        "result": "...",
        "attempts": 1
      }
    ]
  },
  "metadata": {
    "type": "research_plan",
    "schema": {...}
  }
}
```

**2. Gene Expression Data**
```json
{
  "part_type": "DataPart",
  "data": {
    "result": [
      {
        "gene_symbol": "EGFR",
        "log2_fold_change": 2.5,
        "p_value": 0.001,
        "expression_level": "upregulated"
      }
    ]
  },
  "metadata": {
    "type": "gene_expression_data",
    "count": 150
  }
}
```

**3. Pathway Enrichment**
```json
{
  "part_type": "DataPart",
  "data": {
    "result": [
      {
        "pathway_id": "hsa04010",
        "pathway_name": "MAPK signaling pathway",
        "p_value": 0.0001,
        "gene_count": 25
      }
    ]
  },
  "metadata": {
    "type": "pathway_enrichment"
  }
}
```

**4. Visualizations**
```json
{
  "part_type": "DataPart",
  "data": {
    "result": {
      "filename": "volcano_plot.png",
      "format": "png",
      "data": "base64_encoded_image_data...",
      "encoding": "base64"
    }
  },
  "metadata": {
    "type": "visualization",
    "plot_type": "volcano_plot"
  }
}
```

**5. Complete Agent State**
```json
{
  "part_type": "DataPart",
  "data": {
    "result": {
      "query": "...",
      "plan": [...],
      "agent_outputs": {...},
      "shared_data": {...},
      "process_log": [...]
    }
  },
  "metadata": {
    "type": "langgraph_state"
  }
}
```

## 3. Context Preservation Across Turns

### Challenge
Multi-turn conversations need access to:
- Previous analysis results
- Accumulated gene/pathway/paper lists
- Tool call history
- Intermediate computations

### Solution: Context Continuity

**On Each Turn:**
1. Load previous context state
2. Restore LangGraph state (if possible)
3. Make accumulated data available to agent
4. Execute new query with full context
5. Update context with new results
6. Save updated state

**Current Limitation:**
The current LangGraph agent creates fresh state on each invocation. To enable true state restoration:

```python
# TODO: Modify LangGraphOmniCellAgent to accept initial state
async def execute_task(self, task: Task, context_state: Optional[Dict]):
    if context_state:
        # Restore previous state
        initial_state = context_state.copy()
        initial_state["query"] = task.prompt
        # Continue from previous state
    else:
        # Fresh state
        initial_state = create_initial_state(task.prompt)
```

## 4. Asynchronous Execution

### Design Decisions

**Broker Pattern**
- Decouples task submission from execution
- Enables concurrent task processing
- Provides queue-based scheduling
- Handles backpressure

**Benefits:**
- Client gets immediate task ID
- Long-running analyses don't block
- Multiple clients can submit tasks
- Graceful handling of server restarts

**Production Considerations:**
```python
# Current: In-memory broker (simple but not persistent)
# Production options:
# - Celery with Redis/RabbitMQ
# - RQ (Redis Queue)
# - Dramatiq
# - Cloud task queues (AWS SQS, GCP Cloud Tasks)
```

## 5. Metadata and JSON Schemas

### Best Practices

**Include Schema Information:**
```python
Artifact(
    part_type="DataPart",
    data={"result": genes},
    metadata={
        "type": "gene_expression_data",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gene_symbol": {"type": "string"},
                    "log2_fold_change": {"type": "number"},
                    "p_value": {"type": "number"}
                }
            }
        }
    }
)
```

**Benefits:**
- Clients can validate data
- Enables automatic type checking
- Documents data structure
- Supports code generation

## 6. Error Handling

### Strategies

**Task-Level Errors:**
```python
task.status = TaskStatus.FAILED
task.error = {
    "message": "Analysis failed",
    "type": "OmicsAnalysisError",
    "details": {...},
    "traceback": "..."
}
```

**Partial Success:**
- Some sub-tasks succeed, others fail
- Return partial results with error annotations
- Client decides how to proceed

**Retry Logic:**
- Broker can retry failed tasks
- Exponential backoff
- Max retry limits

## 7. Performance Considerations

### Memory Management

**Large Artifacts:**
- Omics data can be huge (100K+ genes)
- Implement artifact size limits
- Paginate large results
- Stream large files separately

**Example:**
```python
if len(genes) > 1000:
    # Return top 1000 + summary
    artifact.data = {
        "result": genes[:1000],
        "total_count": len(genes),
        "truncated": True,
        "full_data_url": f"/artifacts/{artifact_id}/full"
    }
```

### Caching

**Omics Queries:**
- Cache common disease/cell-type combinations
- Reuse differential expression results
- Share pathway enrichment across users

**Implementation:**
```python
cache_key = f"{disease}:{cell_type}:{organ}"
if cache_key in omics_cache:
    return cached_result
```

## 8. Security Considerations

### Input Validation
```python
def validate_prompt(prompt: str):
    if len(prompt) > 10000:
        raise ValueError("Prompt too long")
    # Sanitize for injection attacks
    # Rate limiting
```

### Authentication
```python
# Add API key or OAuth
headers = {"Authorization": "Bearer <token>"}
```

### Resource Limits
```python
# Per-user quotas
# Concurrent task limits
# Storage limits per context
```

## 9. Monitoring and Observability

### Logging
```python
# Task lifecycle events
logger.info("task_submitted", task_id=id, user=user)
logger.info("task_started", task_id=id)
logger.info("task_completed", task_id=id, duration=t)

# Agent execution events
logger.info("agent_invoked", agent="OmicMiningAgent")
logger.info("tool_called", tool="pubmed_search")
```

### Metrics
```python
# Track:
# - Tasks per minute
# - Success/failure rates
# - Average execution time
# - Queue depth
# - Memory usage
```

### Tracing
```python
# OpenTelemetry integration
with tracer.start_span("execute_task"):
    with tracer.start_span("run_agent"):
        result = await agent.run(query)
```

## 10. Testing Strategies

### Unit Tests
```python
def test_artifact_extraction():
    state = create_test_state()
    artifacts = ArtifactExtractor.extract_all_artifacts(state)
    assert len(artifacts) > 0
    assert any(a.metadata["type"] == "gene_expression_data" for a in artifacts)
```

### Integration Tests
```python
async def test_full_workflow():
    task = await broker.submit_task("test query")
    result = await broker.get_task_status(task.task_id)
    assert result.status == TaskStatus.COMPLETED
```

### End-to-End Tests
```python
def test_client_workflow():
    client = A2AClient()
    task = client.submit_task("Find genes in cancer")
    result = client.wait_for_task(task["task_id"])
    assert "artifacts" in result
```

## Summary

Key considerations for FastA2A service beyond simple text:

1. **Dual storage**: A2A format + agent-specific state
2. **Rich artifacts**: Multiple typed DataPart artifacts with schemas
3. **Context continuity**: Preserve state across multi-turn conversations
4. **Async execution**: Broker pattern for long-running tasks
5. **Metadata**: Include schemas and type information
6. **Error handling**: Graceful failures with detailed errors
7. **Performance**: Caching, pagination, resource limits
8. **Security**: Validation, auth, rate limiting
9. **Observability**: Logging, metrics, tracing
10. **Testing**: Comprehensive test coverage

These considerations ensure the service is production-ready, maintainable, and provides a rich user experience beyond simple text responses.
