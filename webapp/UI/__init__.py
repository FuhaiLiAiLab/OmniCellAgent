# UI module for webapp
# Contains UI adapters for different agent systems

# Export the LangGraph adapter
from .langgraph_adapter import UILangGraphAdapter, create_langgraph_adapter

__all__ = [
    'UILangGraphAdapter',
    'create_langgraph_adapter',
]
