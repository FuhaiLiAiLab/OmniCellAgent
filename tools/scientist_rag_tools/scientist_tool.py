"""
author_kb_tool.py - Build and query author-specific scientific knowledge bases using RAG.
FastAPI server implementation for high concurrent loads.
"""

import sys
import os
import json
import argparse

# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio
from dotenv import load_dotenv
import re

from typing import List, Dict, Any, Optional

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn[standard]")
    exit(1)

# Now the tools module can be found because we added the project root to sys.path
from tools.pubmed_tools.get_papers_info_tools import get_papers_info

from paperscraper.pubmed import get_and_dump_pubmed_papers
from paperscraper.pdf import save_pdf_from_dump

import numpy as np
# Modified imports to work with sys.path changes
from hyperrag import HyperRAG, QueryParam
from hyperrag.utils import EmbeddingFunc
from hyperrag.llm import openai_embedding, openai_complete_if_cache

# Import path config
try:
    from utils.path_config import get_path
except ImportError:
    # Fallback for when running as standalone
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.path_config import get_path

load_dotenv(override=True)  

# Constants
BASE_AUTHOR_DIR = get_path('cache.author_kb', absolute=True, create=True)
DEFAULT_TOP_K = 50

LLM_BASE_URL = os.getenv("OPENAI_API_BASE")
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4.1"

EMB_BASE_URL = os.getenv("OPENAI_API_BASE")
EMB_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    try:
        return await openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            **kwargs,
        )
    except Exception as e:
        print(f"LLM API call failed: {e}")
        raise

async def embedding_func(texts: list[str]) -> np.ndarray:
    try:
        return await openai_embedding(
            texts,
            model=EMB_MODEL,
            api_key=EMB_API_KEY,
            base_url=EMB_BASE_URL,
        )
    except Exception as e:
        print(f"Embedding API call failed: {e}")
        raise


# ---------- Knowledge Base Builder ----------

class AuthorKnowledgeBase:
    """
    Build and query a scientific knowledge base from an author's publications.
    """

    def __init__(self, author_name: str, top_k: int = DEFAULT_TOP_K):
        self.author_name = author_name
        self.author_dir_name = self._sanitize_name(author_name)
        self.top_k = top_k

        self.get_papers_info = get_papers_info

        # Directory setup
        self.working_dir = os.path.join(BASE_AUTHOR_DIR, self.author_dir_name, "ragstore")
        self.jsonl_cache_dir = os.path.join(BASE_AUTHOR_DIR, self.author_dir_name, "jsonl_cache")
        self.doi_cache_dir = os.path.join(BASE_AUTHOR_DIR, self.author_dir_name, "doi_cache")

        for d in [self.working_dir, self.jsonl_cache_dir, self.doi_cache_dir]:
            os.makedirs(d, exist_ok=True)

        self.rag = None

    async def initialize_rag(self):
        if (self.rag is None):
            self.rag = HyperRAG(
                working_dir=self.working_dir,
                    embedding_func=EmbeddingFunc(
                    embedding_dim=EMB_DIM, max_token_size=8192, func=embedding_func
                ),
                llm_model_func=llm_model_func,
            )
            print(f"RAG initialized for {self.author_name}")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^\w\s-]", "-", name).replace(" ", "_").lower()

    def _pubmed_author_query(self) -> List[str]:
        return [f"{self.author_name}[Author]"]

    async def fetch_and_cache_papers(self) -> str:
        query_terms = self._pubmed_author_query()
        output_path = os.path.join(self.jsonl_cache_dir, f"{self.author_dir_name}.jsonl")

        print(f"🔍 Querying PubMed for papers by {self.author_name}...")
        get_and_dump_pubmed_papers(query_terms, output_filepath=output_path)

        with open(output_path, 'r') as f:
            lines = f.readlines()[:self.top_k]

        with open(output_path, 'w') as f:
            f.writelines(lines)

        save_pdf_from_dump(output_path, pdf_path=self.doi_cache_dir, key_to_save='doi')
        return self.doi_cache_dir

    async def insert_papers_to_rag(self, papers: List[Dict[str, Any]]):
        if not self.rag:
            raise ValueError("RAG not initialized.")

        # Create tasks for parallel insertion
        tasks = []
        for paper in papers:
            if not paper["text"] or not paper["file_path"]:
                continue
            tasks.append(self.rag.ainsert(paper["text"]))
        
        # Use tqdm to create a progress bar
        from tqdm import tqdm
        total = len(tasks)
        
        # Process in batches to prevent overwhelming resources
        batch_size = 5  # Adjust based on your system capacity
        results = []
        
        with tqdm(total=total, desc=f"Inserting papers for {self.author_name}") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Check for errors
                for j, result in enumerate(batch_results):
                    paper_idx = i + j
                    if paper_idx < len(papers) and isinstance(result, Exception):
                        print(f"[!] Insertion failed for {papers[paper_idx]['file_path']}: {result}")
                
                results.extend(batch_results)
                pbar.update(len(batch))
                
                # Small pause between batches to allow resource cleanup
                await asyncio.sleep(0.1)
        
        return results

    async def build_knowledge_base(self, paper_dir: str = None):
        await self.initialize_rag()
        
        if os.path.exists(self.working_dir) and os.listdir(self.working_dir):
            return
        else:
            paper_dir = await self.fetch_and_cache_papers()
            papers = await self.get_papers_info(paper_dir)
            await self.insert_papers_to_rag(papers)
            print(f"✅ Knowledge base built for author: {self.author_name}")

    async def query_system(self, question: str) -> str:
        """
        Asynchronously query the knowledge base.
        """
        try:
            result = await self.rag.aquery(
                question, 
                param=QueryParam(mode='hyper', only_need_context=False)
            )
            return result
        except Exception as e:
            error_msg = f"Error querying {self.author_name}'s knowledge base asynchronously: {str(e)}"
            print(error_msg)
            return error_msg

async def scientist_rag_retrieval_tool(author_name: str, question: str, kb_dict, display_name: str = None):
    """
    Query a scientist's knowledge base using RAG.
    Let HyperRAG handle its own timeouts and error management.
    """
    # Use display_name for privacy protection in logs, fallback to author_name if not provided
    name_for_display = display_name if display_name else author_name
    print(f"--- Querying ({question}) in {name_for_display}'s knowledge base ---")
    
    if author_name not in kb_dict:
        error_msg = f"Author '{name_for_display}' not found in knowledge base"
        print(error_msg)
        return {"error": error_msg}
    
    try:
        # Trust HyperRAG to handle its own timeouts and rate limiting
        result = await kb_dict[author_name].query_system(question)
        return {"result": result, "author": name_for_display}
    except Exception as e:
        error_msg = f"Error querying {name_for_display}'s knowledge base: {str(e)}"
        print(error_msg)
        return {"error": error_msg}


# Global knowledge base dictionary for the tool
_global_kb_dict = None

# Privacy protection: mapping real names to anonymized identifiers
_author_mapping = {}
_reverse_mapping = {}

def initialize_author_mapping(real_authors: List[str]) -> Dict[str, str]:
    """
    Create privacy-protected mapping from real author names to anonymized identifiers.
    
    Args:
        real_authors: List of real author names
        
    Returns:
        Dictionary mapping real names to anonymized names
    """
    global _author_mapping, _reverse_mapping
    
    if not _author_mapping:
        for i, author in enumerate(real_authors, 1):
            anonymous_name = f"Scientist {i}"
            _author_mapping[author] = anonymous_name
            _reverse_mapping[anonymous_name] = author
            
    
    return _author_mapping

def get_real_author_name(anonymous_name: str) -> str:
    """Convert anonymous name back to real name for internal processing."""
    return _reverse_mapping.get(anonymous_name, anonymous_name)

def get_anonymous_author_name(real_name: str) -> str:
    """Convert real name to anonymous name for external display."""
    return _author_mapping.get(real_name, real_name)

def list_available_scientists() -> List[str]:
    """Get list of available anonymized scientist names."""
    return list(_reverse_mapping.keys())

async def initialize_scientist_kb():
    """Initialize the scientist knowledge bases globally."""
    global _global_kb_dict
    if _global_kb_dict is None:
        print("Initializing scientist knowledge bases...")
        _global_kb_dict = await initialize_all_authors()
        
        # Initialize privacy mapping
        real_authors = list(_global_kb_dict.keys())
        initialize_author_mapping(real_authors)
        
    return _global_kb_dict

async def scientist_rag_tool_wrapper(author_name: str, question: str) -> str:
    """
    Wrapper function for AutoGen FunctionTool that handles kb_dict internally.
    This function only takes author_name and question as parameters.
    
    Parameters:
    - author_name (str): Name of the scientist/author to query (use display names like "Neuroscience Expert")
    - question (str): The scientific question to ask
    
    Returns:
    - str: The response from the scientist's knowledge base or error message
    
    Available scientists:
    - Neuroscience Expert (Alzheimer's disease research, amyloid-beta, tau proteins, neurodegeneration)
    
    Example usage:
    scientist_rag_tool_wrapper("Neuroscience Expert", "What are the latest findings on Alzheimer's disease?")
    """
    try:
        # Initialize KB if not already done
        kb_dict = await initialize_scientist_kb()
        
        # The author_name is the display name which is used directly as the key
        if author_name not in kb_dict:
            # Provide helpful error message with available scientists
            available = list(kb_dict.keys())
            return f"Error: Author '{author_name}' not found. Available scientists: {', '.join(available)}"
        
        # Call the actual tool with the display name
        result = await scientist_rag_retrieval_tool(author_name, question, kb_dict, display_name=author_name)
        
        # Format the response for the LLM
        if "error" in result:
            return f"Error: {result['error']}"
        else:
            return f"Retrieved from {author_name}'s knowledge base:\n\n{result['result']}"
            
    except Exception as e:
        error_msg = f"Failed to query scientist knowledge base: {str(e)}"
        print(error_msg)
        return error_msg


# ---------- FastAPI Server Implementation ----------

class QueryRequest(BaseModel):
    author: str
    question: str

class QueryResponse(BaseModel):
    result: Optional[str] = None
    error: Optional[str] = None
    author: Optional[str] = None

# Global knowledge base dictionary for FastAPI
kb_dict: Dict[str, AuthorKnowledgeBase] = {}

def create_fastapi_app():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Scientist RAG API",
        description="API for querying scientific knowledge bases by author",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """Initialize knowledge bases on startup"""
        global kb_dict
        print("Initializing knowledge bases for FastAPI server...")
        try:
            kb_dict = await initialize_all_authors()
            
            # Initialize privacy mapping
            real_authors = list(kb_dict.keys())
            initialize_author_mapping(real_authors)
            
            print(f"Successfully initialized {len(kb_dict)} knowledge bases")
            print("Available scientists (anonymized):")
            for anon_name in list_available_scientists():
                print(f"  - {anon_name}")
        except Exception as e:
            print(f"Error initializing knowledge bases: {e}")
            raise

    @app.post("/query", response_model=QueryResponse)
    async def query_knowledge_base(request: QueryRequest):
        """Query a specific author's knowledge base"""
        
        # Convert anonymous name to real name for internal processing
        real_author_name = get_real_author_name(request.author)
        
        if real_author_name not in kb_dict:
            # Provide helpful error message with available anonymized scientists
            available = list_available_scientists()
            raise HTTPException(
                status_code=404, 
                detail=f"Author '{request.author}' not found. Available scientists: {', '.join(available)}"
            )
        
        try:
            print(f"Processing query for {request.author} (internal: {real_author_name}): {request.question}")
            
            # Trust HyperRAG to handle its own timeouts and processing
            result = await kb_dict[real_author_name].query_system(request.question)
            
            return QueryResponse(
                result=result,
                author=request.author  # Return the anonymized name
            )
            
        except Exception as e:
            print(f"Error querying {request.author}'s knowledge base: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error querying knowledge base: {str(e)}"
            )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "available_authors": list_available_scientists(),
            "total_authors": len(kb_dict)
        }

    @app.get("/authors")
    async def list_authors():
        """List available authors"""
        return {
            "authors": list_available_scientists(),
            "total": len(kb_dict)
        }

    return app


def run_server(port=8000):
    """Run the FastAPI server"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI is not installed. Please install with: pip install fastapi uvicorn[standard]")
        return
    
    app = create_fastapi_app()
    
    print(f"Starting FastAPI server on port {port}...")
    print("Available endpoints:")
    print(f"  POST http://localhost:{port}/query - Query knowledge base")
    print(f"  GET  http://localhost:{port}/health - Health check")
    print(f"  GET  http://localhost:{port}/authors - List available authors")
    print(f"  GET  http://localhost:{port}/docs - Interactive API documentation")
    
    # Simple configuration - let HyperRAG handle concurrency and rate limiting
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,  # Use 1 worker to share kb_dict
        loop="asyncio",
        access_log=True
    )


# Mapping from anonymized display names to internal database names
SCIENTIST_ALIAS_MAP = {
    "Neuroscience Expert": "neuroscience_expert",  # Maps display name to database folder name
}

async def initialize_all_authors():
    scientists = [
        {"name": "neuroscience_expert", "display_name": "Neuroscience Expert", "top_k": 25},
    ]

    kb_dict = {}
    for scientist in scientists:
        kb = AuthorKnowledgeBase(scientist["name"], top_k=scientist["top_k"])
        await kb.build_knowledge_base(paper_dir=kb.doi_cache_dir)
        # Use display_name as key for external API access
        kb_dict[scientist["display_name"]] = kb
    return kb_dict


async def test_concurrent_queries(num_queries: int):
    """
    Test concurrent queries to validate performance and HyperRAG's internal rate limiting.
    """
    import time
    from datetime import datetime
    
    print(f"Initializing knowledge bases for concurrent testing...")
    kb_dict = await initialize_all_authors()
    print(f"Knowledge bases initialized successfully!")
    
    # Define test queries
    test_queries = [
        "What are the latest findings on Alzheimer's disease?",
        "How does amyloid beta affect brain function?", 
        "What is the role of tau protein in neurodegeneration?",
        "What are the risk factors for Alzheimer's disease?",
        "How effective are current Alzheimer's treatments?",
        "What is the relationship between sleep and Alzheimer's?",
        "How does aging affect brain health?",
        "What are biomarkers for Alzheimer's disease?",
        "How does genetics influence Alzheimer's risk?",
        "What lifestyle factors prevent cognitive decline?"
    ]
    
    # Create concurrent queries
    tasks = []
    start_time = time.time()
    
    for i in range(num_queries):
        query = test_queries[i % len(test_queries)]
        task_name = f"Query-{i+1}"
        
        # Create a task with timing
        async def run_single_query(query_text, task_id):
            query_start = time.time()
            try:
                # Use display name for testing
                available_scientists = list(kb_dict.keys())
                test_scientist = available_scientists[0] if available_scientists else "Neuroscience Expert"
                
                result = await scientist_rag_tool_wrapper(
                    test_scientist,
                    f"{query_text} (Task {task_id})"
                )
                query_end = time.time()
                duration = query_end - query_start
                
                success = "Retrieved from" in result and "Error:" not in result
                result_preview = (result[:100] + "...") if success else result
                
                print(f"✅ {task_id} completed in {duration:.2f}s - Success: {success}")
                print(f"   Preview: {result_preview}")
                
                return {
                    "task_id": task_id,
                    "duration": duration,
                    "success": success,
                    "result": result
                }
            except Exception as e:
                query_end = time.time()
                duration = query_end - query_start
                print(f"❌ {task_id} failed in {duration:.2f}s - Error: {str(e)}")
                return {
                    "task_id": task_id,
                    "duration": duration,
                    "success": False,
                    "error": str(e)
                }
        
        tasks.append(run_single_query(query, task_name))
    
    print(f"\n🚀 Starting {num_queries} concurrent queries...")
    print("=" * 60)
    
    # Run all queries concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_queries = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
    failed_queries = num_queries - successful_queries
    
    if results:
        durations = [r["duration"] for r in results if isinstance(r, dict) and "duration" in r]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
    else:
        avg_duration = min_duration = max_duration = 0
    
    print("=" * 60)
    print(f"📊 CONCURRENT TEST RESULTS:")
    print(f"   Total Queries: {num_queries}")
    print(f"   Successful: {successful_queries}")
    print(f"   Failed: {failed_queries}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Query Time: {avg_duration:.2f}s")
    print(f"   Fastest Query: {min_duration:.2f}s")
    print(f"   Slowest Query: {max_duration:.2f}s")
    print(f"   Queries per Second: {num_queries/total_time:.2f}")
    print("=" * 60)
    
    if failed_queries > 0:
        print(f"⚠️  {failed_queries} queries failed. Check the errors above.")
    else:
        print("🎉 All queries completed successfully!")
    
    return results


def print_usage():
    """Print usage instructions"""
    print("\n=== Scientist RAG Server ===")
    print("\nUsage:")
    print("  python scientist_tool.py [options]")
    print("\nOptions:")
    print("  --port PORT             Port to run server on (default: 8000)")
    print("  --test                  Run a simple test query")
    print("  --test-concurrent N     Run N concurrent test queries")
    print("  --help                  Show this help message")
    print("\nExamples:")
    print("  python scientist_tool.py                    # Start FastAPI server on port 8000")
    print("  python scientist_tool.py --port 8080        # Start on port 8080")
    print("  python scientist_tool.py --test             # Run single test query")
    print("  python scientist_tool.py --test-concurrent 5 # Run 5 concurrent queries")
    print("  python scientist_tool.py --test-concurrent 10 # Test high concurrency")
    print("\nAPI endpoints:")
    print("  POST /query     - Query knowledge base")
    print("  GET  /health    - Health check")
    print("  GET  /authors   - List available authors")
    print("  GET  /docs      - Interactive API documentation")
    print()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scientist RAG Server', add_help=False)
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run server on (default: 8000)')
    parser.add_argument('--test', action='store_true',
                       help='Run a simple test query instead of starting server')
    parser.add_argument('--test-concurrent', type=int, metavar='N',
                       help='Run N concurrent test queries to test performance')
    parser.add_argument('--help', action='store_true',
                       help='Show help message')
    
    args = parser.parse_args()
    
    if args.help:
        print_usage()
    elif args.test:
        # Run test mode
        print("Running single test mode...")
        kb_dict_test = asyncio.run(initialize_scientist_kb())
        
        # Use display name for testing
        available_scientists = list(kb_dict_test.keys())
        test_scientist = available_scientists[0] if available_scientists else "Neuroscience Expert"
        
        result = asyncio.run(scientist_rag_tool_wrapper(
            test_scientist, 
            "What are the latest findings on Alzheimer's disease?"
        ))
        print("Test result:", result)
    elif args.test_concurrent:
        # Run concurrent test mode
        print(f"Running concurrent test mode with {args.test_concurrent} queries...")
        asyncio.run(test_concurrent_queries(args.test_concurrent))
    else:
        print("Starting FastAPI server...")
        run_server(port=args.port)


