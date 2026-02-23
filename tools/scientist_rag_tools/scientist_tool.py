"""
author_kb_tool.py - Build and query author-specific scientific knowledge bases using RAG.
FastAPI server implementation for high concurrent loads.
"""

import sys
import os
import json
import argparse
import yaml

# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio
from dotenv import load_dotenv
import re
import pymupdf          # PDF text extraction
import aiofiles          # async file I/O

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

    def __init__(self, cache_key: str, pubmed_name: str = None, top_k: int = DEFAULT_TOP_K):
        self.author_name = cache_key                  # Anonymized name for logging/display
        self.pubmed_name = pubmed_name or cache_key    # Real name, ONLY for PubMed queries
        self.author_dir_name = self._sanitize_name(cache_key)
        self.top_k = top_k

        # Directory setup — uses anonymized cache_key, NOT real author name
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
        return [f"{self.pubmed_name}[Author]"]

    # Minimum number of downloaded papers we consider acceptable.
    MIN_PAPERS = 50

    async def fetch_and_cache_papers(self) -> str:
        query_terms = self._pubmed_author_query()
        output_path = os.path.join(self.jsonl_cache_dir, f"{self.author_dir_name}.jsonl")

        # Step 1: Query PubMed metadata.
        # Re-fetch if JSONL is missing *or* is stale (fewer entries than top_k).
        need_refetch = True
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_lines = f.readlines()
            if len(existing_lines) >= self.top_k:
                print(f"📄 PubMed metadata already cached for {self.author_name} "
                      f"({len(existing_lines)} entries)")
                need_refetch = False
            else:
                print(f"🔄 JSONL has only {len(existing_lines)} entries "
                      f"(need {self.top_k}) — re-querying PubMed for {self.author_name}")

        if need_refetch:
            print(f"🔍 Querying PubMed for papers by {self.author_name} "
                  f"(max_results={self.top_k})...")
            get_and_dump_pubmed_papers(query_terms, output_filepath=output_path)

            with open(output_path, 'r') as f:
                lines = f.readlines()
            total_found = len(lines)
            lines = lines[:self.top_k]
            with open(output_path, 'w') as f:
                f.writelines(lines)
            print(f"📋 PubMed returned {total_found} papers, kept top {len(lines)}")

        # Step 2: Download PDFs/XMLs — wrap each paper in try/except so a single
        #         network failure doesn't kill the entire pipeline.
        #         save_pdf_from_dump already skips files that exist on disk.
        existing_papers = self._count_doi_papers()
        try:
            save_pdf_from_dump(output_path, pdf_path=self.doi_cache_dir, key_to_save='doi')
        except Exception as e:
            print(f"⚠️  PDF download partially failed for {self.author_name}: {e}")

        final_count = self._count_doi_papers()
        new_downloads = final_count - existing_papers
        print(f"📥 {final_count} papers in doi_cache for {self.author_name} "
              f"({new_downloads} new downloads)")

        if final_count < self.MIN_PAPERS:
            print(f"⚠️  Only {final_count} papers downloaded (target ≥{self.MIN_PAPERS}). "
                  f"Some papers may be paywalled or unavailable.")
        return self.doi_cache_dir

    def _count_doi_papers(self) -> int:
        """Count PDF/XML files in doi_cache."""
        if not os.path.exists(self.doi_cache_dir):
            return 0
        return sum(1 for f in os.listdir(self.doi_cache_dir)
                   if f.endswith('.pdf') or f.endswith('.xml'))

    # ---- text extraction helpers ----

    @staticmethod
    async def _extract_text_from_pdf(file_path: str) -> str:
        """Extract raw text from a PDF file."""
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: chr(12).join(
                [page.get_text() for page in pymupdf.open(file_path)]
            ),
        )
        return text.strip()

    @staticmethod
    async def _extract_text_from_xml(file_path: str) -> str:
        """Extract raw text from a PMC / Elsevier XML file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        # Strip XML tags for a plain-text approximation
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def _extract_texts_from_dir(self, paper_dir: str) -> List[str]:
        """Extract text from every PDF/XML in *paper_dir*."""
        texts: List[str] = []
        files = sorted(os.listdir(paper_dir))
        for fname in files:
            fpath = os.path.join(paper_dir, fname)
            try:
                if fname.endswith('.pdf'):
                    t = await self._extract_text_from_pdf(fpath)
                elif fname.endswith('.xml'):
                    t = await self._extract_text_from_xml(fpath)
                else:
                    continue
                if t and len(t) > 100:          # skip near-empty files
                    texts.append(t)
                    print(f"  📄 Extracted {len(t):,} chars from {fname}")
                else:
                    print(f"  ⚠️  Skipped {fname} (too short: {len(t)} chars)")
            except Exception as e:
                print(f"  ❌ Failed to extract {fname}: {e}")
        return texts

    # ---- RAG insertion ----

    async def insert_texts_to_rag(self, texts: List[str]):
        """Insert a list of plain-text documents into the HyperRAG store."""
        if not self.rag:
            raise ValueError("RAG not initialized.")

        from tqdm import tqdm
        batch_size = 5
        results = []

        with tqdm(total=len(texts), desc=f"Inserting papers for {self.author_name}") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = [self.rag.ainsert(t) for t in texts[i:i + batch_size]]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                for j, res in enumerate(batch_results):
                    if isinstance(res, Exception):
                        print(f"  [!] Insertion failed for doc {i + j}: {res}")
                results.extend(batch_results)
                pbar.update(len(batch))
                await asyncio.sleep(0.1)
        return results

    # ---- main build pipeline ----

    async def build_knowledge_base(self, paper_dir: str = None):
        await self.initialize_rag()

        # 1. Already indexed? → skip
        if os.path.exists(self.working_dir):
            content_files = [f for f in os.listdir(self.working_dir)
                            if f.endswith('.json') or f.endswith('.hgdb')]
            if content_files:
                print(f"📂 KB already built for {self.author_name}, loading from cache")
                return

        # 2. Papers already downloaded? Re-fetch if below MIN_PAPERS.
        doi_files = [f for f in os.listdir(self.doi_cache_dir)
                     if f.endswith('.pdf') or f.endswith('.xml')] \
                    if os.path.exists(self.doi_cache_dir) else []

        if len(doi_files) >= self.MIN_PAPERS:
            print(f"📥 Found {len(doi_files)} papers in doi_cache for {self.author_name}, "
                  f"skipping download (≥{self.MIN_PAPERS})")
            paper_dir = self.doi_cache_dir
        else:
            if doi_files:
                print(f"📥 Found only {len(doi_files)} papers in doi_cache for {self.author_name} "
                      f"(need ≥{self.MIN_PAPERS}), re-fetching...")
            paper_dir = await self.fetch_and_cache_papers()

        # 3. Extract text directly from PDFs / XMLs
        print(f"📖 Extracting text from papers in {paper_dir}")
        texts = await self._extract_texts_from_dir(paper_dir)
        if not texts:
            print(f"⚠️  No usable paper text for {self.author_name} — KB will be empty")
            return

        # 4. Insert into HyperRAG
        await self.insert_texts_to_rag(texts)
        print(f"✅ KB built for {self.author_name}: {len(texts)} papers indexed")

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

def initialize_author_mapping() -> Dict[str, str]:
    """
    Create privacy-protected mapping from real author names to anonymized expert aliases.
    Uses EXPERT_SCIENTISTS config. Real names are never exposed externally.
    """
    global _author_mapping, _reverse_mapping
    
    if not _author_mapping:
        for scientist in EXPERT_SCIENTISTS:
            real_name = scientist["real_name"]
            alias = scientist["alias"]
            if real_name != "CHANGE_ME":
                _author_mapping[real_name] = alias
                _reverse_mapping[alias] = real_name
            else:
                # Unconfigured experts: alias maps to itself
                _reverse_mapping[alias] = alias
            
    
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
        print("Initializing expert scientist knowledge bases...")
        _global_kb_dict = await initialize_all_authors()
        initialize_author_mapping()
        
    return _global_kb_dict

async def scientist_rag_tool_wrapper(expert_alias: str, question: str) -> str:
    """
    Query a domain expert's scientific knowledge base.
    
    Parameters:
    - expert_alias (str): Expert alias to query (e.g., "GenomicsExpert", "NeuroscienceExpert")
    - question (str): The scientific question to ask
    
    Returns:
    - str: The response from the expert's knowledge base or error message
    
    Available experts (when configured):
    - GenomicsExpert
    - NeuroscienceExpert
    - LongevityBiostatsExpert
    - BioinformaticsExpert
    """
    try:
        # Initialize KB if not already done
        kb_dict = await initialize_scientist_kb()
        
        if expert_alias not in kb_dict:
            available = list(kb_dict.keys())
            return f"Error: Expert '{expert_alias}' not found. Available experts: {', '.join(available)}"
        
        result = await scientist_rag_retrieval_tool(expert_alias, question, kb_dict, display_name=expert_alias)
        
        if "error" in result:
            return f"Error: {result['error']}"
        else:
            return f"Retrieved from {expert_alias}'s knowledge base:\n\n{result['result']}"
            
    except Exception as e:
        error_msg = f"Failed to query {expert_alias} knowledge base: {str(e)}"
        print(error_msg)
        return error_msg


async def query_expert_kb(expert_alias: str, question: str) -> str:
    """
    Query a specific expert's knowledge base by their alias.
    Convenience wrapper for use by ScientistsAgent sub-experts.
    
    Args:
        expert_alias: One of GenomicsExpert, NeuroscienceExpert,
                      LongevityBiostatsExpert, BioinformaticsExpert
        question: Scientific query (3-5 words work best)
    
    Returns:
        Retrieved knowledge or error message
    """
    return await scientist_rag_tool_wrapper(expert_alias, question)


def get_available_experts() -> List[Dict[str, str]]:
    """Get list of configured expert aliases and their domains."""
    return [
        {"alias": e["alias"], "domain": e["domain"], "description": e["description"]}
        for e in EXPERT_SCIENTISTS
    ]


def get_built_experts() -> List[Dict[str, str]]:
    """Return only experts whose RAG knowledge bases are actually built on disk.

    Scans each expert's ragstore directory for .json / .hgdb content files.
    Papers in doi_cache but without a built ragstore are NOT included.

    Returns:
        List of dicts with keys: alias, domain, description, cache_key
    """
    built = []
    for expert in EXPERT_SCIENTISTS:
        cache_key = expert["cache_key"]
        ragstore_dir = os.path.join(BASE_AUTHOR_DIR, cache_key, "ragstore")
        if not os.path.isdir(ragstore_dir):
            continue
        content_files = [
            f for f in os.listdir(ragstore_dir)
            if f.endswith(".json") or f.endswith(".hgdb")
        ]
        if content_files:
            built.append({
                "alias": expert["alias"],
                "domain": expert["domain"],
                "description": expert["description"],
                "cache_key": cache_key,
            })
    return built


def get_rag_ready_experts() -> List[Dict[str, str]]:
    """Return experts that can provide RAG — either already built or with papers to build from.

    An expert is 'RAG-ready' if:
    1. Its ragstore is already built (has .json/.hgdb files), OR
    2. It has downloaded papers in doi_cache (KB will be built lazily on first query)

    Experts whose real_name is 'CHANGE_ME' and have no cached data are excluded.

    Returns:
        List of dicts with keys: alias, domain, description, cache_key, status
        status is 'built' (ragstore exists) or 'pending' (papers only, builds on first query)
    """
    ready = []
    for expert in EXPERT_SCIENTISTS:
        cache_key = expert["cache_key"]
        ragstore_dir = os.path.join(BASE_AUTHOR_DIR, cache_key, "ragstore")
        doi_cache_dir = os.path.join(BASE_AUTHOR_DIR, cache_key, "doi_cache")

        # Check ragstore (already built)
        has_ragstore = False
        if os.path.isdir(ragstore_dir):
            content_files = [f for f in os.listdir(ragstore_dir)
                            if f.endswith(".json") or f.endswith(".hgdb")]
            has_ragstore = bool(content_files)

        # Check doi_cache (papers downloaded, can build on demand)
        has_papers = False
        if os.path.isdir(doi_cache_dir):
            paper_files = [f for f in os.listdir(doi_cache_dir)
                          if f.endswith(".pdf") or f.endswith(".xml")]
            has_papers = len(paper_files) > 0

        if has_ragstore or has_papers:
            ready.append({
                "alias": expert["alias"],
                "domain": expert["domain"],
                "description": expert["description"],
                "cache_key": cache_key,
                "status": "built" if has_ragstore else "pending",
            })
        elif expert["real_name"] != "CHANGE_ME":
            # Configured but no papers yet — still include as pending
            ready.append({
                "alias": expert["alias"],
                "domain": expert["domain"],
                "description": expert["description"],
                "cache_key": cache_key,
                "status": "pending",
            })
    return ready


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
        """Initialize expert knowledge bases on startup"""
        global kb_dict
        print("Initializing expert knowledge bases for FastAPI server...")
        try:
            kb_dict = await initialize_all_authors()
            initialize_author_mapping()
            
            print(f"Successfully initialized {len(kb_dict)} expert knowledge bases")
            print("Available experts:")
            for alias in kb_dict:
                print(f"  - {alias}")
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


# ==============================================================================
# EXPERT SCIENTIST CONFIGURATION  (loaded from configs/experts.yaml)
# ==============================================================================
# The YAML file is gitignored — real author names never enter the repo.
# See configs/experts.yaml.example for the template.
#
# Fields per expert:
#   real_name   — PubMed author string (PRIVATE, used only for paper fetching)
#   alias       — Anonymized name exposed to agents
#   cache_key   — Folder under cache/author_kb/
#   domain      — Short domain tag
#   description — Domain description
#   top_k       — Max papers to fetch
# ==============================================================================

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_EXPERTS_YAML = os.path.join(_PROJECT_ROOT, 'configs', 'experts.yaml')
_EXPERTS_YAML_EXAMPLE = os.path.join(_PROJECT_ROOT, 'configs', 'experts.yaml.example')


def _load_expert_config() -> List[Dict[str, Any]]:
    """Load expert scientist config from configs/experts.yaml."""
    yaml_path = _EXPERTS_YAML
    if not os.path.exists(yaml_path):
        # Fall back to the example template so the system still boots
        yaml_path = _EXPERTS_YAML_EXAMPLE
        print(f"⚠️  configs/experts.yaml not found — using example template. "
              f"Copy configs/experts.yaml.example → configs/experts.yaml and fill in real names.")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    experts = data.get('experts', [])
    if not experts:
        print("⚠️  No experts defined in config file.")
    return experts


EXPERT_SCIENTISTS: List[Dict[str, Any]] = _load_expert_config()

# Convenience lookups
_EXPERT_BY_ALIAS = {e["alias"]: e for e in EXPERT_SCIENTISTS}
_EXPERT_BY_DOMAIN = {e["domain"]: e for e in EXPERT_SCIENTISTS}


async def initialize_all_authors():
    """Build / load KBs for all configured expert scientists."""
    kb_dict = {}
    for scientist in EXPERT_SCIENTISTS:
        real_name = scientist["real_name"]
        alias = scientist["alias"]
        cache_key = scientist["cache_key"]

        # Check if KB is already cached on disk (skip PubMed fetch if so)
        working_dir = os.path.join(BASE_AUTHOR_DIR, cache_key, "ragstore")
        already_cached = os.path.exists(working_dir) and os.listdir(working_dir)

        if real_name == "CHANGE_ME" and not already_cached:
            print(f"⚠️  Skipping {alias}: real_name not configured and no cached KB found")
            continue

        kb = AuthorKnowledgeBase(
            cache_key=cache_key,
            pubmed_name=real_name if real_name != "CHANGE_ME" else None,
            top_k=scientist["top_k"],
        )
        await kb.build_knowledge_base(paper_dir=kb.doi_cache_dir)
        kb_dict[alias] = kb
        print(f"✅ {alias} KB ready (cache: {cache_key})")

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
                available_experts = list(kb_dict.keys())
                test_expert = available_experts[0] if available_experts else "NeuroscienceExpert"
                
                result = await scientist_rag_tool_wrapper(
                    test_expert,
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
        available_experts = list(kb_dict_test.keys())
        test_expert = available_experts[0] if available_experts else "NeuroscienceExpert"
        print(f"Testing with expert: {test_expert}")
        
        result = asyncio.run(scientist_rag_tool_wrapper(
            test_expert, 
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


