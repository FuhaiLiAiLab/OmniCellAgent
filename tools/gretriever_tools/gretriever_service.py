#!/usr/bin/env python3
"""
G-Retriever Microservice

A FastAPI-based microservice for serving G-Retriever model for biomedical knowledge graph queries.
The model is loaded once at startup and serves multiple queries efficiently.

Usage:
    python gretriever_service.py

API Endpoints:
    POST /query - Submit a query for G-Retriever inference
    GET /health - Health check endpoint
    GET /status - Service status and model information
"""

import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

# Optimize CPU threading BEFORE importing torch
# i9-13900KF: 8 P-cores (16 threads) + 16 E-cores = 32 total
NUM_THREADS = 24  # Balance between parallelism and overhead
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)

import torch
torch.set_num_threads(NUM_THREADS)

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
from openai import OpenAI

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path

# Import the existing G-Retriever components
from torch_geometric.nn import GAT, GRetriever
from torch_geometric.nn.nlp import LLM
from torch_geometric.data import Data, Batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"CPU optimization: Using {NUM_THREADS} threads (P-cores only)")

# Global variables for model and configurations
g_retriever_wrapper = None
neo4j_config = None
openai_client = None
retrieval_config = None
algo_config = None

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The biomedical query to process", min_length=1)
    max_nodes: Optional[int] = Field(default=None, description="Maximum number of nodes to retrieve (overrides config)")
    include_description: Optional[bool] = Field(default=False, description="Include graph description in response")

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    response: str
    question_formatted: str
    description: Optional[str] = None
    processing_info: Dict[str, Any]
    graph_data: Optional[List[Dict[str, Any]]] = None  # Add graph data field

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str

class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    service: str
    model_loaded: bool
    neo4j_connected: bool
    openai_configured: bool
    model_info: Dict[str, Any]

class GRetrieverWrapper:
    """Wrapper class for G-Retriever model with lazy loading"""
    
    def __init__(self, hidden_channels=1536, num_gnn_layers=4, llm='meta-llama/Llama-3.1-8B-Instruct', weight_path='', device='cpu'):
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.llm_model_name = llm
        self.weight_path = weight_path
        self.model = None
        self.is_loaded = False
        self.device = device  # Force CPU by default
        
    def load_model(self):
        """Load the model components"""
        if self.is_loaded:
            return
            
        logger.info(f"Loading G-Retriever model on device: {self.device}")
        
        # Initialize LLM - force CPU
        self.llm = LLM(model_name=self.llm_model_name, num_params=8, dtype=torch.float32)
        
        # Initialize GNN
        self.gnn = GAT(
            in_channels=1536,
            hidden_channels=self.hidden_channels,
            out_channels=1536,
            num_layers=self.num_gnn_layers,
            heads=4,
        )
        
        # Initialize G-Retriever
        self.model = GRetriever(llm=self.llm, gnn=self.gnn)
        
        # Move model to specified device (CPU)
        self.model = self.model.to(self.device)
        
        # Load weights if provided
        if self.weight_path and os.path.exists(self.weight_path):
            logger.info(f"Loading weights from {self.weight_path}")
            state_dict = self.model.state_dict()
            loaded_state_dict = torch.load(self.weight_path, map_location=self.device)
            state_dict.update(loaded_state_dict)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"Weight path not found: {self.weight_path}")
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"torch.compile() failed, using eager mode: {e}")
        
        self.is_loaded = True
        logger.info("G-Retriever model loaded successfully")
    
    def inference_step(self, batch):
        """Perform inference on a batch"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure batch tensors are on the correct device
        batch = batch.to(self.device)
        
        self.model.eval()
        with torch.no_grad(), torch.inference_mode():
            return self.model.inference(
                batch.question,
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.edge_attr,
                batch.desc
            )

# Import functions from the original gretriever_tool.py
def load_env_and_create_clients(env_path=None):
    """Load environment variables and create clients"""
    if env_path is None:
        # Try to get path from config, fallback to default
        try:
            env_path = get_path('config_files.db_env', absolute=True)
        except:
            # If config fails, try external path
            try:
                env_path = get_path('config_files.db_env_external', absolute=True)
            except:
                # Final fallback to default location
                env_path = "configs/db.env"
    
    # Make sure path is absolute
    if not os.path.isabs(env_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, "..", "..", env_path)
    
    load_dotenv(env_path, override=True)
    neo4j_config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    openai_client = OpenAI()
    return neo4j_config, openai_client

def load_configs():
    """Load retrieval and algorithm configurations"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(script_dir, "../../configs/retrieval_config_v0.yaml"), "r") as f:
        retrieval_config = yaml.safe_load(f)
    with open(os.path.join(script_dir, "../../configs/algo_config_v0.yaml"), "r") as f:
        algo_config = yaml.safe_load(f)
    return retrieval_config, algo_config

def get_query_embedding(query: str, client: OpenAI) -> np.ndarray:
    """Get OpenAI embedding for the query"""
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def get_nodes_by_vector_search(query_embedding: np.ndarray, k_nodes: int, driver) -> list[int]:
    """Get top-k nodes by vector similarity"""
    res = driver.execute_query("""
    CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
    RETURN node.nodeId AS nodeId
    """,
    parameters_={
        "index": "text_embeddings",
        "k": k_nodes,
        "query_embedding": query_embedding
    })
    return [rec.data()['nodeId'] for rec in res.records]

def cypher_retrieval(node_ids: list[int], driver):
    """Retrieve relationships for given node IDs"""
    res = driver.execute_query("""
    UNWIND $nodeIds AS nodeId
    MATCH (m {nodeId:nodeId})-[r]->(n)
    RETURN
    m.nodeId as sourceNodeId, n.nodeId as targetNodeId, type(r) as relationshipType,
    labels(m)[0] as sourceNodeType, labels(n)[0] as targetNodeType
    """,
    parameters_={'nodeIds': node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_nodes(node_ids: list[int], driver) -> pd.DataFrame:
    """Get textual information for nodes"""
    res = driver.execute_query("""
    UNWIND $nodeIds AS nodeId
    MATCH(node:_Entity_ {nodeId:nodeId})
    RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
    """,
    parameters_={"nodeIds": node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_edges(node_pairs: list[tuple[int, int]], driver) -> pd.DataFrame:
    """Get textual information for edges"""
    res = driver.execute_query("""
    UNWIND $node_pairs AS pair
    MATCH(src:_Entity_ {nodeId:pair[0]})-[e]->(tgt:_Entity_ {nodeId:pair[1]})
    RETURN src.nodeId AS src, type(e) AS edge_attr, tgt.nodeId AS dst
    """,
    parameters_={"node_pairs": node_pairs})
    return pd.DataFrame([rec.data() for rec in res.records])

def parse_node_attr(node_attr_str: str) -> Dict[str, Any]:
    """Parse node attribute string to extract name and description, filtering out keys containing '_id'"""
    try:
        if node_attr_str.startswith("name: "):
            # Find the description part
            desc_start = node_attr_str.find(", description: ")
            if desc_start != -1:
                name = node_attr_str[6:desc_start]  # Skip "name: "
                desc_str = node_attr_str[desc_start + 14:]  # Skip ", description: "
                
                # Try to parse the description as JSON
                try:
                    import ast
                    description = ast.literal_eval(desc_str)
                    
                    # Filter out keys containing '_id' if description is a dictionary
                    if isinstance(description, dict):
                        filtered_description = {
                            key: value for key, value in description.items() 
                            if "_id" not in key.lower()
                        }
                        description = filtered_description
                        
                except:
                    description = desc_str
                
                return {
                    "name": name,
                    "description": description
                }
        
        # Fallback: return the original string
        return {"name": "", "description": node_attr_str}
    except Exception as e:
        logger.warning(f"Error parsing node attribute: {e}")
        return {"name": "", "description": node_attr_str}

def create_graph_json(textual_nodes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create JSON structure for graph data"""
    graph_data = []
    
    for _, row in textual_nodes_df.iterrows():
        node_attr = parse_node_attr(row['node_attr'])
        
        graph_data.append({
            "node_id": int(row['node_id']),
            "vector_similarity": float(row['vector_similarity']),
            "node_attr": node_attr
        })
    
    return graph_data

def save_graph_json(graph_data: List[Dict[str, Any]], query: str, output_dir: str = "logs") -> str:
    """Save graph data to JSON file"""    

    filename = f"g_retriever_result.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Graph data saved to: {filepath}")
    return filepath

def textualize_graph(textual_nodes_df, textual_edges_df):
    """Convert graph data to textual description"""
    textual_nodes_df = textual_nodes_df.copy()
    textual_nodes_df.loc[:, 'description'] = textual_nodes_df['description'].fillna("")
    textual_nodes_df['node_attr'] = textual_nodes_df.apply(
        lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
    textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
    nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)
    edges_desc = textual_edges_df.to_csv(index=False)
    return nodes_desc + '\n' + edges_desc

def assign_node_prizes(nodes_df, topn_nodes):
    """Assign prizes to nodes for PCST algorithm"""
    nodes = nodes_df['nodeId'].tolist()
    node_prizes = {node: len(topn_nodes) - rank for rank, node in enumerate(topn_nodes)}
    node_prizes = [4 / len(topn_nodes) * node_prizes.get(node, 0) for node in nodes]
    nodes_df['nodePrize'] = node_prizes

def assign_edge_costs(relationships_df, topn_edges=None):
    """Assign costs to edges for PCST algorithm"""
    edge_costs = 0.5 - np.zeros(len(relationships_df))
    relationships_df['edgeCost'] = edge_costs

def convert_pcst_output(pcst_output):
    """Convert PCST output to nodes and edges"""
    pcst_src = pcst_output['nodeId'].values
    pcst_tgt = pcst_output['parentId'].values
    pcst_nodes = np.unique(np.concatenate((pcst_src, pcst_tgt)))
    pcst_edges = np.stack((pcst_src, pcst_tgt), axis=1)
    return pcst_nodes, pcst_edges

def build_query_graph(query: str, query_emb: np.ndarray, max_nodes: Optional[int] = None) -> tuple[Data, dict, pd.DataFrame]:
    """Build query graph for G-Retriever"""
    global neo4j_config, retrieval_config, algo_config
    
    processing_info = {
        "nodes_found": 0,
        "relationships_found": 0,
        "embeddings_loaded": 0,
        "pcst_nodes": 0,
        "final_nodes": 0
    }
    
    k_nodes = max_nodes if max_nodes is not None else retrieval_config['k_nodes']
    
    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
        # Get initial nodes by vector search
        topk_node_ids = get_nodes_by_vector_search(query_emb, 25 * k_nodes, driver)[:k_nodes]
        processing_info["nodes_found"] = len(topk_node_ids)
        
        if not topk_node_ids:
            logger.warning("No nodes found by vector search")
            # Return minimal graph
            empty_tensor = torch.empty((2, 0), dtype=torch.int64)
            empty_df = pd.DataFrame(columns=['node_id', 'vector_similarity', 'node_attr', 'name', 'description'])
            return Data(
                x=torch.empty(0, 1536),
                edge_index=empty_tensor,
                edge_attr=None,
                question=f"Question: {query}\nAnswer: ",
                label="",
                desc="No relevant nodes found in the knowledge graph."
            ), processing_info, empty_df
        
        # Get relationships
        relationships_df = cypher_retrieval(topk_node_ids, driver)
        processing_info["relationships_found"] = len(relationships_df)
        
        # Handle empty relationships
        if relationships_df.empty or 'sourceNodeId' not in relationships_df.columns:
            node_ids = np.array(topk_node_ids)
        else:
            node_ids = np.unique(np.concatenate((relationships_df['sourceNodeId'], relationships_df['targetNodeId'])))
        
        nodes_df = pd.DataFrame({'nodeId': node_ids})
        
        # Get additional nodes for prizes
        topn_nodes = get_nodes_by_vector_search(query_emb, algo_config["prized_nodes"], driver)
        top25_nodes = get_nodes_by_vector_search(query_emb, algo_config["topk_nodes"], driver)
        
        assign_node_prizes(nodes_df, topn_nodes)
        if not relationships_df.empty:
            assign_edge_costs(relationships_df)

    # Run PCST algorithm
    gds = GraphDataScience(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"]))
    
    if relationships_df.empty:
        pcst_nodes = node_ids
        pcst_edges = np.empty((0, 2), dtype=int)
    else:
        with gds.graph.construct(
            graph_name="pcst-graph", 
            nodes=nodes_df, 
            relationships=relationships_df.drop(['sourceNodeType','targetNodeType'], axis=1), 
            undirected_relationship_types=["*"]
        ) as G:
            pcst_output = gds.prizeSteinerTree.stream(G, prizeProperty="nodePrize", relationshipWeightProperty="edgeCost")
            pcst_nodes, pcst_edges = convert_pcst_output(pcst_output)
    
    pcst_nodes = np.unique(np.concatenate((pcst_nodes, top25_nodes)))
    processing_info["pcst_nodes"] = len(pcst_nodes)

    # Get textual information
    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
        textual_nodes_df = get_textual_nodes(pcst_nodes, driver)
        if len(pcst_edges) > 0:
            textual_edges_df = get_textual_edges(pcst_edges, driver)
        else:
            textual_edges_df = pd.DataFrame(columns=['src', 'edge_attr', 'dst'])

    processing_info["embeddings_loaded"] = len(textual_nodes_df)
    processing_info["final_nodes"] = len(textual_nodes_df)

    if textual_nodes_df.empty:
        logger.warning("No textual nodes found")
        empty_tensor = torch.empty((2, 0), dtype=torch.int64)
        empty_df = pd.DataFrame(columns=['node_id', 'vector_similarity', 'node_attr', 'name', 'description'])
        return Data(
            x=torch.empty(0, 1536),
            edge_index=empty_tensor,
            edge_attr=None,
            question=f"Question: {query}\nAnswer: ",
            label="",
            desc="No textual information found for nodes."
        ), processing_info, empty_df

    # Calculate similarity and prepare embeddings
    textual_nodes_df['vector_similarity'] = textual_nodes_df['textEmbedding'].apply(lambda emb: np.dot(emb, query_emb))
    textual_nodes_df = textual_nodes_df.sort_values(by='vector_similarity', ascending=False)
    textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
    
    # Create node_attr column for JSON export
    textual_nodes_df['node_attr'] = textual_nodes_df.apply(
        lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)

    node_embedding = torch.tensor(textual_nodes_df['textEmbedding'].tolist())
    consecutive_map = {id_: i for i, id_ in enumerate(textual_nodes_df['node_id'])}
    
    # Create edge index
    if len(pcst_edges) > 0:
        edge_index = torch.tensor([
            [consecutive_map[src], consecutive_map[dst]]
            for src, dst in pcst_edges
            if src in consecutive_map and dst in consecutive_map
        ], dtype=torch.int64).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)

    desc = textualize_graph(textual_nodes_df, textual_edges_df)

    return Data(
        x=node_embedding,
        edge_index=edge_index,
        edge_attr=None,
        question=f"Question: {query}\nAnswer: ",
        label="",
        desc=desc
    ), processing_info, textual_nodes_df

async def initialize_service():
    """Initialize the service components"""
    global g_retriever_wrapper, neo4j_config, openai_client, retrieval_config, algo_config
    
    logger.info("Initializing G-Retriever service...")
    
    # Load configurations
    neo4j_config, openai_client = load_env_and_create_clients()
    retrieval_config, algo_config = load_configs()
    
    # Initialize model wrapper - use path from config, force CPU
    g_retriever_model_path = get_path('models.gretriever', absolute=True)

    g_retriever_wrapper = GRetrieverWrapper(
        weight_path=g_retriever_model_path,
        device='cpu'  # Force CPU to avoid GPU memory issues
    )
    
    # Load model in background
    g_retriever_wrapper.load_model()
    
    logger.info("G-Retriever service initialized successfully")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    await initialize_service()
    yield
    # Shutdown
    logger.info("Shutting down G-Retriever service")

# Create FastAPI app
app = FastAPI(
    title="G-Retriever Microservice",
    description="A microservice for biomedical knowledge graph queries using G-Retriever",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Process a biomedical query using G-Retriever"""
    try:
        if not g_retriever_wrapper or not g_retriever_wrapper.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Get query embedding
        query_emb = get_query_embedding(request.query, openai_client)
        
        # Build query graph
        data, processing_info, textual_nodes_df = build_query_graph(request.query, query_emb, request.max_nodes)
        batch = Batch.from_data_list([data])
        
        # Run inference
        response = g_retriever_wrapper.inference_step(batch)
        
        # Handle response format - convert list to string if needed
        if isinstance(response, list):
            response_text = response[0] if response else "No response generated"
        else:
            response_text = str(response)
        
        # Create graph data JSON
        graph_data = []
        json_filepath = None
        
        if not textual_nodes_df.empty:
            try:
                # Ensure we have the required columns
                if 'node_attr' in textual_nodes_df.columns:
                    graph_data = create_graph_json(textual_nodes_df)
                    
                    # Save graph data to JSON file
                    if graph_data:
                        json_filepath = save_graph_json(graph_data, request.query)
                        processing_info["json_file_saved"] = json_filepath
                else:
                    logger.warning("node_attr column missing from textual_nodes_df")
                    processing_info["json_error"] = "node_attr column missing"
            except Exception as json_error:
                logger.error(f"Error creating JSON: {json_error}")
                processing_info["json_error"] = str(json_error)
        
        # Prepare response
        result = QueryResponse(
            query=request.query,
            response=response_text,
            question_formatted=batch.question[0] if batch.question else f"Question: {request.query}\nAnswer: ",
            description=batch.desc[0] if request.include_description and batch.desc else None,
            processing_info=processing_info,
            graph_data=graph_data
        )
        
        logger.info(f"Query processed successfully. Found {processing_info['final_nodes']} nodes. JSON saved to: {json_filepath}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        # Test Neo4j connection
        with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
            driver.execute_query("RETURN 1")
        
        # Test OpenAI
        openai_client.models.list()
        
        return HealthResponse(status="healthy", message="All services operational")
    except Exception as e:
        return HealthResponse(status="unhealthy", message=f"Service check failed: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def status_endpoint() -> StatusResponse:
    """Get service status information"""
    model_loaded = g_retriever_wrapper is not None and g_retriever_wrapper.is_loaded
    
    # Test Neo4j connection
    neo4j_connected = False
    try:
        with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
            driver.execute_query("RETURN 1")
        neo4j_connected = True
    except:
        pass
    
    # Test OpenAI
    openai_configured = False
    try:
        openai_client.models.list()
        openai_configured = True
    except:
        pass
    
    model_info = {}
    if g_retriever_wrapper:
        model_info = {
            "hidden_channels": g_retriever_wrapper.hidden_channels,
            "num_gnn_layers": g_retriever_wrapper.num_gnn_layers,
            "llm_model": g_retriever_wrapper.llm_model_name,
            "weight_path": g_retriever_wrapper.weight_path,
            "loaded": g_retriever_wrapper.is_loaded
        }
    
    return StatusResponse(
        service="G-Retriever Microservice",
        model_loaded=model_loaded,
        neo4j_connected=neo4j_connected,
        openai_configured=openai_configured,
        model_info=model_info
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the service
    uvicorn.run(
        "gretriever_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        workers=1  # Single worker to share model in memory
    )
