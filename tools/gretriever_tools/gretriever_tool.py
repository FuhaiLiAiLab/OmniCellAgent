from openai import OpenAI

import sys
import os
import torch
from torch_geometric.nn import GAT, GRetriever
from torch_geometric.nn.nlp import LLM
from torch_geometric.data import Data, Batch


from dotenv import load_dotenv
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

import yaml

from neo4j import Driver, GraphDatabase
import numpy as np  
import pandas as pd
from pandas import DataFrame

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path

def get_nodes_by_vector_search(query_embedding: np.ndarray, k_nodes: int, driver: Driver) -> list[int]:
    res = driver.execute_query("""
    CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
    RETURN node.nodeId AS nodeId
    """,
                               parameters_={
                                   "index": "text_embeddings",
                                   "k": k_nodes,
                                   "query_embedding": query_embedding})
    return [rec.data()['nodeId'] for rec in res.records]

def cypher_retrieval(node_ids: list[int], driver: Driver):
    res = driver.execute_query("""
                    UNWIND $nodeIds AS nodeId
                    MATCH (m {nodeId:nodeId})-[r]->(n)
                    RETURN
                    m.nodeId as sourceNodeId, n.nodeId as targetNodeId, type(r) as relationshipType,
                    labels(m)[0] as sourceNodeType, labels(n)[0] as targetNodeType
                """,
                               parameters_={'nodeIds': node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_nodes(node_ids: list[int], driver: Driver) -> DataFrame:
    res = driver.execute_query("""
    UNWIND $nodeIds AS nodeId
    MATCH(node:_Entity_ {nodeId:nodeId})
    RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
    """,
                               parameters_={"nodeIds": node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_edges(node_pairs: list[tuple[int, int]], driver: Driver) -> DataFrame:
    res = driver.execute_query("""
    UNWIND $node_pairs AS pair
    MATCH(src:_Entity_ {nodeId:pair[0]})-[e]->(tgt:_Entity_ {nodeId:pair[1]})
    RETURN src.nodeId AS src, type(e) AS edge_attr, tgt.nodeId AS dst
    """,
                               parameters_={"node_pairs": node_pairs})
    return pd.DataFrame([rec.data() for rec in res.records])

def textualize_graph(textual_nodes_df, textual_edges_df):
    textual_nodes_df.description.fillna("")
    textual_nodes_df['node_attr'] = textual_nodes_df.apply(
        lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
    textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
    nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)
    edges_desc = textual_edges_df.to_csv(index=False)
    return nodes_desc + '\n' + edges_desc

def assign_node_prizes(nodes_df, topn_nodes):
    nodes = nodes_df['nodeId'].tolist()
    node_prizes = {node: len(topn_nodes) - rank for rank, node in enumerate(topn_nodes)}
    node_prizes = [4 / len(topn_nodes) * node_prizes.get(node, 0) for node in nodes]
    nodes_df['nodePrize'] = node_prizes

def assign_edge_costs(relationships_df, topn_edges=None):
    edge_costs = .5 - np.zeros(len(relationships_df))
    relationships_df['edgeCost'] = edge_costs # No edge prizes for now (recall drops 3pts, f1 is about the same)

def convert_pcst_output(pcst_output):
    pcst_src = pcst_output['nodeId'].values
    pcst_tgt = pcst_output['parentId'].values
    pcst_nodes = np.unique(np.concatenate((pcst_src, pcst_tgt)))
    pcst_edges = np.stack((pcst_src, pcst_tgt), axis=1)
    return pcst_nodes, pcst_edges


def load_env_and_create_clients(env_path="configs/db.env"):
    # Use absolute path construction to handle relative paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(env_path):
        env_path = os.path.join(script_dir, "..", env_path)
    
    load_dotenv(env_path, override=True)
    neo4j_config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    openai_client = OpenAI()
    return neo4j_config, openai_client


def get_query_embedding(query: str, client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def load_configs():
    with open(os.path.join(os.path.dirname(__file__), "../configs/retrieval_config_v0.yaml"), "r") as f:
        retrieval_config = yaml.safe_load(f)
    with open(os.path.join(os.path.dirname(__file__), "../configs/algo_config_v0.yaml"), "r") as f:
        algo_config = yaml.safe_load(f)
    return retrieval_config, algo_config

def build_query_graph(query: str, query_emb: np.ndarray, neo4j_config, retrieval_config, algo_config) -> Data:
    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
        topk_node_ids = get_nodes_by_vector_search(query_emb, 25 * retrieval_config['k_nodes'], driver)[:retrieval_config['k_nodes']]
        relationships_df = cypher_retrieval(topk_node_ids, driver)
        node_ids = np.unique(np.concatenate((relationships_df['sourceNodeId'], relationships_df['targetNodeId'])))
        nodes_df = pd.DataFrame({'nodeId': node_ids})

        topn_nodes = get_nodes_by_vector_search(query_emb, algo_config["prized_nodes"], driver)
        top25_nodes = get_nodes_by_vector_search(query_emb, algo_config["topk_nodes"], driver)

        assign_node_prizes(nodes_df, topn_nodes)
        assign_edge_costs(relationships_df)

    gds = GraphDataScience(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"]))
    with gds.graph.construct(graph_name="pcst-graph", nodes=nodes_df, relationships=relationships_df.drop(['sourceNodeType','targetNodeType'], axis=1), undirected_relationship_types=["*"]) as G:
        pcst_output = gds.prizeSteinerTree.stream(G, prizeProperty="nodePrize", relationshipWeightProperty="edgeCost")

    pcst_nodes, pcst_edges = convert_pcst_output(pcst_output)
    pcst_nodes = np.unique(np.concatenate((pcst_nodes, top25_nodes)))

    with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
        textual_nodes_df = get_textual_nodes(pcst_nodes, driver)
        textual_edges_df = get_textual_edges(pcst_edges, driver)

    textual_nodes_df['vector_similarity'] = textual_nodes_df['textEmbedding'].apply(lambda emb: np.dot(emb, query_emb))
    textual_nodes_df = textual_nodes_df.sort_values(by='vector_similarity', ascending=False)
    textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)

    node_embedding = torch.tensor(textual_nodes_df['textEmbedding'].tolist())
    consecutive_map = {id_: i for i, id_ in enumerate(textual_nodes_df['node_id'])}
    edge_index = torch.tensor([
        [consecutive_map[src], consecutive_map[dst]]
        for src, dst in pcst_edges
        if src in consecutive_map and dst in consecutive_map
    ], dtype=torch.int64).T

    desc = textualize_graph(textual_nodes_df, textual_edges_df)

    return Data(
        x=node_embedding,
        edge_index=edge_index,
        edge_attr=None,
        question=f"Question: {query}\nAnswer: ",
        label="",
        desc=desc
    )

class GRetrieverWrapper:
    def __init__(self, hidden_channels=1536, num_gnn_layers=4, llm='meta-llama/Llama-3.1-8B-Instruct', weight_path=''):
        self.llm = LLM(model_name=llm, num_params=8)
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.weight_path = weight_path

        self.gnn = GAT(
            in_channels=1536,
            hidden_channels=hidden_channels,
            out_channels=1536,
            num_layers=num_gnn_layers,
            heads=4,
        )

        self.model = GRetriever(llm=self.llm, gnn=self.gnn)

        if self.weight_path:
            self.load_weights()

    def load_weights(self):
        state_dict = self.model.state_dict()
        loaded_state_dict = torch.load(self.weight_path)
        state_dict.update(loaded_state_dict)
        self.model.load_state_dict(state_dict, strict=False)

    def get_model(self):
        return self.model

    def inference_step(self, batch):
        self.model.eval()
        return self.model.inference(
            batch.question,
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.edge_attr,
            batch.desc
        )

def retriever_response(query: str):
    """
    
    Use the G-Retriever to get relevant nodes from a knowledge-graph.
    Args:
        query (str): The query to search for.
        Returns:
        str: The retrieved documents in the following format:
        {
            "question": [
                "Question: <your query>\nAnswer: "
            ],
            "response": [
                "<retrieved documents>"
            ]
        }
        """
    g_retriever_model_path = get_path('models.gretriever', absolute=True)
    neo4j_config, openai_client = load_env_and_create_clients()
    retrieval_config, algo_config = load_configs()

    query_emb = get_query_embedding(query, openai_client)
    data = build_query_graph(query, query_emb, neo4j_config, retrieval_config, algo_config)
    batch = Batch.from_data_list([data])

    retriever = GRetrieverWrapper(
        weight_path=g_retriever_model_path
    )
    response = retriever.inference_step(batch)

    return {
        "question": batch.question,
        "response": response,
        # "desciption": batch.desc
    }

if __name__ == "__main__":
    query = "which pharmaceutical compound, currently under research for addressing both HIV Infection and Alzheimer's Disease, acts upon genes or proteins that associate with death effector domain interactions?"
    model_path = '/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/neo4j-gnn-llm-example/stark_qa_v0_0/models/0_0_0_gnn-llm-llama3.1-8b_best_val_loss_ckpt.pt'
    result = retriever_response(query)
    print(result)