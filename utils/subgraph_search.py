from neo4j import Driver, GraphDatabase
import numpy as np  
import pandas as pd
from pandas import DataFrame

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