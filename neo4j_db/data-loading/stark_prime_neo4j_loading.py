from stark_qa import load_qa, load_skb
from tqdm import tqdm
import pandas as pd
from typing import Tuple, Union
from numpy.typing import ArrayLike
import re

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

import torch
from langchain_openai import OpenAIEmbeddings

import pdb


dataset_name = 'prime'

# Load the retrieval dataset
qa_dataset = load_qa(dataset_name)
idx_split = qa_dataset.get_idx_split()

# Load the semi-structured knowledge base
skb = load_skb(dataset_name, download_processed=True, root=None)

# create node_df
node_list = []

for i in tqdm(range(skb.num_nodes())):
  node = skb[i].dictionary
  node['nodeId'] = i
  node_list.append(skb[i].dictionary)
node_df = pd.DataFrame(node_list)

# format details
node_df.loc[node_df.details.isna(), 'details'] = ''
node_df.details = node_df.details.astype(str)


# function for formatting
def format_node_label(s):
  ss = s.replace('/', '_or_').lower().split('_')
  return ''.join(t.title() for t in ss)

[(k,format_node_label(v)) for k,v in  skb.node_type_dict.items()]


# helper functions for laoding nodes & rels

def _make_map(x):
    if type(x) == str:
        return x, x
    elif type(x) == tuple:
        return x
    else:
        raise Exception("Entry must of type string or tuple")

def _make_constraint_query(constraint_type: str, node_label, prop_name) -> str:
  const_name = f'{constraint_type.lower()}_{node_label.lower()}_{prop_name.lower()}'
  return f'CREATE CONSTRAINT {const_name} IF NOT EXISTS FOR (n:{node_label}) REQUIRE n.{prop_name} IS {constraint_type}'


def _make_set_clause(prop_names: ArrayLike, element_name='n', item_name='rec'):
    clause_list = []
    for prop_name in prop_names:
        clause_list.append(f'{element_name}.{prop_name} = {item_name}.{prop_name}')
    return 'SET ' + ', '.join(clause_list)


def _make_node_merge_query(node_key_name: str, node_label: str, cols: ArrayLike):
    template = f'''UNWIND $recs AS rec\nMERGE(n:{node_label} {{{node_key_name}: rec.{node_key_name}}})'''
    prop_names = [x for x in cols if x != node_key_name]
    if len(prop_names) > 0:
        template = template + '\n' + _make_set_clause(prop_names)
    return template + '\nRETURN count(n) AS nodeLoadedCount'


def _make_rel_merge_query(source_target_labels: Union[Tuple[str, str], str],
                          source_node_key: Union[Tuple[str, str], str],
                          target_node_key: Union[Tuple[str, str], str],
                          rel_type: str,
                          cols: ArrayLike,
                          rel_key: str = None):
    source_target_label_map = _make_map(source_target_labels)
    source_node_key_map = _make_map(source_node_key)
    target_node_key_map = _make_map(target_node_key)

    merge_statement = f'MERGE(s)-[r:{rel_type}]->(t)'
    if rel_key is not None:
        merge_statement = f'MERGE(s)-[r:{rel_type} {{{rel_key}: rec.{rel_key}}}]->(t)'

    template = f'''UNWIND $recs AS rec
    MATCH(s:{source_target_label_map[0]} {{{source_node_key_map[0]}: rec.{source_node_key_map[1]}}})
    MATCH(t:{source_target_label_map[1]} {{{target_node_key_map[0]}: rec.{target_node_key_map[1]}}})\n''' + merge_statement
    prop_names = [x for x in cols if x not in [rel_key, source_node_key_map[1], target_node_key_map[1]]]
    if len(prop_names) > 0:
        template = template + '\n' + _make_set_clause(prop_names, 'r')
    return template + '\nRETURN count(r) AS relLoadedCount'


def chunks(xs, n: int = 10_000):
    """
    split an array-like objects into chunks of size n.

    Parameters
    -------
    :param n: int
        The size of chunk. The last chunk will be the remainder if there is one.
    """
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]

def load_nodes(node_df: pd.DataFrame,
               node_key_col: str,
               node_label: str,
               chunk_size: int = 5_000,
               constraint: str = 'UNIQUE',
               neo4j_uri: str = 'bolt://localhost:7687',
               neo4j_password: str = 'password',
               neo4j_username: str = 'neo4j'):
    """
    Load nodes from a dataframe.

    Parameters
    -------
    :param node_df: pd.DataFrame
        The dataframe containing node data
    :param node_key_col: str
        The column of the dataframe to use as the MERGE key property
    :param node_label: str
        The node label to use (only one allowed).
    :param chunk_size: int , default 5_000
        The chunk size to use when batching rows for loading
    :param constraint: str , default "UNIQUE"
        The constraint to use for the node key. Can be "UNIQUE", "KEY", or None.
        More details at https://neo4j.com/docs/cypher-manual/current/constraints/examples/#constraints-examples-node-uniqueness.
        Using 'None' (no node constraint) can result in very poor load performance.
    :param neo4j_uri: str , default "bolt://localhost:7687"
        The uri for the Neo4j database
    :param neo4j_password: str , default "password"
        The password for the Neo4j database
    :param neo4j_username: str , default "neo4j"
        The password for the Neo4j database
    """

    print(f'======  loading {node_label} nodes  ======')

    records = node_df.to_dict('records')
    total = len(records)
    print(f'staged {total:,} records')
    with GraphDatabase.driver(neo4j_uri,
                              auth=(neo4j_username, neo4j_password)) as driver:
      if constraint:
        constraint = constraint.upper()
        if constraint not in ["UNIQUE", "KEY"]:
          raise ValueError(f'constraint must be one of ["UNIQUE", "KEY", None] but was {constraint}')
        const_query = _make_constraint_query(constraint, node_label, node_key_col)
        print(f'\ncreating constraint:\n```\n{const_query}\n```\n')
        driver.execute_query(const_query)

      query = _make_node_merge_query(node_key_col, node_label, node_df.columns.copy())
      print(f'\nusing this Cypher query to load data:\n```\n{query}\n```\n')
      cumulative_count = 0
      for recs in chunks(records, chunk_size):
          res = driver.execute_query(query, parameters_={'recs': recs})
          cumulative_count += res[0][0][0]
          print(f'loaded {cumulative_count:,} of {total:,} nodes')


#load neo4j credentials

load_dotenv('/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/neo4j-gnn-llm-example/configs/db.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

print(os.getenv('NEO4J_URI'))

pdb.set_trace()

for ind, node_type in skb.node_type_dict.items():
  single_node_type_df = (node_df[node_df['type']==node_type]
                         .drop(columns=['type']))
  node_label = format_node_label(node_type)
  load_nodes(single_node_type_df,
                   'nodeId',
                   node_label,
                   neo4j_uri=NEO4J_URI,
                   neo4j_password=NEO4J_PASSWORD)
  


rel_df = pd.DataFrame(
    torch.cat([skb.edge_index,
               skb.edge_types.reshape(1, skb.edge_types.size()[0])],
              dim=0).t(),
     columns = ['src', 'tgt', 'typeId'])


rel_types = skb.edge_type_dict

def format_rel_type(s):
  return re.sub('[^0-9A-Z]+', '_', s.upper())

# creating unifying node label for relationship load

with GraphDatabase.driver(NEO4J_URI,
                              auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
  driver.execute_query('MATCH(n) SET n:_Entity_')
  driver.execute_query('CREATE CONSTRAINT unique__entity__nodeid IF NOT EXISTS FOR (n:_Entity_) REQUIRE n.nodeId IS UNIQUE')

def load_rels(rel_df: pd.DataFrame,
              source_target_labels: Union[Tuple[str, str], str],
              source_node_key: Union[Tuple[str, str], str],
              target_node_key: Union[Tuple[str, str], str],
              rel_type: str,
              rel_key: str = None,
              chunk_size: int = 10_000,
              neo4j_uri: str = 'bolt://localhost:7687',
              neo4j_password: str = 'password',
              neo4j_username: str = 'neo4j'):
    """
    Load relationships from a dataframe.

    Parameters
    -------
    :param rel_df: pd.DataFrame
        The dataframe containing relationship data
    :param source_target_labels: Union[Tuple[str, str], str]
        The source and target node labels to use.
        Can pass a single string if source and target nodes have the same labels,
        otherwise a tuple of the form (source_node_label, target_node_label)
    :param source_node_key: Union[Tuple[str, str], str]
        The column of the dataframe to use as the source node MERGE key property.
        Can optionally pass a tuple of the form (source_node_key_name, df_column_name) to map as appropriate if the
        column name is different
    :param target_node_key: Union[Tuple[str, str], str]
        The column of the dataframe to use as the target node MERGE key property.
        Can optionally pass a tuple of the form (target_node_key_name, df_column_name) to map as appropriate if the
        column name is different
    :param rel_type: str
        The relationship type to use (only one allowed).
    :param rel_key: str
        A key to distinguish unique parallel relationships.
        The default behavior of this function is to assume only one instance of a relationship type between two nodes.
        A duplicate insert will have the behavior of overriding the existing relationship.
        If this behavior is undesirable, and you want to allow multiple instances of the same relationship type between
        two nodes (a.k.a parallel relationships), provide this key to use for merging relationships uniquely
    :param chunk_size: int , default 5_000
        The chunk size to use when batching rows for loading
    :param neo4j_uri: str , default "bolt://localhost:7687"
        The uri for the Neo4j database
    :param neo4j_password: str , default "password"
        The password for the Neo4j database
    :param neo4j_username: str , default "neo4j"
        The password for the Neo4j database
    """
    records = rel_df.to_dict('records')
    print(f'======  loading {rel_type} relationships  ======')
    total = len(records)
    print(f'staged {total:,} records')
    with GraphDatabase.driver(neo4j_uri,
                              auth=(neo4j_username, neo4j_password)) as driver:
      query = _make_rel_merge_query(source_target_labels, source_node_key,
                                  target_node_key, rel_type, rel_df.columns.copy(), rel_key)
      print(f'\nusing this cypher query to load data:\n```\n{query}\n```\n')
      cumulative_count = 0
      for recs in chunks(records, chunk_size):
          res = driver.execute_query(query, parameters_={'recs': recs})
          cumulative_count += res[0][0][0]
          print(f'loaded {cumulative_count:,} of {total:,} relationships')

for ind, edge_type in skb.edge_type_dict.items():
  single_rel_type_df = (rel_df[rel_df['typeId']==ind]
                         .drop(columns=['typeId']))
  rel_type = format_rel_type(edge_type)
  load_rels(single_rel_type_df,
              source_target_labels='_Entity_',
              source_node_key=('nodeId', 'src'),
              target_node_key=('nodeId', 'tgt'),
              rel_type=rel_type ,
              neo4j_uri=NEO4J_URI,
              neo4j_password=NEO4J_PASSWORD)


emb = torch.load('/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/neo4j-gnn-llm-example/data-loading/emb/prime/text-embedding-ada-002/doc/candidate_emb_dict.pt')

# format embedding records
emb_records = []
for k,v in tqdm(emb.items()):
  emb_records.append({"nodeId":k ,"textEmbedding": v.squeeze().tolist()})

# load embeddings

print(f'======  loading text embeddings ======')

total = len(emb_records)
print(f'staged {total:,} records')
with GraphDatabase.driver(NEO4J_URI,
                          auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:

  query = """
  UNWIND $recs AS rec
  MATCH(n:_Entity_ {nodeId: rec.nodeId})
  CALL db.create.setNodeVectorProperty(n, "textEmbedding", rec.textEmbedding)
  RETURN count(n) AS embeddingLoadedCount
  """
  print(f'\nusing this Cypher query to load data:\n```\n{query}\n```\n')
  cumulative_count = 0
  for recs in chunks(emb_records, 1_000):
      res = driver.execute_query(query, parameters_={'recs': recs})
      cumulative_count += res[0][0][0]
      print(f'loaded {cumulative_count:,} of {total:,} embeddings')


# create vector index

with GraphDatabase.driver(NEO4J_URI,
                          auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
  driver.execute_query('''
  CREATE VECTOR INDEX text_embeddings IF NOT EXISTS FOR (n:_Entity_) ON (n.textEmbedding)
  OPTIONS {indexConfig: {
  `vector.dimensions`: toInteger($dimension),
  `vector.similarity_function`: 'cosine'
  }}''', parameters_={'dimension': len(emb_records[0]['textEmbedding'])})
  driver.execute_query('CALL db.awaitIndex("text_embeddings", 300)')

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
reltype_emb = {format_rel_type(v): embedding_model.embed_query(v) for k,v in  skb.edge_type_dict.items()}
torch.save(reltype_emb, '/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/neo4j-gnn-llm-example/data-loading/emb/prime/text-embedding-ada-002/doc/reltype_emb_dict.pt')