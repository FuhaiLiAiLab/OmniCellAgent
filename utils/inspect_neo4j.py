#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

def inspect_neo4j_data():
    # Load environment variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, "..", "configs", "db.env")
    load_dotenv(env_path, override=True)
    
    neo4j_config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    
    try:
        with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
            # Check all nodes and their properties
            result = driver.execute_query("MATCH (n) RETURN n LIMIT 10")
            print("Sample nodes in database:")
            for i, record in enumerate(result.records):
                node = record['n']
                print(f"  Node {i+1}:")
                print(f"    Labels: {list(node.labels)}")
                print(f"    Properties: {dict(node)}")
                print()
            
            # Check if there are nodes with nodeId property
            result = driver.execute_query("MATCH (n) WHERE n.nodeId IS NOT NULL RETURN labels(n) as labels, n.nodeId as nodeId, keys(n) as properties LIMIT 5")
            print("Nodes with nodeId property:")
            for record in result.records:
                print(f"  Labels: {record['labels']}, NodeId: {record['nodeId']}, Properties: {record['properties']}")
            
            # Check if there are nodes with textEmbedding property
            result = driver.execute_query("MATCH (n) WHERE n.textEmbedding IS NOT NULL RETURN labels(n) as labels, size(n.textEmbedding) as embeddingSize LIMIT 5")
            print("Nodes with textEmbedding property:")
            for record in result.records:
                print(f"  Labels: {record['labels']}, Embedding size: {record['embeddingSize']}")
            
            # Try to find vector indexes
            try:
                result = driver.execute_query("SHOW INDEXES")
                print("Available indexes:")
                for record in result.records:
                    print(f"  - {record}")
            except Exception as e:
                print(f"Could not list indexes: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_neo4j_data()
