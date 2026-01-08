#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add the parent directory to the path so we can import from the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_neo4j_connection():
    # Load environment variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, "..", "configs", "db.env")
    load_dotenv(env_path, override=True)
    
    neo4j_config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    
    print(f"Neo4j config: {neo4j_config}")
    
    try:
        with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
            # Test basic connection
            result = driver.execute_query("RETURN 'Connection successful' AS message")
            print(f"Connection test: {result.records[0]['message']}")
            
            # Check available databases
            result = driver.execute_query("SHOW DATABASES")
            print("Available databases:")
            for record in result.records:
                print(f"  - {record['name']}: {record['currentStatus']}")
            
            # Check node count
            result = driver.execute_query("MATCH (n) RETURN count(n) AS nodeCount")
            node_count = result.records[0]['nodeCount']
            print(f"Total nodes in database: {node_count}")
            
            # Check node labels
            result = driver.execute_query("CALL db.labels()")
            print("Available node labels:")
            for record in result.records:
                print(f"  - {record['label']}")
            
            # Check if the specific _Entity_ label exists
            result = driver.execute_query("MATCH (n:_Entity_) RETURN count(n) AS entityCount")
            entity_count = result.records[0]['entityCount']
            print(f"Nodes with _Entity_ label: {entity_count}")
            
            # Check for vector indexes
            result = driver.execute_query("CALL db.indexes()")
            print("Available indexes:")
            for record in result.records:
                print(f"  - {record['name']}: {record['type']} on {record['labelsOrTypes']}")
            
            # Check for vector embeddings
            if entity_count > 0:
                result = driver.execute_query("MATCH (n:_Entity_) RETURN n.nodeId, n.name, n.textEmbedding IS NOT NULL AS hasEmbedding LIMIT 5")
                print("Sample _Entity_ nodes:")
                for record in result.records:
                    print(f"  - ID: {record['n.nodeId']}, Name: {record['n.name']}, Has Embedding: {record['hasEmbedding']}")
            
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

if __name__ == "__main__":
    test_neo4j_connection()
