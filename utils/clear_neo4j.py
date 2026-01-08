#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

def clear_neo4j_database():
    """
    Completely clear all data from the Neo4j database
    """
    # Load environment variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, "..", "configs", "db.env")
    load_dotenv(env_path, override=True)
    
    neo4j_config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    
    print("Starting Neo4j database cleanup...")
    
    try:
        with GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["username"], neo4j_config["password"])) as driver:
            # First, check what we have before clearing
            result = driver.execute_query("MATCH (n) RETURN count(n) AS nodeCount")
            node_count = result.records[0]['nodeCount']
            print(f"Current node count: {node_count}")
            
            result = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS relCount")
            rel_count = result.records[0]['relCount']
            print(f"Current relationship count: {rel_count}")
            
            # Drop all indexes first
            print("\nDropping all indexes...")
            try:
                result = driver.execute_query("SHOW INDEXES")
                indexes = [record['name'] for record in result.records if record['name'] != 'id' and record['name'] != 'sys_id']
                
                for index_name in indexes:
                    try:
                        driver.execute_query(f"DROP INDEX {index_name}")
                        print(f"  Dropped index: {index_name}")
                    except Exception as e:
                        print(f"  Could not drop index {index_name}: {e}")
            except Exception as e:
                print(f"Could not list indexes: {e}")
            
            # Drop all constraints
            print("\nDropping all constraints...")
            try:
                result = driver.execute_query("SHOW CONSTRAINTS")
                constraints = [record['name'] for record in result.records]
                
                for constraint_name in constraints:
                    try:
                        driver.execute_query(f"DROP CONSTRAINT {constraint_name}")
                        print(f"  Dropped constraint: {constraint_name}")
                    except Exception as e:
                        print(f"  Could not drop constraint {constraint_name}: {e}")
            except Exception as e:
                print(f"Could not list constraints: {e}")
            
            # Delete all relationships first
            print("\nDeleting all relationships...")
            driver.execute_query("MATCH ()-[r]->() DELETE r")
            
            # Delete all nodes
            print("Deleting all nodes...")
            driver.execute_query("MATCH (n) DELETE n")
            
            # Verify the database is empty
            result = driver.execute_query("MATCH (n) RETURN count(n) AS nodeCount")
            final_node_count = result.records[0]['nodeCount']
            
            result = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS relCount")
            final_rel_count = result.records[0]['relCount']
            
            print(f"\nDatabase cleanup complete!")
            print(f"Final node count: {final_node_count}")
            print(f"Final relationship count: {final_rel_count}")
            
            if final_node_count == 0 and final_rel_count == 0:
                print("✅ Database successfully cleared!")
            else:
                print("⚠️  Warning: Database may not be completely empty")
                
    except Exception as e:
        print(f"Error clearing database: {e}")

if __name__ == "__main__":
    # Auto-confirm for this run
    print("Clearing the Neo4j database...")
    clear_neo4j_database()
