
from smolagents import tool

import json
# from gretriever import *
# query = "What is the latest treatment for Alzheimer's disease?"
# g_retriever_model_path = '/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/neo4j-gnn-llm-example/stark_qa_v0_0/models/0_0_0_gnn-llm-llama3.1-8b_best_val_loss_ckpt.pt'
# result = retriever_response(query, g_retriever_model_path)
# print(result)



@tool
def g_retriever(query: str) -> str:
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
    # Example JSON response
    response_json = """
    {
      "question": [
        "Question: What is the latest treatment for Alzheimer's disease?\\nAnswer: "
      ],
      "response": [
        "gantacurium[/s]aducanumab[/s]verubecestat[/s]vivitrol[/s]memantine"
      ]
    }
    """
    response_dict = json.loads(response_json)
    response_dict["response"] = response_dict["response"][0].split("[/s]")
    return response_dict