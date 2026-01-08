from gretriever import *
query = "What is the latest treatment for Alzheimer's disease?"
g_retriever_model_path = '/storage1/fs1/fuhai.li/Active/di.huang/Research/LLM/RAG-MLLM/neo4j-gnn-llm-example/stark_qa_v0_0/models/0_0_0_gnn-llm-llama3.1-8b_best_val_loss_ckpt.pt'
result = retriever_response(query, g_retriever_model_path)
print(result)