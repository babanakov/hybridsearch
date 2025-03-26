from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.config import config

# 🔁 Load reranker once globally
reranker = TextCrossEncoder(model_name=config["embedding_models"]["reranker"])

def rerank_results(query: str, documents: list, top_k: int = None):
    """
    Rerank a list of documents based on their relevance to the query.

    Args:
        query (str): The user query.
        documents (list): List of text documents (title + overview) to be reranked.
        top_k (int, optional): Number of top results to return. Defaults to config["rerank_top_n"].

    Returns:
        list of float: Relevance scores in descending order.
    """
    top_k = top_k or config["rerank_top_n"]
    scores = reranker.rerank(query=query, documents=documents, k=top_k)
    return scores
