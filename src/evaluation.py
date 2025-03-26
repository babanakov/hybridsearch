import json
import random
import time
from pathlib import Path
from src.config import config
from src.hybrid_search import hybrid_search, rrf_fusion
from src.reranker import rerank_results

def log_result(record: dict):
    log_path = Path(config["query_log_path"])
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

def run_batch_queries(query_set: list, user_ids: list = None, top_k: int = None, rerank_top_n: int = None):
    log_path = Path(config["query_log_path"])
    log_path.write_text("")  # Clear old logs

    top_k = top_k or config["top_k"]
    rerank_top_n = rerank_top_n or config["rerank_top_n"]
    user_ids = user_ids or list(range(1, 11))

    for i, query in enumerate(query_set, 1):
        user_id = random.choice(user_ids)
        start = time.time()

        dense_results, sparse_results = hybrid_search(query, user_id, top_k=top_k)
        fused = rrf_fusion([dense_results, sparse_results])
        candidates = fused[:rerank_top_n]

        documents = [hit.payload["title"] + " " + hit.payload["overview"] for hit in candidates]
        scores = rerank_results(query=query, documents=documents, top_k=rerank_top_n)

        elapsed = round(time.time() - start, 3)
        top_titles = [hit.payload["title"] for hit in candidates]

        record = {
            "query": query,
            "user_id": user_id,
            "latency_sec": elapsed,
            "top_titles": top_titles,
            "rerank_scores": [round(s, 4) for s in scores],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"✅ [{i}/{len(query_set)}] \"{query}\" | Top: {top_titles[0]} | ⏱️ {elapsed}s")
        log_result(record)