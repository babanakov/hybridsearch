from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import torch

from src.config import config

# Initialize shared resources
_qdrant = QdrantClient(
    host=config["qdrant"]["host"],
    port=config["qdrant"]["port"],
    timeout=config["qdrant"].get("timeout", 60.0)
)

_dense_model = SentenceTransformer(config["embedding_models"]["dense"]).eval()
_sparse_model = SparseTextEmbedding(
    model_name=config["embedding_models"]["sparse"],
    providers=["CPUExecutionProvider"],
    quantize=True
)

def hybrid_search(query: str, user_ids, top_k: int = None):
    dense_vec = _dense_model.encode(query, convert_to_numpy=True).tolist()

    sparse_embedding = next(_sparse_model.embed(query))
    sparse_vec = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )

    if isinstance(user_ids, int):
        user_ids = [user_ids]

    # Pass the list directly to match_any
    filter_obj = models.Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchAny(any=user_ids),
            )
        ]
    )

    top_k = top_k or config["top_k"]

    requests = [
        models.SearchRequest(
            vector=models.NamedVector(name="dense", vector=dense_vec),
            filter=filter_obj,
            limit=top_k,
            with_payload=["title", "overview"]
        ),
        models.SearchRequest(
            vector=models.NamedSparseVector(name="sparse", vector=sparse_vec),
            filter=filter_obj,
            limit=top_k,
            with_payload=["title", "overview"]
        )
    ]

    results = _qdrant.search_batch(collection_name=config["collection_name"], requests=requests)
    return results

def rrf_fusion(results_list: list, k: int = 60):
    fused_scores = {}
    all_hits = {}

    for results in results_list:
        for rank, hit in enumerate(results, 1):
            if not hit.payload or "title" not in hit.payload:
                continue
            if hit.id not in fused_scores:
                fused_scores[hit.id] = 0.0
                all_hits[hit.id] = hit
            fused_scores[hit.id] += 1.0 / (rank + k)

    sorted_hits = sorted(all_hits.values(), key=lambda x: fused_scores[x.id], reverse=True)
    return sorted_hits
