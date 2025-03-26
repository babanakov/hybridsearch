import pandas as pd
import numpy as np
import torch
import time
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from fastembed.sparse import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.config import config, is_dev_mode, max_rows

# Start measuring total processing time
start_time = time.time()

def load_and_clean_data():
    print("📥 load_and_clean_data: started...")
    t0 = time.time()

    rows = max_rows() if is_dev_mode() else None
    df = pd.read_csv(os.path.expanduser(config['dataset_path']), nrows=rows)
    text_cols = ["title", "overview", "tagline", "genres", "keywords"]
    df[text_cols] = df[text_cols].fillna("")
    df["full_text"] = df[text_cols].agg(" ".join, axis=1)
    df["user_id"] = np.random.randint(1, 11, size=len(df))

    print(f"✅ load_and_clean_data: complete - {len(df)} rows loaded in {time.time() - t0:.2f}s")
    return df

def load_and_clean_data(dataset_path=None):
    print("📥 load_and_clean_data: started...")
    t0 = time.time()

    # Use the provided dataset path or fall back to the config
    dataset_path = dataset_path or os.path.expanduser(config['dataset_path'])

    rows = max_rows() if is_dev_mode() else None
    df = pd.read_csv(dataset_path, nrows=rows)
    text_cols = ["title", "overview", "tagline", "genres", "keywords"]
    df[text_cols] = df[text_cols].fillna("")
    df["full_text"] = df[text_cols].agg(" ".join, axis=1)
    df["user_id"] = np.random.randint(1, 11, size=len(df))

    print(f"✅ load_and_clean_data: complete - {len(df)} rows loaded in {time.time() - t0:.2f}s")
    return df

def upload_to_qdrant(df):
    print("📡 Connecting to Qdrant...")
    client = QdrantClient(
        host=config['qdrant']['host'], 
        port=config['qdrant']['port'],
        timeout=30  # 30 seconds, sometimes it takes a while to create a collection
        )
    collection_name = config['collection_name']
    batch_size = config['batch_size']

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100)
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                    full_scan_threshold=20000
                )
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000
        ),
        shard_number=3,
        replication_factor=2
    )
    print(f"✅ Qdrant collection '{collection_name}' created.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense_model = SentenceTransformer(config['embedding_models']['dense'], device=device).eval()
    sparse_model = SparseTextEmbedding(
        model_name=config['embedding_models']['sparse'],
        providers=["CPUExecutionProvider"],
        quantize=True
    )

    print("🚀 Starting upload to Qdrant...")
    total_records = 0
    for batch_num, start in enumerate(range(0, len(df), batch_size), start=1):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        texts = batch_df["full_text"].tolist()

        t0 = time.time()
        with torch.inference_mode():
            dense_vecs = dense_model.encode(texts, convert_to_numpy=True, batch_size=256)
        sparse_vecs = list(sparse_model.embed(texts, batch_size=256))

        points = []
        for i, (row, dense, sparse) in enumerate(zip(batch_df.itertuples(), dense_vecs, sparse_vecs)):
            points.append(models.PointStruct(
                id=int(row.Index),
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                    )
                },
                payload={
                    "user_id": int(row.user_id),
                    "title": row.title,
                    "overview": row.overview
                }
            ))

        client.upsert(collection_name=collection_name, points=points)
        total_records += len(points)
        print(f"📦 Uploaded batch {batch_num} | rows: {len(points)} | time: {time.time() - t0:.2f}s")
    
    end_time = time.time()
    elapsed_time = timedelta(seconds=int(end_time - start_time))
    print(f"✅ Upload complete. {total_records} records in {str(elapsed_time)}")

if __name__ == "__main__":
    df = load_and_clean_data()
    upload_to_qdrant(df)
