import pandas as pd
import numpy as np
import torch
import time
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from fastembed.sparse import SparseTextEmbedding
from qdrant_client import QdrantClient, models
import os
import sys
from src.config import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

start_time = time.time()

def load_and_clean_data(dataset_path=None, rows=None):
    print("📥 Dataset load started...")
    t0 = time.time()
    dataset_path = dataset_path or os.path.expanduser(config['dataset_path'])
    rows = rows if rows is not None else config.get("rows")
    df = pd.read_csv(dataset_path, nrows=rows)
    text_cols = ["title", "overview", "tagline", "genres", "keywords"]
    df[text_cols] = df[text_cols].fillna("")
    df["full_text"] = df[text_cols].agg(" ".join, axis=1)
    df["user_id"] = np.random.randint(1, 11, size=len(df))
    print(f"✅ Load complete - {len(df)} rows loaded in {time.time() - t0:.2f}s")
    return df

def upload_to_qdrant(df):
    print("📡 Connecting to Qdrant...")
    client = QdrantClient(
        host=config['qdrant']['host'], 
        port=config['qdrant']['port'],
        timeout=30
    )
    collection_name = config['collection_name']
    batch_size = config['batch_size']

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False, full_scan_threshold=20000)
            )
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000, memmap_threshold=20000),
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

    print(f"🚀 Using {device.upper()} for dense embedding")
    print("✨ Encoding and upload starting...")

    total_records = 0
    for batch_num, start in enumerate(range(0, len(df), batch_size), start=1):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end]
        texts = batch_df["full_text"].tolist()

        t_batch_start = time.time()

        t_sparse_start = time.time()
        sparse_vecs = list(sparse_model.embed(texts, batch_size))
        t_sparse_end = time.time()

        t_dense_start = time.time()
        with torch.inference_mode():
            dense_vecs = dense_model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
        t_dense_end = time.time()

        t_upsert_start = time.time()
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
        t_upsert_end = time.time()

        batch_time = time.time() - t_batch_start
        print(f"\n📦 Uploaded batch {batch_num} | points: {len(points)} | time: {batch_time:.2f}s")
        print(f"  ⏱️ Sparse: {t_sparse_end - t_sparse_start:.2f}s | Dense: {t_dense_end - t_dense_start:.2f}s | Upsert: {t_upsert_end - t_upsert_start:.2f}s")

        total_records += len(points)

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n✅ Upload complete. {len(df)} records in {str(elapsed_time)}, {int(len(df)/elapsed_time.total_seconds())} rec/sec")

if __name__ == "__main__":
    df = load_and_clean_data()
    upload_to_qdrant(df)
