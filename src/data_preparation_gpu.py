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
import psutil
from concurrent.futures import ProcessPoolExecutor
from src.config import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optimize for g6.4xlarge (L4 GPU + 16 vCPUs)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8  # g6.4xlarge: balance CPU for sparse parallelism


def load_and_clean_data(dataset_path=None, rows=None):
    print("\U0001F4E5 load_and_clean_data: started...")
    t0 = time.time()

    dataset_path = dataset_path or os.path.expanduser(config['dataset_path'])
    df = pd.read_csv(dataset_path, nrows=rows)
    text_cols = ["title", "overview", "tagline", "genres", "keywords"]
    df[text_cols] = df[text_cols].fillna("")
    df["full_text"] = df[text_cols].agg(" ".join, axis=1)
    df["user_id"] = np.random.randint(1, 11, size=len(df))

    print(f"✅ load_and_clean_data: complete - {len(df)} rows loaded in {time.time() - t0:.2f}s")
    return df


def encode_sparse(texts):
    sparse_model = SparseTextEmbedding(
        model_name=config['embedding_models']['sparse'],
        providers=["CPUExecutionProvider"],
        quantize=True
    )
    return list(sparse_model.embed(texts, batch_size=config['batch_size']))


def upload_to_qdrant(df):
    print("\U0001F4E1 Connecting to Qdrant...")
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
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False, full_scan_threshold=20000)
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000, memmap_threshold=20000
        ),
        shard_number=3,
        replication_factor=2
    )
    print(f"✅ Qdrant collection '{collection_name}' created.")

    dense_model = SentenceTransformer(config['embedding_models']['dense'], device=device).eval()
    print(f"🚀 Using device: {device} for dense embedding")
    print("✨ Encoding and upload starting...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for batch_num, start_idx in enumerate(range(0, len(df), batch_size)):
            batch_start = time.time()
            df_chunk = df.iloc[start_idx:start_idx + batch_size].copy()
            texts = df_chunk["full_text"].tolist()

            with torch.inference_mode():
                dense_vecs = dense_model.encode(texts, convert_to_numpy=True, batch_size=batch_size)

            sparse_future = executor.submit(encode_sparse, texts)
            sparse_vecs = sparse_future.result()

            points = []
            for i, (row, dense, sparse) in enumerate(zip(df_chunk.itertuples(), dense_vecs, sparse_vecs)):
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
            mem_usage = psutil.Process().memory_info().rss / 1024**3
            batch_time = time.time() - batch_start
            print(f"✅ Uploaded batch {batch_num} | points: {len(points)} | time: {batch_time:.2f}s | RAM: {mem_usage:.1f} GB")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"✅ Upload complete. {len(df)} records in {str(elapsed_time)}")


if __name__ == "__main__":
    df = load_and_clean_data()
    upload_to_qdrant(df)