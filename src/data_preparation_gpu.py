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
import subprocess
import threading
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
NUM_WORKERS = 8

utilization_line = ""
monitor_active = True


def system_monitor():
    global utilization_line
    while monitor_active:
        try:
            gpu_output = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"
            ])
            gpu_util, gpu_mem = gpu_output.decode().strip().split(',')
            gpu_status = f"GPU {gpu_util.strip()}% | {gpu_mem.strip()} MiB"
        except Exception:
            gpu_status = "GPU unavailable"

        cpu_util = psutil.cpu_percent(interval=0.1)
        ram_usage = psutil.Process().memory_info().rss / 1024**3

        utilization_line = f"\r📊 CPU: {cpu_util:.1f}% | RAM: {ram_usage:.1f} GB | {gpu_status}"
        print(utilization_line, end="", flush=True)
        time.sleep(0.3)


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


def sparse_worker(sub_texts):
    model = SparseTextEmbedding(
        model_name=config['embedding_models']['sparse'],
        providers=["CPUExecutionProvider"],
        quantize=True,
        options={"intra_op_num_threads": 2}
    )
    return list(model.embed(sub_texts, batch_size=config['batch_size']))


def parallel_encode_sparse(texts, num_chunks=NUM_WORKERS):
    chunk_size = len(texts) // num_chunks + int(len(texts) % num_chunks > 0)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        results = executor.map(sparse_worker, chunks)
    sparse_vectors = []
    for chunk_result in results:
        sparse_vectors.extend(chunk_result)
    return sparse_vectors


def upload_to_qdrant(df):
    global monitor_active
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
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
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
    dense_model = SentenceTransformer(config['embedding_models']['dense'], device=device).eval()
    print(f"🚀 Using {device.upper()} for dense embedding")
    print("✨ Encoding and upload starting...")

    monitor_thread = threading.Thread(target=system_monitor, daemon=True)
    monitor_thread.start()
    start_time = time.time()

    for batch_num, start_idx in enumerate(range(0, len(df), batch_size)):
        batch_start = time.time()
        df_chunk = df.iloc[start_idx:start_idx + batch_size].copy()
        texts = df_chunk["full_text"].tolist()

        t_dense_start = time.time()
        with torch.inference_mode():
            dense_vecs = dense_model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
        t_dense_end = time.time()

        t_sparse_start = time.time()
        sparse_vecs = parallel_encode_sparse(texts, num_chunks=NUM_WORKERS)
        t_sparse_end = time.time()

        t_points_start = time.time()
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
        t_points_end = time.time()

        batch_time = time.time() - batch_start
        print(f"\n📦 Uploaded batch {batch_num} | points: {len(points)} | time: {batch_time:.2f}s")
        print(f"  ⏱️ Sparse: {t_sparse_end - t_sparse_start:.2f}s | Dense: {t_dense_end - t_dense_start:.2f}s | Upsert: {t_upsert_end - t_upsert_start:.2f}s")

    monitor_active = False
    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"\n✅ Upload complete. {len(df)} records in {str(elapsed_time)}, {int(len(df)/elapsed_time.total_seconds())} rec/sec")

if __name__ == "__main__":
    df = load_and_clean_data()
    upload_to_qdrant(df)
