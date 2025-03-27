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
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from src.config import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optimize Torch thread usage
# Use available efficiency + performance cores effectively
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Detect if running on Apple Silicon (M1/M2/M3)
IS_APPLE_SILICON = sys.platform == "darwin" and "arm" in os.uname().machine

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


def process_batch(start_idx, df_chunk, dense_model_name, sparse_model_name, batch_size):
    import torch
    from sentence_transformers import SentenceTransformer
    from fastembed.sparse import SparseTextEmbedding

    device = "mps" if IS_APPLE_SILICON else ("cuda" if torch.cuda.is_available() else "cpu")
    dense_model = SentenceTransformer(dense_model_name, device=device).eval()

    sparse_provider = "CoreMLExecutionProvider" if IS_APPLE_SILICON else "CPUExecutionProvider"
    sparse_model = SparseTextEmbedding(
        model_name=sparse_model_name,
        providers=[sparse_provider],
        quantize=True
    )

    texts = df_chunk["full_text"].tolist()

    with torch.inference_mode():
        dense_vecs = dense_model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
    sparse_vecs = list(sparse_model.embed(texts, batch_size=batch_size))

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
    return points


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

    dense_model_name = config['embedding_models']['dense']
    sparse_model_name = config['embedding_models']['sparse']

    print("✨ Parallel encoding and upload starting...")
    start_time = time.time()

    batch_starts = list(range(0, len(df), batch_size))
    futures = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, start_idx in enumerate(batch_starts):
            df_chunk = df.iloc[start_idx:start_idx + batch_size].copy()
            futures.append(executor.submit(
                process_batch, start_idx, df_chunk, dense_model_name, sparse_model_name, batch_size
            ))

        for i, future in enumerate(futures):
            try:
                print(f"⏳ Waiting for future {i}...")
                points = future.result(timeout=600)  # 10 min timeout
                client.upsert(collection_name=collection_name, points=points)
                mem_usage = psutil.Process().memory_info().rss / 1024**3
                print(f"✅ Uploaded batch {i} | RAM: {mem_usage:.1f} GB")
            except TimeoutError:
                print(f"⏱️ Batch {i} timed out.")
            except Exception as e:
                print(f"❌ Error in batch {i}: {str(e)}")

    elapsed_time = timedelta(seconds=int(time.time() - start_time))
    print(f"✅ Upload complete. {len(df)} records in {str(elapsed_time)}")


if __name__ == "__main__":
    df = load_and_clean_data()
    upload_to_qdrant(df)
