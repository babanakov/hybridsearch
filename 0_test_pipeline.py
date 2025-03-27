from src.data_preparation import load_and_clean_data, upload_to_qdrant # data_preparation_gpu for mac M3, data_preparation for generic CPU
from src.evaluation import run_batch_queries
from src.visualize import plot_latency, plot_score_dist
from src.config import config
import os
import time

# 1. Load and upload a small dev-mode dataset to Qdrant
print("🚀 Step 1: Loading and uploading data...")

# Update the dataset path in the config dynamically
dataset_path = "data/TMDB_movie_dataset_v11_small.csv"

df = load_and_clean_data(dataset_path=dataset_path)
upload_to_qdrant(df)

# 2. Run batch queries
print("\n🧪 Step 2: Running hybrid + rerank queries...")
queries = [
    "space survival and disaster",
    "romantic comedy in New York",
    "war movie with tanks",
    "epic fantasy battle",
    "robots taking over the world"
]
run_batch_queries(queries)

# 3. Visualize results (optional)
# print("\n📊 Step 3: Visualizing logs...")
# plot_latency()
# plot_score_dist()

print(f"\n🎉 Success!")