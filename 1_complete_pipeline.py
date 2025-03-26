from src.data_preparation import load_and_clean_data, upload_to_qdrant
from src.evaluation import run_batch_queries
from src.visualize import plot_latency, plot_score_dist
from src.config import config
import os
import time

# 1. Processing and loading complete dataset to Qdrant
print("🚀 Step 1: Loading and uploading data...")

# Update the dataset path in the config dynamically
dataset_path = config["dataset_path"] # Update the config with your path
rows = config["rows"] # Will read the entire file (it's big), set to desired number of rows in the config, i.e. 1000 for dev-mode

df = load_and_clean_data(dataset_path=dataset_path, rows=rows)
upload_to_qdrant(df)

# 2. Run test queries
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