import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from src.config import config

def load_logs():
    log_path = Path(config["query_log_path"])
    with log_path.open("r") as f:
        return [json.loads(line) for line in f]

def plot_latency():
    records = load_logs()
    df = pd.DataFrame(records)

    plt.figure(figsize=(8, 4))
    plt.hist(df["latency_sec"], bins=10, edgecolor="black")
    plt.title("Query Latency Distribution")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_score_dist():
    records = load_logs()
    all_scores = [score for r in records for score in r["rerank_scores"]]

    plt.figure(figsize=(8, 4))
    plt.hist(all_scores, bins=20, edgecolor="black", color="orange")
    plt.title("Reranker Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()