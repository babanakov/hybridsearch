# 📘 Companion Notes – Personal Insights from Building a Hybrid Search System

## 1. 🧭 Starting with the Right Data (and Problem)

I began this project using structured police incident data. While useful for testing how to build the system (schemas, ingestion, etc.), it didn’t offer much value for search. The data had little meaningful text, which made it hard to test dense or hybrid retrieval effectively.

I switched to a movie dataset instead — something rich in language (titles, descriptions, genres) and something I personally cared about. This helped me better judge the accuracy of results and test how dense and sparse vectors worked together.

**Lesson learned**: it’s easy to jump into building things without really thinking about the problem or the data. I’ve done it — and I now recognize it’s something many teams do too. Choosing the right dataset early would save time, make testing easier, and lead to better results.

## 2. 🏗️ Architecture & Design Decisions

- **Models**: I used SentenceTransformers for semantic search and SPLADE for sparse retrieval. Together, they balance speed, general coverage, and relevance.
- **Hybrid Strategy**: I used Reciprocal Rank Fusion (RRF) to combine dense and sparse results — simpler than score normalization and worked well out of the box.
- **Modularity**: I separated ingestion, search, reranking, and monitoring into their own files and notebooks. This makes it easier to reuse and adapt for presales demos or PoCs.
- **Demo UX**: I chose a Jupyter Notebook instead of building a UI. It made iteration faster and was good enough for showing results.
- **Infrastructure**: I used an EC2 `r5a.4xlarge` instance—a realistic choice for customer setups with large datasets and memory needs.

## 3. 🤯 Challenges & Lessons Learned

- **Sparse vectors were tricky**: Encoding with SPLADE had to be carefully batched. It used more memory than expected, and Qdrant indexing was slower.
- **Late reranking worked well**: Using a cross-encoder brought a clear relevance boost. But it made evaluation harder, and slowed things down — a trade-off worth knowing.
- **Filtering by user_id mattered**: This helped simulate real-world personalization. It became a useful way to talk about user-specific search and access control in demos.

## 4. 📊 What Could Be Next

- Integrate more data fields (plot, budget, popularity, etc.) for more sophisticated search scenarios.
- Add latency tracking dashboards (percentiles, outliers).
- Support real-time updates to the index.
- Try different rerankers (ColBERT, multilingual GTE).
- Build a tagging UI for human-in-the-loop feedback or clustering.

## 5. 🤖 How I Used AI to Work Faster

- I used LLMs to help draft pseudocode, sketch boilerplate, and test logic.
- I verified design decisions (e.g., how Qdrant handles sparse + dense).
- I iterated faster by using LLMs as a second pair of eyes — especially when tuning configs or trying out different model setups.
