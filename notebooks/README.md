Hybrid search demo
"
Hybrid Search (with Sparse and Dense vectors) example app with Qdrant using Python. You can use any dataset with at least 1M data points and any embedding models. Make use of Binary Quantization for Dense vectors. The result of the hybrid search should be reranked using a late interaction model. Additionally, the payload should have a "user_id" field to filter by. Let's assume ten different users.
The Qdrant cluster should consist of two nodes. The collection should be replicated across the nodes and have three shards.
"

Qdrant in Docker on Intel/AMD:

#bash
    docker-compose up -d

    docker ps

    docker logs qdrant-node1 -f

    curl http://localhost:6333/cluster | jq

    docker-compose down -v

Public CSV dataset - Seattle Police reports (236MB), 1,440,681 records:
https://data.seattle.gov/api/views/tazs-3rd5/rows.csv?accessType=DOWNLOAD

... 