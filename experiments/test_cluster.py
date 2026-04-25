from collections import Counter, defaultdict

import hdbscan
from sentence_transformers import SentenceTransformer

import sys_config

embedding_model = SentenceTransformer(
    sys_config.TRANSFORMER_MINI_L6_V2_EMBEDDING_MODEL_384
)

chunks = []

if __name__ == "__main__":
    embeddings = embedding_model.encode(chunks)
    cluster = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster.fit_predict(embeddings)
    cluster_dict = defaultdict(list)
    for term, label in zip(chunks, cluster_labels):
        if label != -1:
            cluster_dict[label].append(term)

    for cluster_group in cluster_dict.values():
        print("-------------------------")
        for c in cluster_group:
            print(c)
