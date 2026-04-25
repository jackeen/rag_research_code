"""
Qdrant demon
"""

# from typing import Sequence

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Payload,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import sys_config


def embedding(content: str) -> list[float]:
    res = ollama.embeddings(
        model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768, prompt=content
    )
    return list(res.embedding)


client = QdrantClient(host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT)


def dense_query():
    res = client.query_points(
        collection_name="exp_3_md",
        query=embedding("color-scheme"),
        limit=4,
        score_threshold=0.5,
    )
    for p in res.points:
        print("------------------------------------------")
        print(p.score)
        if p.payload:
            print(p.payload.get("page_content"))


if __name__ == "__main__":
    dense_query()
