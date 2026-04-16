from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

import sys_config

client = QdrantClient(host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT)

if __name__ == "__main__":
    keywords_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.matched_keywords",
                match=MatchAny(any=["Responsive Designs"]),
            )
        ]
    )
    samples, _ = client.scroll(
        collection_name="exp_3_entity_filtered", limit=2, scroll_filter=keywords_filter
    )

    print(samples)
