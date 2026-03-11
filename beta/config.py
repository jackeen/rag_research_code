"""
The config for ingestion and agent.
"""
from dataclasses import dataclass
from enum import Enum
import sys_config


class CustomerMetadata(Enum):
    """
    The metadata set by customer is for Qdrant database, which is stored in the metadata of the documents.
    """
    CHUNK_REFERENCE_NAME = 'reference'


class RetrieveType(Enum):
    """
    Retrieving selection algorithm
    """
    SIMILARITY = 'similarity_score_threshold'
    MMR = 'mmr'


class FilterName(Enum):
    """
    The filters for ingestion before to store the docs into vector database.
    """
    COSINE_SIMILARITY_THRESHOLD = 'threshold'


@dataclass
class AgentConfig:
    """
    The configuration for the agent, it includes the default values following the previous research.
    The future experiments can change
    """
    name: str = 'beta'
    collection_name: str = 'default'

    # retrieve
    retrieve_top_k: int = 4
    retrieve_similarity: float = 0.0

    # embedding
    # the dimensions of embedding model from OpenAI are flexible
    embedding_model: str = sys_config.OPEN_AI_EMBEDDING_MODEL
    embedding_dimensions: int = 1536
    chunk_size: int = 100
    chunk_overlap: int = 0

    # db search


    # filter
    is_use_filter: bool = False
    filter: FilterName = None
    ingestion_filter_name: str = ''
    ingestion_similarity_filter_threshold: float = 0.6


if __name__ == '__main__':
    c = AgentConfig(collection_name='x')
    print(c.collection_name)

