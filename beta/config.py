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

    CHUNK_REFERENCE_NAME = "reference"


class RetrieveType(Enum):
    """
    Retrieving selection algorithm
    """

    SIMILARITY = "similarity_score_threshold"
    MMR = "mmr"


class FilterName(Enum):
    """
    The filters for ingestion before to store the docs into vector database.
    """

    COSINE_SIMILARITY_THRESHOLD = "threshold"


class VectorNames(Enum):
    """
    The vector names for collection of database under hybrid searching.
    """

    DENSE = "text_dense_vector"
    SPARSE = "text_sparse_vector"


@dataclass
class AgentConfig:
    """
    The configuration for the agent, it includes the default values following the previous research.
    The future experiments can change
    """

    # the agent name
    name: str = "beta"

    # the system used collection stored in the database
    collection_name: str = "default"

    # vector database setting
    is_hybrid_search: bool = False

    # retrieve
    retrieve_top_k: int = 4
    retrieve_similarity: float = 0.0

    # generate
    is_not_use_llm_knowledge: bool = False

    # embedding
    # the dimensions of embedding model from OpenAI are flexible
    embedding_model: str = ""
    embedding_dimensions: int = 0
    sparse_embedding_model: str = sys_config.SPARES_PP_EN_v1
    chunk_size: int = 100
    chunk_overlap: int = 0

    # generation
    llm_model: str = ""

    # ingestion filter
    is_use_filter: bool = False
    filter: FilterName | None = None
    ingestion_filter_name: str = ""
    ingestion_similarity_filter_threshold: float = 0.6


class AgentConfigChoseModel:
    @staticmethod
    def chose_openai_llm_model(config: AgentConfig):
        config.llm_model = sys_config.OPEN_AI_MODEL

    @staticmethod
    def chose_ollama_llm_model(config: AgentConfig, model_name: str):
        config.llm_model = model_name

    @staticmethod
    def chose_transformer_llm_model(config: AgentConfig, model_name: str):
        config.llm_model = model_name

    @staticmethod
    def chose_openai_embedding(config: AgentConfig):
        config.embedding_model = sys_config.OPEN_AI_EMBEDDING_MODEL
        config.embedding_dimensions = sys_config.OPEN_AI_EMBEDDING_DEFAULT_DIMENSIONS

    @staticmethod
    def chose_ollama_embedding(config: AgentConfig, model_name: str, dimensions: int):
        config.embedding_model = model_name
        config.embedding_dimensions = dimensions

    @staticmethod
    def chose_transformer_embedding(
        config: AgentConfig, model_name: str, dimensions: int
    ):
        config.embedding_model = model_name
        config.embedding_dimensions = dimensions


if __name__ == "__main__":
    c = AgentConfig(collection_name="x")
    print(c.collection_name)
