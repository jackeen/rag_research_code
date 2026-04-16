
# 256 token
# EMBEDDING_MODEL_384 = 'sentence-transformers/all-MiniLM-L6-v2'
# EMBEDDING_MODEL_768 = 'sentence-transformers/all-mpnet-base-v2'

# 512 token
# https://www.ibm.com/new/announcements/ibm-granite-3-1-powerful-performance-long-context-and-more
# OLLAMA_GRANITE_EMBEDDING_MODEL_768 = 'granite-embedding:278m'


# local vector database for RAG
QDRANT_PORT = 6333
QDRANT_HOST = "localhost"
# this collection name for testing, different experiments should depend on different collection
QDRANT_COLLECTION_NAME = "RAG_Granite"

# local ollama models service for system testing and evaluation
OLLAMA_URL_BASE = "http://localhost:11434/"
OLLAMA_GRANITE_MODEL_3_3_8B = "granite3.3:8b"
OLLAMA_GRANITE_MODEL_4_3B_H = "granite4:3b-h"
OLLAMA_GRANITE_EMBEDDING_MODEL_768 = 'granite-embedding:278m'
OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS = 768

# OpenAI model
OPEN_AI_MODEL = "gpt-4.1-mini"
OPEN_AI_MODEL_5_4_MINI = "gpt-5.4-mini"

# OpenAI embedding model default 1536, 8192 token, genetic model
OPEN_AI_EMBEDDING_MODEL = "text-embedding-3-small"
OPEN_AI_EMBEDDING_DEFAULT_DIMENSIONS = 1536

# other embedding models
# based on attention, it is the upgrade version of BM25 which depends on statistic as the default spares
SPARES_QDRANT_ATTENTIONS = "Qdrant/all_miniLM_L6_v2_with_attentions"
# Bert based model, support synonyms, near-synonyms, abbreviations
SPARES_PP_EN_v2 = "prithivida/Splade_PP_en_v2"

# Transformer embeddings
# 256 token
TRANSFORMER_MINI_L6_V2_EMBEDDING_MODEL_384 = "all-MiniLM-L6-v2"
TRANSFORMER_MINI_L6_V2_EMBEDDING_MODEL_DIMENSIONS = 384

# 8192 token, focus on limited area, better anti-noisy
TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_384 = "ibm-granite/granite-embedding-small-english-r2"
TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_DIMENSIONS = 384

# 8192 token
TRANSFORMER_GRANITE_R2_EMBEDDING_MODEL_768 = "ibm-granite/granite-embedding-english-r2"
TRANSFORMER_GRANITE_R2_EMBEDDING_MODEL_DIMENSIONS = 768

# 32k token, multi language
TRANSFORMER_QWEN3_06B_EMBEDDING_MODEL_1024 = "Qwen/Qwen3-Embedding-0.6B"
TRANSFORMER_QWEN3_06B_EMBEDDING_MODEL_DIMENSIONS = 1024

# 2048 token, multi-language, licence limited
# TRANSFORMER_GOOGLE_GEMMA_EMBEDDING_MODEL_768 = "google/embeddinggemma-300m"
# TRANSFORMER_GOOGLE_GEMMA_EMBEDDING_MODEL_DIMENSIONS = 768

#
# TRANSFORMER_BAAI_BGE_BASE_EMBEDDING_MODEL_768 = "BAAI/bge-base-en-v1.5"
# TRANSFORMER_BAAI_BGE_BASE_EMBEDDING_MODEL_DIMENSIONS = 768

# about 30Gb VRAM cost
# TRANSFORMER_NV_EMBEDDING_MODEL_4096 = "nvidia/NV-Embed-v2"
# TRANSFORMER_NV_EMBEDDING_MODEL_DIMENSIONS = 4096
