
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

# local model service for system testing
OLLAMA_URL_BASE = "http://localhost:11434/"
OLLAMA_GRANITE_MODEL = "granite3.3:8b"
OLLAMA_GRANITE_EMBEDDING_MODEL_768 = 'granite-embedding:278m'

# OpenAI model
OPEN_AI_MODEL = "gpt-4.1-mini"

# OpenAI embedding model default 1536
OPEN_AI_EMBEDDING_MODEL = "text-embedding-3-small"

# for similarity
TRANSFORMER_EMBEDDING_MODEL_384 = "all-MiniLM-L6-v2"