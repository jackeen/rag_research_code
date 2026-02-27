"""
pip install langchain-community unstructured
from langchain_community.document_loaders import UnstructuredMarkdownLoader
to support MD file
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import config


def ingest_pdf(file_path, collection_name):
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        base_url=config.OLLAMA_URL_BASE
    )
    qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    qdrant_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        vector_name="text-vector",
        sparse_vector_name="sparse_text_vector",
    )

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=60,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    split_docs = r_splitter.split_documents(docs)
    qdrant_store.add_documents(split_docs)


