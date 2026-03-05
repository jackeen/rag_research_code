"""
pip install langchain-community unstructured
from langchain_community.document_loaders import UnstructuredMarkdownLoader
to support MD file
"""

import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import dotenv
import config
from tools import similarity


dotenv.load_dotenv()


def get_qdrant_store(c_name) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model=config.OPEN_AI_EMBEDDING_MODEL)
    qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    qdrant_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=c_name,
        embedding=embeddings,
    )
    return qdrant_store


def chunk_documents_from_file_path(file_path: str) -> list[Document]:
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.page_content = doc.page_content.replace("\r\n", "\n").strip()
        # why the chinese sign can influence the number of chunks?
        doc.page_content = re.sub(r"([。！？])", r"\1 ", doc.page_content)

    # the configuration is not explain in the previous paper
    # 1000, 60
    # 100, 0
    # 300, 60
    # separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    # [r"(?<=[。！？\.!?])\s"] this can make the sentence keeping completely, why?
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        separators=[r"(?<=[。！？\.!?])\s"],
        is_separator_regex=True,
    )
    split_docs = r_splitter.split_documents(docs)
    return split_docs


def ingest_pdf(file_path: str, collection_name: str) -> int:
    """
    Ingest a pdf file
    :param file_path: the path of the file to ingest
    :param collection_name: the name of the collection in vector database
    :return: the number of documents that were successfully ingested
    """
    qdrant_store = get_qdrant_store(collection_name)
    docs = chunk_documents_from_file_path(file_path)
    qdrant_store.add_documents(docs)
    return len(docs)


def ingest_pdf_with_threshold_filter(
        file_path: str,
        collection_name: str,
        std_answers: list[str],
        threshold: float = 0.6,
) -> int:
    """
    Ingest a pdf file and filter the documents by cosine similarity threshold that were successfully ingested
    :param file_path:
    :param collection_name:
    :param std_answers:
    :param threshold:
    :return: the number of documents that were successfully ingested
    """
    qdrant_store = get_qdrant_store(collection_name)
    docs = chunk_documents_from_file_path(file_path)
    chunks = [doc.page_content for doc in docs]
    max_similarities = similarity.calculate_max_cosine_similarities(chunks, std_answers)
    # print(f"max_similarities: {max_similarities}")
    filtered_docs = []
    for doc, max_sim in zip(docs, max_similarities):
        if max_sim >= threshold:
            filtered_docs.append(doc)
    qdrant_store.add_documents(filtered_docs)
    return len(filtered_docs)

