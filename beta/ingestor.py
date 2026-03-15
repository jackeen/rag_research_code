import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
import dotenv
import sys_config
from tools import similarity
from .config import AgentConfig, CustomerMetadata, VectorNames, AgentConfigChoseModel


class Ingestor:
    """
    The tool designed to split the files into the documents.
    """

    embeddings: Embeddings
    sparse_embeddings: FastEmbedSparse
    qdrant_client: QdrantClient

    def __init__(self, config: AgentConfig):
        self.config = config

        # The hybrid mode needs it, the default model is Qdrant bm25 in this class if not set anything
        self.sparse_embeddings = FastEmbedSparse(model=self.config.sparse_embedding_model)

        # For collection operation and langchain document store interface
        self.qdrant_client = QdrantClient(host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT)

    def use_openai_embeddings(self):
        dotenv.load_dotenv()
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions
        )

    def use_ollama_embeddings(self):
        self.embeddings = OllamaEmbeddings(
            base_url=sys_config.OLLAMA_URL_BASE,
            model=self.config.embedding_model,
        )

    def use_transformer_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
        )

    def create_collection(self):
        """
        To create a collection in the vector database following the config.
        The creation of the collection is done by Qdrant client, and the Qdrant supports similarity and hybrid search.
        :return:
        """
        if not self.qdrant_client.collection_exists(self.config.collection_name):
            if self.config.is_hybrid_search:
                # https://qdrant.tech/documentation/concepts/hybrid-queries/
                self.qdrant_client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config={
                        VectorNames.DENSE.value: VectorParams(
                            size=self.config.embedding_dimensions,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        VectorNames.SPARSE.value: SparseVectorParams(
                            index=SparseIndexParams(
                                on_disk=False
                            )
                        )
                    }
                )
            else:
                self.qdrant_client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimensions,
                        distance=Distance.COSINE
                    )
                )
            print(f'Collection {self.config.collection_name} created')
        else:
            print(f'Collection {self.config.collection_name} already exists')

    def force_create_collection(self):
        if self.qdrant_client.collection_exists(self.config.collection_name):
            self.qdrant_client.delete_collection(self.config.collection_name)
            print(f'Collection {self.config.collection_name} is renewing')
        self.create_collection()

    def chunk_documents_from_file_path(self, file_path: str) -> list[Document]:
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
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=[r"(?<=[。！？\.!?])\s"],
            is_separator_regex=True,
        )
        split_docs = r_splitter.split_documents(docs)
        return split_docs

    def ingest_pdf(self, file_path: str, file_ref: str) -> int:
        """
        Ingest a pdf file
        :param file_path: the path of the file to ingest
        :param file_ref: the reference of the given file
        :return: the number of documents that were successfully ingested
        """
        docs = self.chunk_documents_from_file_path(file_path)
        self.save_docs(docs, file_ref)
        return len(docs)

    def ingest_pdf_with_threshold_filter(self, file_path: str, std_answers: list[str], file_ref: str) -> int:
        """
        Ingest a pdf file and filter the documents by cosine similarity threshold that were successfully ingested
        :param file_path:
        :param std_answers:
        :param file_ref: the reference of the given file
        :return: the number of documents that were successfully ingested
        """
        docs = self.chunk_documents_from_file_path(file_path)
        chunks = [doc.page_content for doc in docs]
        max_similarities = similarity.calculate_max_cosine_similarities(chunks, std_answers)
        # print(f"max_similarities: {max_similarities}")
        filtered_docs = []
        for doc, max_sim in zip(docs, max_similarities):
            if max_sim >= self.config.ingestion_similarity_filter_threshold:
                filtered_docs.append(doc)
        # self.qdrant_store.add_documents(filtered_docs)
        self.save_docs(filtered_docs, file_ref)
        return len(filtered_docs)

    def save_docs(self, docs: list[Document], file_ref: str) -> int:
        qdrant_store: QdrantVectorStore
        if self.config.is_hybrid_search:
            qdrant_store = QdrantVectorStore(
                retrieval_mode=RetrievalMode.HYBRID,
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                vector_name=VectorNames.DENSE.value,
                sparse_vector_name=VectorNames.SPARSE.value,
            )
        else:
            qdrant_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings,
            )

        for doc in docs:
            doc.metadata[CustomerMetadata.CHUNK_REFERENCE_NAME.value] = file_ref
        qdrant_store.add_documents(docs)
        return len(docs)


if __name__ == '__main__':
    pass
