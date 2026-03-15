from tools.data_loader import get_book_path, BookNames, References
from beta.ingestor import Ingestor
from beta.config import AgentConfig, AgentConfigChoseModel
import sys_config


def test_beta_ingestion():
    collection_name = 'test'
    agent_config = AgentConfig(
        collection_name=collection_name,
        chunk_size=500,
        chunk_overlap=100
    )
    AgentConfigChoseModel.chose_ollama_llm_model(
        config=agent_config,
        model_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
    )
    # AgentConfigChoseModel.chose_ollama_embedding(
    #     config=agent_config,
    #     model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    #     dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS
    # )
    AgentConfigChoseModel.chose_ollama_embedding(
        config=agent_config,
        model_name=sys_config.TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_384,
        dimensions=sys_config.TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_DIMENSIONS
    )

    book_path = str(get_book_path(BookNames.INTRO_WEB_DEV_1.value))
    ingestor = Ingestor(agent_config)
    # ingestor.use_ollama_embeddings()
    ingestor.use_transformer_embeddings()
    ingestor.force_create_collection()
    n = ingestor.ingest_pdf(book_path, References.INTRO_WEB_DEV_1.value)
    print(f'Ingested {n} docs in {collection_name}')


if __name__ == '__main__':
    test_beta_ingestion()

