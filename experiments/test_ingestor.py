from tools.data_loader import get_book_path, BookNames, References
from beta.ingestor import Ingestor
from beta.config import agent_config


def test_ingestion():
    collection_name = 'test'
    agent_config.collection_name = collection_name

    # config not default
    agent_config.chunk_size = 500
    agent_config.chunk_overlap = 100

    book_path = str(get_book_path(BookNames.INTRO_WEB_DEV_1.value))
    ingestor = Ingestor(agent_config)
    ingestor.force_create_collection()
    n = ingestor.ingest_pdf(book_path, References.INTRO_WEB_DEV_1.value)
    print(f'Ingested {n} docs in {collection_name}')


if __name__ == '__main__':
    test_ingestion()

