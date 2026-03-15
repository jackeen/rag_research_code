"""
This experiment leverages the local RAG system powered by ChatGPT 4.1 mini model.
Its evaluation depends on cosine similarity.

This is the branch experiment based exp 2 evl fix, which uses the same embedding model for evaluation.
It used chunk size 500
It changes the embedding model as granite 384
"""
import pandas as pd
import sys_config
from tools.data_loader import get_book_path, BookNames, get_excel_data_path, References
from tools.excel_operator import copy_qa_sheet_as_target_sheet, write_target_sheet_column_cells
from tools.similarity import calculate_cosine_similarity_transformer_embedding
from tools.summary_recorder import SummaryEntity, append_summary
from beta.ingestor import Ingestor
from beta.agent import Agent
from beta.config import AgentConfig, CustomerMetadata, FilterName, AgentConfigChoseModel


def ingest_full_data(ingestor: Ingestor, book_file_name: str, book_ref: str):
    """
    To ingest a book into database without any other steps.
    :param ingestor: the instance of Ingestor for specific task
    :param book_file_name:
    :param book_ref:
    :return: the number of documents that ingested in database
    """
    # renew_collection(c_name)
    # ingestor.force_create_collection()
    book_file_path = str(get_book_path(book_file_name))
    doc_n = ingestor.ingest_pdf(book_file_path, book_ref)
    c_name = ingestor.config.collection_name
    print(f'Ingested {doc_n} documents into {c_name}.')


def ingest_filtered_data(ingestor: Ingestor, book_file_name: str, target_data_file_name: str, book_ref: str):
    """
    To ingest a book into database with the threshold filter before loading in the database.
    :param ingestor:
    :param book_file_name:
    :param target_data_file_name:
    :param book_ref:
    :return:
    """
    target_data_path = str(get_excel_data_path(target_data_file_name))
    df = pd.read_excel(target_data_path, sheet_name='QAs')
    std_answer_col = df['standard_answers']

    # here includes all standard answers, which follows the previous researcher, but it wastes some computing resources
    # consider use smaller collection of answers based on which book is used in ingestion process
    std_answer_list = (std_answer_col
                       .dropna()
                       .loc[lambda s: s != '']
                       .tolist())

    # renew the collection by given name
    ingestor.force_create_collection()
    book_file_path = str(get_book_path(book_file_name))
    doc_n = ingestor.ingest_pdf_with_threshold_filter(book_file_path, std_answer_list, book_ref)
    threshold = ingestor.config.ingestion_similarity_filter_threshold
    print(f'Ingested {doc_n} documents into {ingestor.config.collection_name} by threshold {threshold}.')


def asking_agent(agent: Agent, data_file_name: str, sheet_name: str, book_name: str, book_index: int):
    c_name = agent.config.collection_name
    print(f'Asking questions in book {book_index}. {book_name} based on {c_name} collection')

    target_excel_path = str(get_excel_data_path(data_file_name))
    target_sheet_name = copy_qa_sheet_as_target_sheet(data_file_name, sheet_name)

    df = pd.read_excel(target_excel_path, sheet_name=target_sheet_name)
    question_col = df['three_asking_ways']

    question_number_per_book = 30
    start_index = book_index * question_number_per_book

    # each book 30 questions
    question_slice = question_col.iloc[start_index: start_index + question_number_per_book]

    # get answers from the RAG system
    answers = []
    for question in question_slice:
        answers.append(agent.invoke(question))

    # evaluation
    std_answers = df['standard_answers'].ffill().tolist()[start_index: start_index + question_number_per_book]
    # similarities = calculate_cosine_similarity_openai_embedding(answers, std_answers)
    similarities = calculate_cosine_similarity_transformer_embedding(
        model_name=sys_config.TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_384,
        s1_list=answers,
        s2_list=std_answers,
    )

    # write the results into the target file
    write_target_sheet_column_cells(data_file_name, target_sheet_name, 'answers', answers, start_index)
    write_target_sheet_column_cells(data_file_name, target_sheet_name, 'similarities', similarities, start_index)

    # record the summary
    summary = SummaryEntity(
        experiment_name=data_file_name,
        dataset_version='v1',
        ingest_file_name=book_name,
        filter_name='',
        model_name=agent.config.llm_model,
        db_collection_name=c_name,
        chunking_size=agent.config.chunk_size,
        chunking_overlap=agent.config.chunk_overlap,
        question_number=len(answers),
    )
    if agent.config.is_use_filter:
        if agent.config.filter == FilterName.COSINE_SIMILARITY_THRESHOLD:
            summary.filter_name = f'threshold_{int(agent.config.ingestion_similarity_filter_threshold*100)}'
    append_summary(summary, similarities)


def run(book_i, book_name, book_file_name, threshold_for_filter=0.6):
    """
    Start the experiment in one book.
    :return: None
    """
    # init collections
    collection_name = 'rag_gpt'
    # filtered_collection_name = 'rag_gpt_filtered'

    # these are output detail, questions and answers where is stored, they are also the sheet name in the summary
    data_file_name = 'exp_2_rag_gpt_emb_granite'
    # data_file_name_filter = f'exp_2_rag_gpt_filter_{int(threshold_for_filter*100)}'

    # this is for name used in database and sheet, which is more clear
    book_file_name_sign = book_file_name.replace('.', '_')

    collection_name = f'{collection_name}_{book_file_name_sign}'
    # filtered_collection_name = f'{filtered_collection_name}_{book_file_name_sign}'

    # the agent config for normal test
    agent_config = AgentConfig(
        collection_name=collection_name,
        chunk_size=500
    )
    AgentConfigChoseModel.chose_openai_llm_model(agent_config)
    AgentConfigChoseModel.chose_transformer_embedding(
        config=agent_config,
        model_name=sys_config.TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_384,
        dimensions=sys_config.TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_DIMENSIONS
    )

    # the agent config for filter test
    # filtered_agent_config = AgentConfig(
    #     collection_name=filtered_collection_name,
    #     is_use_filter=True,
    #     filter=FilterName.COSINE_SIMILARITY_THRESHOLD,
    #     ingestion_similarity_filter_threshold=threshold_for_filter,
    # )
    # AgentConfigChoseModel.chose_openai_llm_model(filtered_agent_config)
    # AgentConfigChoseModel.chose_openai_embedding(filtered_agent_config)

    # Prepare current book reference for ingestion workflow
    book_ref = References[book_name].value

    # normal ingestor
    ingestor = Ingestor(agent_config)
    ingestor.use_transformer_embeddings()
    ingestor.force_create_collection()

    # filter ingestor
    # ingestor_with_filter = Ingestor(filtered_agent_config)
    # ingestor_with_filter.use_openai_embeddings()

    # agent with normal data collection
    agent = Agent(agent_config)
    agent.use_openai_llm()
    agent.use_transformer_embeddings()
    agent.generate_work_flow()

    # agent with filtered data collection
    # agent_with_filter = Agent(filtered_agent_config)
    # agent_with_filter.use_openai_llm()
    # agent_with_filter.use_openai_embeddings()
    # agent_with_filter.generate_work_flow()

    # prepare the vector database for current book
    ingest_full_data(ingestor, book_file_name, book_ref)
    # ingest_filtered_data(ingestor_with_filter, book_file_name, data_file_name_filter, book_ref)

    # asking system
    asking_agent(agent, data_file_name, book_file_name_sign, book_name, book_i)
    # asking_agent(agent_with_filter, data_file_name_filter, book_file_name_sign, book_name, book_i)


def bach_run(filter_threshold: float):
    for i, book in enumerate(BookNames):
        run(i, book.name, book.value, filter_threshold)


if __name__ == '__main__':
    bach_run(0.0)
