"""
This experiment leverages the local RAG system powered by local Gemma models.
Its evaluation depends on cosine similarity.

It used chunk size 512 about 126 tokens
"""

import numpy as np
import pandas as pd

import sys_config
from beta.agent import Agent
from beta.config import AgentConfig, AgentConfigChoseModel, FilterName
from beta.ingestor import Ingestor
from tools.data_loader import (
    # BookNames,
    PageGroupNames,
    get_book_path,
    get_csv_data_path,
    get_csv_log_path,
    # get_excel_data_path,
)

# from tools.excel_operator import (
#     copy_qa_sheet_as_target_sheet,
#     write_target_sheet_column_cells,
# )
# from tools.similarity import calculate_cosine_similarity_transformer_embedding
from tools.similarity import calculate_cosine_similarity

# from tools.summary_recorder import SummaryEntity, append_summary


def ingest_full_data(ingestor: Ingestor, book_file_name: str, book_ref: str):
    """
    To ingest a book into database without any other steps.
    :param ingestor: the instance of Ingestor for specific task
    :param book_file_name:
    :param book_ref:
    :return: the number of documents that ingested in database
    """
    book_file_path = str(get_book_path(book_file_name))
    doc_n = ingestor.ingest_pdf(book_file_path, book_ref)
    c_name = ingestor.config.collection_name
    print(f"Ingested {doc_n} documents into {c_name}.")


def ingest_filtered_data(
    ingestor: Ingestor, book_file_name: str, target_data_file_name: str, book_ref: str
):
    """
    To ingest a book into database with the threshold filter before loading in the database.
    :param ingestor:
    :param book_file_name:
    :param target_data_file_name:
    :param book_ref:
    :return:
    """
    target_data_path = str(get_csv_data_path(target_data_file_name))
    df = pd.read_csv(target_data_path)
    std_answer_col = df["standard_answers"]

    # here includes all standard answers, which follows the previous researcher, but it wastes some computing resources
    # consider use smaller collection of answers based on which book is used in ingestion process
    std_answer_list = std_answer_col.dropna().loc[lambda s: s != ""].tolist()

    # renew the collection by given name
    # ingestor.force_create_collection()
    book_file_path = str(get_book_path(book_file_name))
    doc_n = ingestor.ingest_pdf_with_threshold_filter(
        book_file_path, std_answer_list, book_ref
    )
    threshold = ingestor.config.ingestion_similarity_filter_threshold
    print(
        f"Ingested {doc_n} documents into {ingestor.config.collection_name} by threshold {threshold}."
    )


def asking_agent(agent: Agent, data_file_name: str, book_index: int):
    c_name = agent.config.collection_name
    print(f"Asking questions in book {book_index} based on {c_name} collection")

    qa_data_path = get_csv_data_path(data_file_name)
    df = pd.read_csv(qa_data_path)

    question_col = df["three_asking_ways"]

    question_number_per_book = 30
    start_index = book_index * question_number_per_book

    # each book 30 questions
    question_slice = question_col.iloc[
        start_index : start_index + question_number_per_book
    ]

    # get answers from the RAG system
    answers = []
    contexts = []
    questions = []
    for question in question_slice:
        answer, context, _ = agent.invoke_with_retrieved_contents(question, [])
        questions.append(question)
        answers.append(answer)
        contexts.append("\n\n".join(context))

    # evaluation
    std_answers = (
        df["standard_answers"]
        .ffill()
        .tolist()[start_index : start_index + question_number_per_book]
    )
    similarities = calculate_cosine_similarity(
        s1_list=answers,
        s2_list=std_answers,
    )

    print(f"book {book_index} statistic result:")
    print("mean, median, standard deviation")
    print(
        np.mean(similarities).round(4),
        np.median(similarities).round(4),
        np.std(similarities).round(4),
    )

    log_df = pd.DataFrame(
        {
            "question": questions,
            "std_answers": std_answers,
            "answers": answers,
            "similarity": similarities,
            "retrieve": contexts,
        }
    )
    log_path = get_csv_log_path(f"baseline_ret_{book_index}")
    log_df.to_csv(log_path, index=False, encoding="utf-8")


def run(book_i, book_file_name, threshold_for_filter=0.0):
    """
    Start the experiment in one book.
    :return: None
    """
    # init collections
    # collection_name = "baseline_exp"
    filtered_collection_name = f"baseline_exp_filtered_book_{book_i}"

    # the agent config for filter test
    filtered_agent_config = AgentConfig(
        collection_name=filtered_collection_name,
        is_use_filter=True,
        is_not_use_llm_knowledge=True,
        filter=FilterName.COSINE_SIMILARITY_THRESHOLD,
        ingestion_similarity_filter_threshold=threshold_for_filter,
    )
    AgentConfigChoseModel.chose_ollama_llm_model(
        filtered_agent_config, sys_config.OLLAMA_GEMMA_MODEL_4_E2B
    )
    AgentConfigChoseModel.chose_ollama_embedding(
        filtered_agent_config,
        sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
        sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_DIMENSIONS,
    )

    # Prepare current book reference for ingestion workflow
    # book_ref = References[book_name].value
    book_ref = ""

    # filter ingestor
    ingestor_with_filter = Ingestor(filtered_agent_config)
    ingestor_with_filter.use_ollama_embeddings()
    ingestor_with_filter.force_create_collection()

    # prepare the vector database for current book
    # ingest_full_data(ingestor, book_file_name, book_ref)
    ingest_filtered_data(
        ingestor_with_filter, book_file_name, "questions_and_answers", book_ref
    )

    # agent with filtered data collection
    agent_with_filter = Agent(filtered_agent_config)
    agent_with_filter.use_ollama_llm()
    agent_with_filter.use_ollama_embeddings()
    agent_with_filter.generate_work_flow()

    # asking system
    asking_agent(agent_with_filter, "questions_and_answers", book_i)


def bach_run():
    # 124568 books
    threshold_arr = [0.5, 0.55, 0.72, 0.7, 0.72, 0.5]

    for i, book in enumerate(PageGroupNames):
        run(i, book.value + ".pdf", threshold_arr[i])


if __name__ == "__main__":
    bach_run()
