"""
This experiment leverages the local RAG system powered by ChatGPT 4.1 mini model.
Its evaluation depends on cosine similarity.
"""


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pandas as pd
from openpyxl import load_workbook
import config
from tools.data_loader import get_book_path, BookNames, get_excel_data_path
from tools.excel_operator import copy_qa_sheet_as_target_sheet, write_target_sheet_column_cells
from tools.similarity import calculate_cosine_similarity
from tools.summary_recorder import SummaryEntity, append_summary
from beta import ingester
from beta.agent import Agent


client = QdrantClient(
    host=config.QDRANT_HOST,
    port=config.QDRANT_PORT,
)


def renew_collection(c_name):
    """
    To create an empty collection, if the collection is existed,
    delete it then create a new collection with the given name.
    :param c_name: the name of the collection
    :return: None
    """
    if client.collection_exists(c_name):
        client.delete_collection(c_name)
    client.create_collection(
        collection_name=c_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        )
    )
    print(f'The collection {c_name} is renewed.')


def ingest_full_data(c_name: str, book_file_name: str):
    """

    :param c_name:
    :param book_file_name:
    :return:
    """
    renew_collection(c_name)
    book_file_path = get_book_path(book_file_name)
    ingester.ingest_pdf(str(book_file_path), c_name)


def ingest_filtered_data(c_name: str, threshold, book_file_name: str, target_data_file_name: str):
    """

    :param c_name:
    :param threshold:
    :param book_file_name:
    :param target_data_file_name:
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
    renew_collection(c_name)
    book_file_path = get_book_path(book_file_name)
    ingester.ingest_pdf_with_threshold_filter(str(book_file_path), c_name, std_answer_list, threshold)


def asking_agent(c_name: str, data_file_name: str, sheet_name: str, book_name: str, book_index: int):
    print(f'Asking questions in book {book_index}. {book_name} based on {c_name} collection')

    target_excel_path = str(get_excel_data_path(data_file_name))
    target_sheet_name = copy_qa_sheet_as_target_sheet(data_file_name, sheet_name)

    df = pd.read_excel(target_excel_path, sheet_name=target_sheet_name)
    question_col = df['three_asking_ways']

    # each book 30 questions
    question_slice = question_col.iloc[book_index*30: book_index*30 + 30]

    # get answers from the RAG system
    agent = Agent(c_name)
    agent.compile_agent()
    answers = []
    for question in question_slice:
        answers.append(agent.invoke(question))

    # evaluation
    std_answers = df['standard_answers'].ffill().tolist()[book_index*30: book_index*30 + 30]
    similarities = calculate_cosine_similarity(answers, std_answers)

    # write the results into the target file
    write_target_sheet_column_cells(data_file_name, target_sheet_name, 'answers', answers, book_index*30)
    write_target_sheet_column_cells(data_file_name, target_sheet_name, 'similarities', similarities, book_index*30)

    # record the summary
    summary = SummaryEntity(
        experiment_name=data_file_name,
        dataset_version='v1',
        book_name=book_name,
        filter_name='threshold_73',
        model_name=config.OPEN_AI_MODEL,
        db_collection_name=c_name,
        question_number=len(answers),
    )
    append_summary(summary, similarities)


def run(book_i, book_name, book_file_name):
    """
    Start the experiment in one book.
    :return: None
    """
    # init collections
    collection_name = 'rag_gpt'
    filtered_collection_name = 'rag_gpt_filtered'
    perfect_threshold_for_filter = 0.73

    # these are output detail, questions and answers where is stored, they are also the sheet name in the summary
    data_file_name = 'exp_2_rag_gpt'
    data_file_name_filter = 'exp_2_rag_gpt_filter_73'

    # this is for name used in database and sheet, which is more clear
    book_file_name_sign = book_file_name.replace('.', '_')

    collection_name = f'{collection_name}_{book_file_name_sign}'
    filtered_collection_name = f'{filtered_collection_name}_{book_file_name_sign}'

    # prepare the vector database for current book
    ingest_full_data(collection_name, book_file_name)
    ingest_filtered_data(filtered_collection_name, perfect_threshold_for_filter, book_file_name, data_file_name_filter)

    # asking system
    asking_agent(collection_name, data_file_name, book_file_name_sign, book_name, book_i)
    asking_agent(filtered_collection_name, data_file_name_filter, book_file_name_sign, book_name, book_i)


def bach_run():
    for i, book in enumerate(BookNames):
        run(i, book.name, book.value)


def test_book_enumerating():
    target_excel_path = str(get_excel_data_path('exp_2_rag_gpt'))
    df = pd.read_excel(target_excel_path, sheet_name='QAs')
    question_col = df['three_asking_ways']

    offset = 30
    for i, book in enumerate(BookNames):
        df_slice = question_col.iloc[i*30: i*30 + 30]
        print(i, book.name, book.value)
        print(df_slice)
        print('-----------------\n')


if __name__ == '__main__':
    bach_run()
    # test_book_enumerating()
