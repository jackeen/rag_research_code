from datetime import datetime

import numpy as np
import pandas as pd

import sys_config
from beta.agent import Agent
from beta.config import AgentConfig, AgentConfigChoseModel

# from ingestion.kg_builder.entity import gliner_extraction_str_keywords
from tools.data_loader import get_csv_data_path, get_csv_log_path
from tools.similarity import calculate_cosine_similarity

from .exp_3_config import CollectionNames


def test_agent_based_on_entity_filter(
    c_name: str,
    questions: list[str],
    std_answers: list[str],
    topk: int = 4,
    score_limit: float = 0.0,
    is_hybrid=False,
):
    agent_config = AgentConfig(
        collection_name=c_name,
        is_hybrid_search=is_hybrid,
        is_not_use_llm_knowledge=True,
        retrieve_top_k=topk,
        retrieve_similarity=score_limit,
    )
    # AgentConfigChoseModel.chose_ollama_llm_model(
    #     config=agent_config, model_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
    # )
    # AgentConfigChoseModel.chose_ollama_embedding(
    #     config=agent_config,
    #     model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    #     dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
    # )

    AgentConfigChoseModel.chose_ollama_llm_model(
        config=agent_config, model_name=sys_config.OLLAMA_GEMMA_MODEL_4_E2B
    )
    AgentConfigChoseModel.chose_ollama_embedding(
        config=agent_config,
        model_name=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
        dimensions=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_DIMENSIONS,
    )

    agent = Agent(agent_config)
    agent.use_ollama_llm()
    agent.use_ollama_embeddings()
    agent.generate_work_flow()

    agent_answers: list[str] = []
    retrieved_docs: list[str] = []
    # query_keywords_list: list[str] = []
    for question in questions:
        # currently, just record the log, kws is not used for searching
        # q_kws = gliner_extraction_str_keywords(question, g_kw.cluster)
        answer, docs, ref_list = agent.invoke_with_retrieved_contents(question, [])

        # collect logs
        # query_keywords_list.append(", ".join(q_kws))
        agent_answers.append(answer)
        retrieved_docs.append("\n\n".join(docs))

    similarity_list = calculate_cosine_similarity(std_answers, agent_answers)
    print("mean, media, std")
    print(
        np.mean(similarity_list).round(4),
        np.median(similarity_list).round(4),
        np.std(similarity_list).round(4),
    )

    result_df = pd.DataFrame(
        {
            "question": questions,
            "std_answers": std_answers,
            "answers": agent_answers,
            "similarity": similarity_list,
            "retrieve": retrieved_docs,
        }
    )
    result_path = get_csv_log_path("slim_exp_3_ret")
    result_df.to_csv(result_path, index=False, encoding="utf-8")


def exploring_qa():

    print("---------------------------")
    print(datetime.now().isoformat())
    print("---------------------------")

    q_list, a_list = load_questions_answers("questions_and_answers", 30, 60)

    # print("fixed chunking under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.FIX_CHUNKING.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=8,
    # )
    # print("fixed chunking under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.FIX_CHUNKING.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=4,
    # )

    # print("under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.FIX_CHUNKING_1_KEYWORD.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=8,
    # )
    # print("under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.FIX_CHUNKING_1_KEYWORD.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=4,
    # )

    # print("under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.FIX_CHUNKING_2_KEYWORDS.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=8,
    # )
    # print("under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.FIX_CHUNKING_2_KEYWORDS.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=4,
    # )

    # print("under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=8,
    # )
    # print("under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=4,
    # )

    # print("under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING_LLM.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=8,
    # )
    # print("under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING_LLM.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     topk=4,
    # )

    # print("under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     is_hybrid=True,
    #     topk=8,
    # )
    # print("under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     is_hybrid=True,
    #     topk=4,
    # )

    # print("under 8 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING_ML_PAGE.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     is_hybrid=True,
    #     topk=8,
    # )
    # print("under 4 chunks")
    # test_agent_based_on_entity_filter(
    #     c_name=CollectionNames.MD_HEAD_CHUNKING_ML_PAGE.value,
    #     questions=q_list,
    #     std_answers=a_list,
    #     is_hybrid=True,
    #     topk=4,
    # )

    print("under 8 chunks")
    test_agent_based_on_entity_filter(
        c_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH_GEMMA.value,
        questions=q_list,
        std_answers=a_list,
        is_hybrid=True,
        topk=8,
    )
    print("under 4 chunks")
    test_agent_based_on_entity_filter(
        c_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH_GEMMA.value,
        questions=q_list,
        std_answers=a_list,
        is_hybrid=True,
        topk=4,
    )


def extended_qa(
    c_name: str,
    questions: list[str],
    std_answers: list[str],
    is_hybrid_search: bool,
    topk: int = 4,
    score_limit: float = 0.5,
):
    test_agent_based_on_entity_filter(
        c_name=c_name,
        topk=topk,
        questions=questions,
        std_answers=std_answers,
        score_limit=score_limit,
        is_hybrid=is_hybrid_search,
    )


def load_questions_answers(
    file_name: str, start: int, end: int
) -> tuple[list[str], list[str]]:
    """
    Load the source of answers by given indexes,
    for example: 30,60 get book 2 anwers
    Arguments:
        start: the start index of answers list, from 1
        end: the end index of answers list
    return: the tuple of questions and answers (ground truth)
    """
    df = pd.read_csv(str(get_csv_data_path(file_name)))
    q_list = df["three_asking_ways"][start:end].to_list()
    a_list = df["standard_answers"][start:end].ffill().to_list()
    return (q_list, a_list)


def extended_exp(book_n: int):
    print(f"\nAgent QA for book {book_n}")

    if book_n == 1:
        q_list, a_list = load_questions_answers("questions_and_answers", 0, 30)
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_1_TMP.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_1.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )

    if book_n == 2:
        q_list, a_list = load_questions_answers("questions_and_answers", 30, 60)
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_2_TMP.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_2.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )

    if book_n == 4:
        q_list, a_list = load_questions_answers("questions_and_answers", 60, 90)
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_4_TMP.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_4.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )

    if book_n == 5:
        q_list, a_list = load_questions_answers("questions_and_answers", 90, 120)
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_5_TMP.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_5.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )

    if book_n == 6:
        q_list, a_list = load_questions_answers("questions_and_answers", 120, 150)
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_6_TMP.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_6.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )

    if book_n == 8:
        q_list, a_list = load_questions_answers("questions_and_answers", 150, 180)
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_8_TMP.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )
        extended_qa(
            c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_8.value,
            questions=q_list,
            std_answers=a_list,
            is_hybrid_search=True,
            topk=2,
            score_limit=0.0,
        )


if __name__ == "__main__":
    pass

    print("---------------------------")
    print(datetime.now().isoformat())
    print("---------------------------")

    exploring_qa()

    # extended_exp(1)
    # extended_exp(2)
    # extended_exp(4)
    # extended_exp(5)
    # extended_exp(6)
    # extended_exp(8)
