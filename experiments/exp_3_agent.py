import numpy as np
import pandas as pd

import sys_config
from beta.agent import Agent
from beta.config import AgentConfig, AgentConfigChoseModel
from ingestion.kg_builder.entity import gliner_extraction_str_keywords
from tools.data_loader import get_csv_data_path, get_csv_log_path
from tools.similarity import calculate_cosine_similarity

from .exp_3_ingestion import load_extracted_entities


def test_agent_based_on_entity_filter(c_name: str, topk: int = 4):
    agent_config = AgentConfig(
        collection_name=c_name,
        is_hybrid_search=False,
        is_not_use_llm_knowledge=True,
        retrieve_top_k=topk,
    )
    AgentConfigChoseModel.chose_ollama_llm_model(
        agent_config, sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
    )
    AgentConfigChoseModel.chose_ollama_embedding(
        config=agent_config,
        model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
    )

    agent = Agent(agent_config)
    agent.use_ollama_llm()
    agent.use_ollama_embeddings()
    agent.generate_work_flow()

    df = pd.read_csv(str(get_csv_data_path("questions_and_answers")))
    questions_df = df["three_asking_ways"]
    answers_df = df["standard_answers"]

    # responsive web 30:60
    # 42:60
    questions = questions_df[30:60].tolist()
    std_answers = answers_df[30:60].ffill().tolist()

    # for std_answer in std_answers:
    #     print(std_answer)

    # load pre-extracted keywords
    g_kw = load_extracted_entities()

    agent_answers: list[str] = []
    retrieved_docs: list[str] = []
    query_keywords_list: list[str] = []
    for question in questions:
        # currently, just record the log, kws is not used for searching
        q_kws = gliner_extraction_str_keywords(question, g_kw.cluster)
        answer, docs, ref_list = agent.invoke_with_retrieved_contents(question, [])

        # collect logs
        query_keywords_list.append(", ".join(q_kws))
        agent_answers.append(answer)
        retrieved_docs.append("\n\n".join(docs))

    similarity_list = calculate_cosine_similarity(std_answers, agent_answers)
    print("mean, media, std")
    print(
        np.mean(similarity_list).round(4),
        np.median(similarity_list).round(4),
        np.std(similarity_list).round(4),
    )

    # filter the do not know answer's score
    # similarity_list_without_no_answer = list(filter(lambda v: v > 0.1, similarity_list))
    # print("mean, media, std just without do not know")
    # print(
    #     np.mean(similarity_list_without_no_answer).round(4),
    #     np.median(similarity_list_without_no_answer).round(4),
    #     np.std(similarity_list_without_no_answer).round(4),
    # )

    result_df = pd.DataFrame(
        {
            "question": questions,
            "question_kws": query_keywords_list,
            "std_answers": std_answers,
            "answers": agent_answers,
            "similarity": similarity_list,
            "retrieve": retrieved_docs,
        }
    )
    result_path = get_csv_log_path("exp_3_ret")
    result_df.to_csv(result_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    filtered_collection = "exp_3_entity_filtered"
    all_collection = "exp_3_entity"
    md_collection = "exp_3_md"

    print("Under 20 chunks")
    test_agent_based_on_entity_filter(md_collection, 20)

    print("Under 4 chunks")
    test_agent_based_on_entity_filter(md_collection, 4)
    # test_agent_based_on_entity_filter(all_collection)
