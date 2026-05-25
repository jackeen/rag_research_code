"""
This module depends on 0.3.4 Ragas
"""

import csv
from typing import Sequence, cast

import dotenv
import numpy
import pandas as pd
from datasets import Dataset

# from openai import AsyncOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult

# from ragas.embeddings import OpenAIEmbeddings
from ragas.llms.base import llm_factory
from ragas.metrics import (
    # ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    # NoiseSensitivity,
    # ResponseRelevancy,
    # SemanticSimilarity,
)
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

import sys_config
from tools.data_loader import get_csv_log_path, get_no_tail_csv_log_path

# load env for OpenAI key
dotenv.load_dotenv()


def extract_retrieved_chunks(text: str) -> list[str]:
    text = str(text)
    if text is None or text == "":
        return []
    chunks = text.split("\n\n")
    raw_chunks: list[str] = []
    for c in chunks:
        raw_chunk = c.split("QCS:")[0]
        raw_chunks.append(raw_chunk)
    return raw_chunks


def evaluate_log(csv_name: str, book_i: int, filter: str) -> str:
    """Score the experiment by its log file and return the log file name, which includes score details"""
    log_file_path = get_no_tail_csv_log_path(csv_name)
    df = pd.read_csv(log_file_path)

    # just for book 2
    # df_skip = df.iloc[3:12]
    # df = pd.concat([df.iloc[0:3], df.iloc[12:30]])
    # print(df_skip["similarity"].to_list())

    questions = df["question"].tolist()
    standard_answers = df["std_answers"].tolist()
    answers = df["answers"].tolist()
    similarities = df["similarity"].to_list()
    retrieves = df["retrieve"].tolist()
    retrieved_chunks: list[list[str]] = []

    for raw_retrieve in retrieves:
        retrieved_chunks.append(extract_retrieved_chunks(raw_retrieve))

    data_dict = {
        "question": questions,
        "ground_truth": standard_answers,
        "answer": answers,
        "contexts": retrieved_chunks,
    }
    data_set = Dataset.from_dict(data_dict)

    # openai_client = AsyncOpenAI()
    # judge_emb = OpenAIEmbeddings(
    #     client=openai_client, model=sys_config.OPEN_AI_EMBEDDING_MODEL
    # )

    judge_llm = llm_factory(model=sys_config.OPEN_AI_MODEL)

    metrics = cast(
        Sequence[Metric],
        [
            ContextPrecision(llm=judge_llm, name="context_precision"),
            ContextRecall(llm=judge_llm, name="context_recall"),
            # ContextEntityRecall(llm=judge_llm, name="context_entity_recall"),
            Faithfulness(llm=judge_llm, name="faithfulness"),
            # NoiseSensitivity(
            #     llm=judge_llm, mode="relevant", name="noise_sensitivity_relevant"
            # ),
            # NoiseSensitivity(
            #     llm=judge_llm, mode="irrelevant", name="noise_sensitivity_irrelevant"
            # ),
            # SemanticSimilarity(embeddings=judge_emb, name="semantic_similarity"),
            # not work one
            # ResponseRelevancy(
            #     embeddings=judge_emb, llm=judge_llm, name="response_relevancy"
            # ),
        ],
    )

    score = evaluate(
        dataset=data_set,
        metrics=metrics,
        run_config=RunConfig(max_workers=2),
    )
    score = cast(EvaluationResult, score)
    result_dataframe = score.to_pandas()

    cp = result_dataframe["context_precision"].dropna().to_list()
    cr = result_dataframe["context_recall"].dropna().to_list()
    ff = result_dataframe["faithfulness"].dropna().to_list()

    cp_mean = numpy.around(numpy.array(cp).mean(), decimals=4)
    cr_mean = numpy.around(numpy.array(cr).mean(), decimals=4)
    ff_mean = numpy.around(numpy.array(ff).mean(), decimals=4)

    cp_median = numpy.around(numpy.median(numpy.array(cp)), decimals=4)
    cr_median = numpy.around(numpy.median(numpy.array(cr)), decimals=4)
    ff_median = numpy.around(numpy.median(numpy.array(ff)), decimals=4)

    # cosine similarity
    cs_mean = numpy.around(numpy.mean(numpy.array(similarities)), decimals=4)
    cs_median = numpy.around(numpy.median(numpy.array(similarities)), decimals=4)

    print(f"context_precision: {cp_mean}, {cp_median}")
    print(f"context_recall: {cr_mean}, {cr_median}")
    print(f"faithfulness: {ff_mean}, {ff_median}")

    #
    eval_log_df = pd.DataFrame(
        {
            "question": questions,
            "similarity": similarities,
            "context_precision": cp,
            "context_recall": cr,
            "faithfulness": ff,
        }
    )

    result_path = get_csv_log_path("exp_3_eval")
    eval_log_df.to_csv(result_path, encoding="utf-8", index=True)

    append_summary(
        "summary",
        {
            "book_index": book_i,
            "filter": filter,
            "question_n": len(similarities),
            "similarity_mean": cs_mean,
            "similarity_median": cs_median,
            "context_precision_mean": cp_mean,
            "context_precision_median": cp_median,
            "context_recall_mean": cr_mean,
            "context_recall_median": cr_median,
            "faithfulness_mean": ff_mean,
            "faithfulness_median": ff_median,
        },
    )

    return result_path.stem


def append_summary(target_file, data_dict):
    summary_path = get_no_tail_csv_log_path(target_file)
    fields = [
        "book_index",
        "filter",
        "question_n",
        "similarity_mean",
        "similarity_median",
        "context_precision_mean",
        "context_precision_median",
        "context_recall_mean",
        "context_recall_median",
        "faithfulness_mean",
        "faithfulness_median",
    ]

    # this should be at before the open file
    file_exist = summary_path.is_file()

    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        if not file_exist:
            writer.writeheader()
        writer.writerow(data_dict)


if __name__ == "__main__":
    pass

    sub_path = "z_multilayer_111-888"

    result_log_names_0 = [
        # "slim_exp_3_ret_20260523_092204_0",
        # "slim_exp_3_ret_20260523_092539_0",
        # "slim_exp_3_ret_20260523_093741_0",
        # "slim_exp_3_ret_20260523_092256_0_filter",
        # "slim_exp_3_ret_20260523_092635_0_filter",
        # "slim_exp_3_ret_20260523_093832_0_filter",
    ]

    result_log_names_1 = [
        # "slim_exp_3_ret_20260523_094642_1",
        # "slim_exp_3_ret_20260523_094954_1",
        # "slim_exp_3_ret_20260523_095316_1",
        # "slim_exp_3_ret_20260523_094739_1_filter",
        # "slim_exp_3_ret_20260523_095048_1_filter",
        # "slim_exp_3_ret_20260523_095410_1_filter",
    ]

    result_log_names_2 = [
        # "slim_exp_3_ret_20260523_101024_2",
        # "slim_exp_3_ret_20260523_105809_2",
        # "slim_exp_3_ret_20260523_110240_2",
        # "slim_exp_3_ret_20260523_101138_2_filter",
        # "slim_exp_3_ret_20260523_105926_2_filter",
        # "slim_exp_3_ret_20260523_110355_2_filter",
    ]

    result_log_names_3 = [
        # "slim_exp_3_ret_20260523_111132_3",
        # "slim_exp_3_ret_20260523_111818_3",
        # "slim_exp_3_ret_20260523_112218_3",
        "slim_exp_3_ret_20260523_111233_3_filter",
        "slim_exp_3_ret_20260523_111922_3_filter",
        "slim_exp_3_ret_20260523_112320_3_filter",
    ]

    result_log_names_4 = [
        # "slim_exp_3_ret_20260523_114154_4",
        # "slim_exp_3_ret_20260523_120331_4",
        # "slim_exp_3_ret_20260523_121055_4",
        # "slim_exp_3_ret_20260523_114540_4_filter",
        # "slim_exp_3_ret_20260523_120715_4_filter",
        # "slim_exp_3_ret_20260523_121730_4_filter",
    ]

    result_log_names_5 = [
        # "slim_exp_3_ret_20260523_144052_5",
        # "slim_exp_3_ret_20260523_145853_5",
        # "slim_exp_3_ret_20260523_150332_5",
        # "slim_exp_3_ret_20260523_144216_5_filter",
        # "slim_exp_3_ret_20260523_150017_5_filter",
        # "slim_exp_3_ret_20260523_150454_5_filter",
    ]

    for ret_log_name in result_log_names_5:
        eval_log_name = evaluate_log(f"{sub_path}/{ret_log_name}", 5, "Y")
        print(eval_log_name)
