"""
This module depends on 0.3.4 Ragas
"""

from typing import Sequence, cast

import dotenv
import pandas as pd
from datasets import Dataset
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms.base import llm_factory
from ragas.metrics import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    # ResponseRelevancy,
    SemanticSimilarity,
)
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

import sys_config
from tools.data_loader import get_csv_log_path, get_no_tail_csv_log_path

# load env for OpenAI key
dotenv.load_dotenv()


def extract_retrieved_chunks(text: str) -> list[str]:
    chunks = text.split("\n\n")
    raw_chunks: list[str] = []
    for c in chunks:
        raw_chunk = c.split("\nQCS")[0]
        raw_chunks.append(raw_chunk)
    return raw_chunks


def evaluate_log(csv_name: str) -> str:
    """Score the experiment by its log file and return the log file name, which includes score details"""
    log_file_path = get_no_tail_csv_log_path(csv_name)
    df = pd.read_csv(log_file_path)
    questions = df["question"].tolist()
    standard_answers = df["std_answers"].tolist()
    answers = df["answers"].tolist()
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

    openai_client = AsyncOpenAI()
    judge_emb = OpenAIEmbeddings(
        client=openai_client, model=sys_config.OPEN_AI_EMBEDDING_MODEL
    )

    judge_llm = llm_factory(model=sys_config.OPEN_AI_MODEL)

    metrics = cast(
        Sequence[Metric],
        [
            ContextPrecision(llm=judge_llm, name="context_precision"),
            ContextRecall(llm=judge_llm, name="context_recall"),
            ContextEntityRecall(llm=judge_llm, name="context_entity_recall"),
            Faithfulness(llm=judge_llm, name="faithfulness"),
            NoiseSensitivity(
                llm=judge_llm, mode="relevant", name="noise_sensitivity_relevant"
            ),
            NoiseSensitivity(
                llm=judge_llm, mode="irrelevant", name="noise_sensitivity_irrelevant"
            ),
            SemanticSimilarity(embeddings=judge_emb, name="semantic_similarity"),
            # not work
            # ResponseRelevancy(
            #     embeddings=judge_emb, llm=judge_llm, name="response_relevancy"
            # ),
        ],
    )

    score = evaluate(
        dataset=data_set,
        metrics=metrics,
        run_config=RunConfig(max_workers=4),
    )
    score = cast(EvaluationResult, score)

    result_dataframe = score.to_pandas()
    result_path = get_csv_log_path("exp_3_eval")
    result_dataframe.to_csv(result_path, encoding="utf-8")
    return result_path.stem


if __name__ == "__main__":
    # result_log_names = ["exp_3_ret_1_1", "exp_3_ret_3_4"]
    result_log_names = ["exp_3_ret_1_1"]
    for ret_log_name in result_log_names:
        eval_log_name = evaluate_log(ret_log_name)
        print(eval_log_name)
