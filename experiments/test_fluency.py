"""
The sentence fluency experiment.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import sys_config

llm = ChatOllama(
    base_url=sys_config.OLLAMA_URL_BASE,
    model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
    temperature=0,
)


def fluency_score(text: str) -> float:
    prompt = f'''
    You are a language quality evaluator.

    Your task is to assess how fluent and natural a given piece of text is in English.

    Fluency includes:
    - grammatical correctness
    - clarity of expression
    - natural phrasing (native-like)
    - logical flow between sentences

    Instructions:
    - Output a single number between 0 and 1
    - 1 = perfectly fluent and natural
    - 0 = completely broken or incomprehensible
    - Do NOT provide explanations or any extra text
    - Only return the score

    Text to evaluate:

    "{text}"
    '''

    chain = llm | StrOutputParser()

    result = chain.invoke(input=prompt)
    return float(result)


if __name__ == "__main__":
    content = """
    """
    content2 = """
    """

    content3 = """"""

    print(fluency_score(content), fluency_score(content2), fluency_score(content3))
