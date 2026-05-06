import dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

import sys_config
from tools.similarity import calculate_cosine_similarity


def prompt_generate() -> ChatPromptTemplate:

    # "the answer should be very briefly", this prompt can shorter the answer
    # 1 currently, use the old one
    # 2 future to use the new version

    system_prompt_tpl = """
    Agent role:
    You are a friendly and helpful expert about web development knowledge and responsible for answering any questions
    about web development.

    Answer users with the context information and ensure that the answer you provide is rich and accurate
    and slightly shorter.

    Use the following context to answer the question, if the context is not contain associated information,
    you should provide the general answer based on your knowledge.

    Context: {context}
    """

    system_prompt_tpl_not_use_llm = """
    You are a strict Retrieval-Augmented Generation (RAG) system.

    You MUST follow these constraints:
    * Use ONLY the information provided in the [Context]
    * DO NOT use any external knowledge or prior training data
    * DO NOT infer, assume, or expand beyond what is explicitly stated
    * If the answer cannot be directly derived from the Context, respond with: I don't know
    * Keep the answer concise and do not add explanations

    [Contexts]
    {context}
    """

    human_prompt_tpl = """
        Question: {question}
        """

    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_tpl_not_use_llm),
            HumanMessagePromptTemplate.from_template(human_prompt_tpl),
        ]
    )
    return prompt


def open_ai_invoke_with_context_and_question(query: str, context: str):
    dotenv.load_dotenv()
    llm = ChatOpenAI(
        model=sys_config.OPEN_AI_MODEL,
        temperature=0,
    )
    prompt = prompt_generate()
    formatted_prompt = prompt.format_prompt(
        context=context,
        question=query,
    ).to_messages()

    response = llm.invoke(formatted_prompt)
    return response.content


if __name__ == "__main__":
    question = """
    What are the three fundamental tenets that constitute the core of responsive web design?
    """

    std_answer = """
    Three tenets of responsive web design are media queries, flexible layouts, and flexible media
    """

    context = """
    """

    context_better = """
    """

    answer = open_ai_invoke_with_context_and_question(question, context_better)

    print(answer)
    print(calculate_cosine_similarity([answer], [std_answer]))
    # 0.84 - 0.87
    # 0.88
