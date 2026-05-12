import dotenv
import pandas as pd
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

import sys_config
from tools.data_loader import get_csv_data_path


def prompt_generate() -> ChatPromptTemplate:
    system_prompt_tpl = """
    Task: Generate a "shallow distractor passage" for evaluating QA system
    robustness. The passage must mention keywords from the question but contain
    no substantive information that could help answer it.

    Requirements:

    (1) Keyword Surface Match: The passage must explicitly mention 2-3 key
        terms or entities from the question (e.g., proper nouns, topic words).
        This makes it superficially appear relevant to a keyword-based retriever.

    (2) Semantic Emptiness: The passage must NOT provide any meaningful,
        specific, or actionable information about the topic. It should consist
        of generic statements, vague descriptions, filler phrases, or tangential
        commentary.

    (3) No Answer Content: The passage must NOT contain, imply, or hint at
        the correct answer in any form.

    (4) Natural Phrasing: The passage should still read as grammatical,
        natural prose (3-4 sentences). It should look like low-quality web
        content — the kind of vague, padded text often found in SEO-optimized
        pages or generic introductions.

    Techniques to use:
    - Generic adjectives without specifics ("important", "interesting",
      "well-known", "significant")
    - Vague historical/contextual framing ("over the years", "many people
      have discussed", "throughout history")
    - Tautological statements ("X is a topic that relates to X-related matters")
    - Mentioning the entity without saying anything specific about it
    - Empty meta-commentary ("This subject has attracted much attention")

    Avoid:
    - Any concrete facts, dates, numbers, names beyond the keywords themselves
    - Any information that could lead a reader toward the correct answer
    - Obviously ungrammatical or nonsense text (it must look plausibly written)

    Examples:

    Question: Who directed the film "Inception"?
    Correct Answer: Christopher Nolan
    Shallow Distractor: The film "Inception" is widely recognized as an
    important work in modern cinema. Many viewers and critics have shared
    their thoughts about "Inception" over the years, contributing to ongoing
    discussions in film communities. The movie continues to be referenced
    in various contexts, reflecting its place in popular culture. It remains
    a frequently mentioned title when people talk about notable films.

    Question: In what year was the Eiffel Tower completed?
    Correct Answer: 1889
    Shallow Distractor: The Eiffel Tower is one of the most well-known
    landmarks associated with Paris. Throughout history, the Eiffel Tower
    has attracted significant attention from visitors and observers alike.
    It is often mentioned in discussions about famous structures, and many
    people have offered their perspectives on its significance. The tower
    remains a frequently referenced subject in various contexts.

    Question: What is the capital of Australia?
    Correct Answer: Canberra
    Shallow Distractor: Australia is a country that has been the subject
    of considerable interest over time. When people discuss Australia,
    various aspects of the country often come up in conversation. The
    topic of Australia's geography and political structure has been
    mentioned in numerous contexts. Many discussions about Australia
    touch on different elements of its identity.

    Now generate a shallow distractor passage for the following:
    """

    human_prompt_tpl = """
    Question: {question}
    Correct Answer: {answer}
    Supportive Noise Passage:
    """

    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_tpl),
            HumanMessagePromptTemplate.from_template(human_prompt_tpl),
        ]
    )
    return prompt


def open_ai_invoke_for_noise(question: str, answer: str):
    dotenv.load_dotenv()
    llm = ChatOpenAI(
        model=sys_config.OPEN_AI_MODEL,
        temperature=0.7,
    )
    prompt = prompt_generate()
    formatted_prompt = prompt.format_prompt(
        question=question, answer=answer
    ).to_messages()

    response = llm.invoke(formatted_prompt)
    return response.content


def make_noise():
    qas_path = get_csv_data_path("questions_and_answers")
    df = pd.read_csv(qas_path)
    q_list = df["questions"].ffill().to_list()
    a_list = df["standard_answers"].ffill().to_list()

    noise_list = []
    for i, q in enumerate(q_list):
        noise_list.append(open_ai_invoke_for_noise(q, a_list[i]))

    noise_df = pd.DataFrame(
        {
            "questions": q_list,
            "standard_answers": a_list,
            "noises": noise_list,
        }
    )

    noise_path = get_csv_data_path("noises")
    noise_df.to_csv(noise_path, index=True, encoding="utf-8")


if __name__ == "__main__":
    # q = """What does LAMP stand for?"""
    # a = """
    # LAMP, standing for Linux, Apache, MySQL, and PHP, named in the same order.
    # """
    # print(open_ai_invoke_for_noise(q, a))
    make_noise()
