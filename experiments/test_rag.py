"""
The RAG unit test
"""

from ollama import ChatResponse, Message, chat

import sys_config


def generate_system_prompt() -> str:
    system_prompt = """
    You are a question-answering assistant. Answer the user's question using
    only the information in the context below.
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def generate_user_prompt(question: str, context: str) -> str:
    user_prompt = f"""
    How to answer:
    - The answer may be stated directly, or it may appear as a list,
      in parentheses, or phrased differently from the question.
      Extract the answer as long as the information is present in the context.
    - Treat synonyms and rephrasing as valid. For example, if the question
      asks about "core principles" and the context describes "key techniques"
      or "main components" covering the same idea, use that information.
    - Only reply "I don't know." if the context genuinely does not contain
      the information needed.
    - Do not add facts that are not supported by the context.

    Format:
    - Keep the answer concise (1-2 sentences, or a short list if the question
      asks for multiple items).
    - Use natural language; rephrase rather than copying verbatim.
    - Do not explain your reasoning or add commentary.

    Context:
    {context}

    Question:
    {question}
    """
    return user_prompt


question = ""


context = """"""


if __name__ == "__main__":
    # role: system, user, assistant, tool
    system_msg = Message(
        role="system",
        content=generate_system_prompt(),
    )
    user_msg = Message(
        role="user",
        content=generate_user_prompt(question, context),
    )

    # do not try to use tools on small LLMs
    res: ChatResponse = chat(
        model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
        messages=[user_msg],
    )
    print(res.message.content)
