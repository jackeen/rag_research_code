"""Not used"""

from ollama import ChatResponse, Message, chat

import sys_config


def _generate_system_prompt() -> str:
    system_prompt = """
    You are a paragraph classifier.
    Your task is to determine whether a given paragraph is a "transitional paragraph" or "not transitional paragraph".
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def _generate_user_prompt(text: str) -> str:
    user_prompt = f"""
    ## Rules
    A transitional paragraph typically:
    - Introduces a topic without making a claim (e.g., "In this section, we will discuss...")
    - Provides context without analysis or evidence

    ## Paragraph
    {text}

    ## Output label
    "transitional paragraph" or "not transitional paragraph"
    """

    return user_prompt


def is_not_introduction(text: str):
    system_msg = Message(
        role="system",
        content=_generate_system_prompt(),
    )
    user_msg = Message(
        role="user",
        content=_generate_user_prompt(text),
    )

    res: ChatResponse = chat(
        model=sys_config.OLLAMA_GEMMA_MODEL_4_E2B,
        messages=[system_msg, user_msg],
    )

    res_content = res.message.content
    if res_content is not None and res_content == "not transitional paragraph":
        return True
    else:
        return False
