"""
LLM-as-a-Judge for Topic-based Chunk Filtering
"""

from ollama import ChatResponse, Message, chat

import sys_config


def generate_system_prompt() -> str:
    system_prompt = """
    You are a precise classifier for technical documentation retrieval.
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def generate_user_prompt(claim: str, paragraph: str) -> str:
    user_prompt = f"""
    ## Question
    Determine whether the Paragraph can explain the Claim.

    ## Definition of "supports"
    - The paragraph mentions, describes, or implies the key facts in the claim
    - Information appearing anywhere in the paragraph counts, even if it's not the main point

    Paragraph:
    {paragraph}

    Claim:
    {claim}

    ## Answer
    YES or NO only, without other information, such as explainations.
    """

    return user_prompt


claim = ""

paragraphs = """
"""


if __name__ == "__main__":
    paragraph_list = paragraphs.split("\n\n")

    for p in paragraph_list:
        system_msg = Message(
            role="system",
            content=generate_system_prompt(),
        )
        user_msg = Message(
            role="user",
            content=generate_user_prompt(claim, p),
        )

        # do not try to use tools on small LLMs
        res: ChatResponse = chat(
            model=sys_config.OLLAMA_GRANITE_MODEL_3_3_8B,
            messages=[system_msg, user_msg],
        )
        print(res.message.content)
