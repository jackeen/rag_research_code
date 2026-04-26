"""
The anchor test

https://github.com/ollama/ollama-python
"""

from ollama import ChatResponse, Message, chat

import sys_config


def generate_system_prompt() -> str:
    system_prompt = """
    You are a content classifier for a technical book indexing system.
    Your job is to determine whether a text chunk contains substantive information or not.
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def generate_user_prompt(text: str) -> str:
    user_prompt = f"""
    A chunk has SUBSTANTIVE information if it delivers concrete knowledge that a reader would lose if the chunk were deleted. Specifically, it must contain at least one of the following:
    1. A definition or explanation of how something works
    2. Code, commands, configuration, syntax, or API usage
    3. Specific parameters, constraints, return values, or behavioral details
    4. A comparison or trade-off with concrete criteria ("use X when..., use Y when...")
    5. A warning, pitfall, common error, or best practice with specific detail
    6. A step-by-step procedure or actionable instruction
    7. Architecture, data flow, or design reasoning with specific components named

    A chunk has NO SUBSTANTIVE information if it only does one or more of the following without delivering any concrete knowledge:
    1. Summarizes or restates what was covered ("In this chapter we learned...")
    2. Previews what will be covered ("In the next chapter we will...")
    3. References or names technical terms without explaining, defining, or demonstrating them
    4. Uses rhetorical questions or motivational framing ("Imagine you are...", "Have you ever wondered...")
    5. Provides commentary or opinion without technical detail ("This is a very powerful feature", "You'll find this surprisingly easy")
    6. Connects sections with transitional language only ("As we discussed earlier...", "Let's now turn to...")
    7. Gives an oversimplified analogy that is immediately replaced by a proper explanation elsewhere

    Key distinction: mentioning a concept is not the same as explaining it. "We learned what media queries are and how to use them" mentions the topic. "A media query consists of a media type and one or more expressions that check for conditions of particular media features" explains it.

    Respond with a single JSON object, reason first:
    {{"reason": "one sentence explaining your judgment", "label": "SUBSTANTIVE" or "NO_SUBSTANTIVE"}}

    Chunk to classify:
    {text}
    """

    return user_prompt


intro_chunk = """
"""

info_chunk = """
"""


segments = [intro_chunk, info_chunk]

if __name__ == "__main__":
    for seg in segments:
        # role: system, user, assistant, tool
        system_msg = Message(
            role="system",
            content=generate_system_prompt(),
        )
        user_msg = Message(
            role="user",
            content=generate_user_prompt(seg),
        )

        # do not try to use tools on small LLMs
        res: ChatResponse = chat(
            model=sys_config.OLLAMA_GRANITE_MODEL_3_3_8B,
            messages=[system_msg, user_msg],
        )
        print(res.message.content)
