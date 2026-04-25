"""
This module is working for squeezing the code block into as an inline tag.
It depends on LLM and replacing of string.
"""

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from ollama import ChatResponse, Message, chat

"""
Summarize what this code does in one concise English sentence starting with a verb.
Start with the programming language in square brackets, e.g. [CSS], [Python], [JavaScript].
Focus on functionality, not explanation. Do not use backticks or special formatting in the output.
"""


@dataclass
class SummarizedCode(DataClassJsonMixin):
    summary: str = ""
    lang: str = ""


class CodeTag:
    _llm_name: str = ""

    def __init__(self, llm_name) -> None:
        self._llm_name = llm_name

    def _generate_system_prompt(self) -> str:
        system_prompt = """
        You are a software engneer.
        """
        return system_prompt

    def _generate_user_prompt(self, code_block: str) -> str:
        user_prompt = f"""
        ## Task
        - Summarize what this code does in one concise English sentence starting with a verb
        - Detecte the programming language
        - Output the result by following the given JSON example

        ## Output Example
        {{
            "summary": "",
            "lang": ""
        }}

        Code block:
        {code_block}
        """

        return user_prompt

    def get_code_tag(self, code_block: str) -> SummarizedCode | None:

        system_msg = Message(
            role="system",
            content=self._generate_system_prompt(),
        )
        user_msg = Message(
            role="user",
            content=self._generate_user_prompt(code_block),
        )

        res: ChatResponse = chat(
            model=self._llm_name,
            messages=[system_msg, user_msg],
        )

        tag = res.message.content
        if tag is None:
            return None
        else:
            s = SummarizedCode()
            return s.from_json(tag)


if __name__ == "__main__":
    # gemma4:e2b
    # granite4:3b-h
    code_tag = CodeTag("granite4:3b-h")
    code = """
    ```
    body {
        font-color: dark-grey;
    }
    ```
    """
    print(code_tag.get_code_tag(code))
