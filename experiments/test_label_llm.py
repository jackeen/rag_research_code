"""
The anchor test

https://github.com/ollama/ollama-python
"""

from ollama import ChatResponse, Message, chat

import sys_config


def generate_system_prompt() -> str:
    system_prompt = """
    You are a document analysis assistant working for labeling the document.
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def generate_user_prompt(text: str) -> str:
    user_prompt = f"""
    ## text
    {text}

    ## Option labels
    - table_of_contents
    - index
    - code
    - introduction
    - prose
    - other

    ## Rules
    - If the text contains a list of chapters or sections with page numbers, label it as "table_of_contents"
    - If the text contains list of short line and each line started from "-" char, label it as "index"
    - If the text contains programming code, scripts, or command-line instructions, label it as "code"
    - If the text contains question sentences, summary, label it as "introduction"
    - If the text contains paragraphs, explanations, or detailed content, label it as "prose"
    - If the text does not fit any of the above categories, label it as "other"

    ## Output format
    Output one name of the list of labels, without any other information.
    """

    return user_prompt


python_sample = """
def train_classifier(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model
"""

css_sample = """
.container {
    display: flex;
    justify-content: center;
    padding: 20px;
}

.button:hover {
    background-color: #007bff;
    cursor: pointer;
}
"""


segments = [python_sample, css_sample]

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
            model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
            messages=[system_msg, user_msg],
        )
        print(res.message.content)
