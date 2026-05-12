from ollama import ChatResponse, Message, chat

import sys_config


def _generate_system_prompt() -> str:
    system_prompt = """
    You are a text relevance checker.
    Your job is to decide whether a given TEXT contains information that supports a given TOPIC, even partially.

    ## Rules
    1. Answer "Yes" if the TEXT mentions the key points of the TOPIC, even if:
       - Only some (not all) of the key points are mentioned.
       - The wording is different but the meaning is the same (synonyms or paraphrases count).
       - The information is brief or only one sentence.
    2. Answer "No" only if the TEXT does not mention any of the specific key points in the TOPIC, and only talks about the subject in a general or unrelated way.
    3. Focus on meaning, not exact words. For example, "flexible grid layout" and "flexible layouts" mean the same thing.

    ## Output Format
    Answer in exactly this format:
    Decision: Yes / No
    Reason: <one short sentence explaining which key points are or are not covered>

    ## Examples

    Example 1:
    TOPIC: The three primary colors are red, blue, and yellow.
    TEXT: Colors play an important role in art. Artists mix different colors to create new ones.
    Decision: No
    Reason: The text talks about colors generally but does not mention red, blue, or yellow.

    Example 2:
    TOPIC: The three tenets of responsive web design are media queries, flexible layouts, and flexible media.
    TEXT: Ethan Marcotte consolidated three existing techniques — flexible grid layout, flexible images, and media queries — into one unified approach called responsive web design.
    Decision: Yes
    Reason: The text mentions all three tenets using equivalent terms (flexible grid layout = flexible layouts, flexible images = flexible media, media queries = media queries).

    Example 3:
    TOPIC: Cats are mammals that purr.
    TEXT: Cats are small animals that many people keep as pets.
    Decision: No
    Reason: The text mentions cats but does not say they are mammals or that they purr.

    Example 4:
    TOPIC: Python supports object-oriented programming, functional programming, and procedural programming.
    TEXT: Python is a versatile language that supports multiple paradigms, including object-oriented and functional styles.
    Decision: Yes
    Reason: The text supports the topic partially by mentioning object-oriented and functional programming.
    """
    return system_prompt


def _generate_system_prompt_2() -> str:
    system_prompt = """
    # Does the following text discuss the given topic?

    ## Output Format
    Answer in exactly this format:
    Decision: Yes / No
    Reason: <one short sentence explaining which key points are or are not covered>
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def _generate_user_prompt(text: str, topic: str) -> str:
    user_prompt = f"""

    ## Topic
    {topic}

    ## Text
    {text}
    """

    return user_prompt


def is_usefull_information(text: str, topic: str):
    system_msg = Message(
        role="system",
        content=_generate_system_prompt(),
    )
    user_msg = Message(
        role="user",
        content=_generate_user_prompt(text, topic),
    )

    res: ChatResponse = chat(
        model=sys_config.OLLAMA_GEMMA_MODEL_4_E2B,
        messages=[system_msg, user_msg],
    )

    res_content = res.message.content

    if res_content is not None:
        res_arr = res_content.split("\n")
        if res_arr[0].lower().find("yes") > 0:
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    topic = """
    Three tenets of responsive web design are media queries, flexible layouts, and flexible media
    """

    text = """
    Responsive web design is an approach aimed at creating websites that provide an optimal viewing experience across a wide range of devices. This methodology emphasizes adaptability, ensuring that content adjusts smoothly to different screen sizes and orientations. Techniques often involve the use of CSS and HTML to enhance usability and accessibility. Over time, responsive design has become a standard practice in web development, influencing how designers and developers think about layout and user interaction.
    """

    text_2 = """
    The term **responsive web design** was coined by Ethan Marcotte. In his seminal List Apart article (<a href="http://www.alistapart.com/articles/responsive-web-design/">http://www.alistapart.com/articles/responsive-web-design/</a>) he consolidated three existing techniques (flexible grid layout, flexible images, and media and media queries) into one unified approach and named it responsive web design. The term is often used to infer the same meaning as a number of other descriptions such as fluid design, elastic layout, rubber layout, liquid design, adaptive layout, cross-device design, and flexible design.
    To name just a few! However, as Mr. Marcotte and others have eloquently argued, a truly responsive methodology is actually more than merely altering the layout of a site based upon viewport sizes. Instead, it is to invert our entire current approach to web design. Instead of beginning with a fixed width desktop site design and scaling it down and re-flowing the content for smaller viewports, we should design for the smallest viewport first and then progressively enhance the design and content for larger viewports.
    """

    text_3 = """
    Responsive web design is an approach aimed at creating websites that provide an optimal viewing experience across a wide range of devices. This methodology emphasizes adaptability, ensuring that content adjusts smoothly to different screen sizes and orientations. Developers often focus on techniques that enhance usability and accessibility, allowing users to interact with web pages effectively whether on desktops, tablets, or smartphones. The evolution of web standards and browser capabilities has played a significant role in enabling these adaptive design strategies.
    """

    print(is_usefull_information(text_3, topic))
