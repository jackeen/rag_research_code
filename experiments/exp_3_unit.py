import dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
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

    prompt = ChatPromptTemplate([
        SystemMessagePromptTemplate.from_template(system_prompt_tpl_not_use_llm),
        HumanMessagePromptTemplate.from_template(human_prompt_tpl),
    ])
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


if __name__ == '__main__':
    question = """
    What are the three fundamental tenets that constitute the core of responsive web design?
    """

    std_answer = """
    Three tenets of responsive web design are media queries, flexible layouts, and flexible media
    """

    context = """
    Examples of responsive web design
    Get your viewport testing tools here!
    Online sources of inspiration
    HTML5—why it's so good
    Saving time and code with HTML5
    New, semantically meaningful HTML5 tag elements
    CSS3 enables responsive designs and more
    The bottom line—CSS3 won't break anything!
    How can CSS3 solve everyday design problems?
    Look Ma'—no images!
    What else has CSS3 got to offer?
    Can HTML5 and CSS3 work for us today?

    Defining responsive web design
    The term responsive web design was coined by Ethan Marcotte. In his seminal List
    Apart article (http://www.alistapart.com/articles/responsive-web-design/) he
    consolidated three existing techniques (flexible grid layout, flexible images, and media
    and media queries) into one unified approach and named it responsive web design.

    The lynch pin in making a fully responsive web design is CSS3. Before we use CSS3
    to add visual flair such as the gradients, rounded corners, text shadows, animations
    and transforms to our design, we will first use it to serve a more fundamental role. By
    using CSS3 media queries, we will be able to target specific CSS rules at specific
    viewports. The next chapter is where we will start our "responsive web design" quest
    in earnest.
    Frain, Ben.

    Why stop at responsive design?
    A responsive web design will handle the flow of our page content as viewports
    change but let's go further. HTML5 offers us more than HTML 4 ever could and it's
    more meaningful semantic elements will form the basis of our markup. CSS3 media
    queries are an essential ingredient to a responsive design but additional CSS3
    modules empower us with previously unseen levels of flexibility.
    """

    context_better = """
    Examples of responsive web design
    Get your viewport testing tools here!
    Online sources of inspiration
    HTML5—why it's so good
    Saving time and code with HTML5
    New, semantically meaningful HTML5 tag elements
    CSS3 enables responsive designs and more
    The bottom line—CSS3 won't break anything!
    How can CSS3 solve everyday design problems?
    Look Ma'—no images!
    What else has CSS3 got to offer?
    Can HTML5 and CSS3 work for us today?
    """

    answer = open_ai_invoke_with_context_and_question(question, context_better)

    print(answer)
    print(calculate_cosine_similarity([answer], [std_answer]))
    # 0.84 - 0.87
    # 0.88
