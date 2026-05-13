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

    ## Contexts
    {context}
    """

    human_prompt_tpl = """
    ## Question
    {question}

    ## Answer
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
    In the context of Lisp’s evaluation model, how does recursion enable functions to compute results without using traditional looping constructs?
    """

    std_answer = """
    The evaluation rule is recursive in nature; that is, it includes, as one of its steps, the need to invoke the rule itself. Notice how succinctly the idea of recursion can be used to express what, in the case of a deeply nested combination, would otherwise be viewed as a rather complicated process.
    """

    context = """
    Recursion is a concept that many have explored in various programming languages, including Lisp. Over time, the idea of recursion has been discussed in relation to evaluation and other programming principles. The topic of recursion often comes up when considering how evaluation rules are applied, and it continues to be a subject of interest in programming discussions. Many people have reflected on recursion’s role in different contexts without always reaching a single definitive explanation.
    """

    context_better = """

    One of our goals in this chapter is to isolate issues about thinking procedurally. As a case in point, let us consider that, in evaluating combinations, the interpreter is itself following a procedure.
    To evaluate a combination, do the following:
    - 1. Evaluate the subexpressions of the combination.
    - 2. Apply the procedure that is the value of the lemost subexpression (the operator) to the arguments that are the values of the other subexpressions (the operands).
    Even this simple rule illustrates some important points about processes in general. First, observe that the first step dictates that in order to accomplish the evaluation process for a combination we must first perform the evaluation process on each element of the combination. us, the evaluation rule is *recursive* in nature; that is, it includes, as one of its steps, the need to invoke the rule itself.<sup>10</sup>
    Notice how succinctly the idea of recursion can be used to express what, in the case of a deeply nested combination, would otherwise be viewed as a rather complicated process. For example, evaluating
    <sup>9</sup>Chapter 3 will show that this notion of environment is crucial, both for understanding how the interpreter works and for implementing interpreters.
    <sup>10</sup>It may seem strange that the evaluation rule says, as part of the first step, that we should evaluate the lemost element of a combination, since at this point that can only be an operator such as + or \\* representing a built-in primitive procedure such as addition or multiplication. We will see later that it is useful to be able to work with combinations whose operators are themselves compound expressions.
    **Figure 1.1:** Tree representation, showing the value of each subcombination.

    requires that the evaluation rule be applied to four different combinations. We can obtain a picture of this process by representing the combination in the form of a tree, as shown in Figure 1.1. Each combination is represented by a node with branches corresponding to the operator and the operands of the combination stemming from it. e terminal nodes (that is, nodes with no branches stemming from them) represent either operators or numbers. Viewing evaluation in terms of the tree, we can imagine that the values of the operands percolate upward, starting from the terminal nodes and then combining at higher and higher levels. In general, we shall see that recursion is a very powerful technique for dealing with hierarchical, treelike objects. In fact, the "percolate values upward" form of the evaluation rule is an example of a general kind of process known as *tree accumulation*.
    Next, observe that the repeated application of the first step brings us to the point where we need to evaluate, not combinations, but primitive expressions such as numerals, built-in operators, or other names. We take care of the primitive cases by stipulating that
    - the values of numerals are the numbers that they name,
    - the values of built-in operators are the machine instruction sequences that carry out the corresponding operations, and
    - the values of other names are the objects associated with those names in the environment.
    We may regard the second rule as a special case of the third one by stipulating that symbols such as + and \\* are also included in the global environment, and are associated with the sequences of machine instructions that are their "values." e key point to notice is the role of the environment in determining the meaning of the symbols in expressions. In an interactive language such as Lisp, it is meaningless to speak of the value of an expression such as (+ x 1) without specifying any information about the environment that would provide a meaning for the symbol x (or even for the symbol +). As we shall see in Chapter 3, the general notion of the environment as providing a context in which evaluation takes place will play an important role in our understanding of program execution.
    Notice that the evaluation rule given above does not handle definitions. For instance, evaluating (define x 3) does not apply define to two arguments, one of which is the value of the symbol x and the other of which is 3, since the purpose of the define is precisely to associate x with a value. (at is, (define x 3) is not a combination.)
    Such exceptions to the general evaluation rule are called *special forms*. define is the only example of a special form that we have seen so far, but we will meet others shortly. Each special form has its own evaluation rule. e various kinds of expressions (each with its associated
    evaluation rule) constitute the syntax of the programming language. In comparison with most other programming languages, Lisp has a very simple syntax; that is, the evaluation rule for expressions can be described by a simple general rule together with specialized rules for a small number of special forms.<sup>11</sup>
    """

    answer = open_ai_invoke_with_context_and_question(question, context)
    print(answer)

    if answer is not None:
        print(calculate_cosine_similarity([str(answer)], [std_answer]))
    # 0.84 - 0.87
    # 0.88
