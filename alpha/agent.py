from langchain_qdrant import QdrantVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, TypedDict
from qdrant_client import QdrantClient
import sys_config

from langchain_core.globals import set_debug
set_debug(True)


#
class AgentState(TypedDict):
    query: str
    context: List[str]
    response: str


# init resources
# embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_384)
embeddings = OllamaEmbeddings(
    model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    base_url=sys_config.OLLAMA_URL_BASE
)
qdrant_client = QdrantClient(host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=sys_config.QDRANT_COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}, score_threshold=0.8)


def retriever_node(state: AgentState) -> AgentState:
    query = state["query"]
    retrieved_chunks = retriever.invoke(query)
    state["context"] = [doc.page_content for doc in retrieved_chunks]
    return state


llm = ChatOllama(
    base_url=sys_config.OLLAMA_URL_BASE,
    model=sys_config.OLLAMA_GRANITE_MODEL_3_3_8B,
    validate_model_on_init=True,
    reasoning=False,
    temperature=0,
)


def prompt_generate():
    system_prompt_tpl = """
    You are an assistant who flow the truth of the retrieved documents.
    Use the following context to answer the question, if the context is not contain information,
    you should provide the general answer based on your knowledge but with the sign [From the LLM].
    
    Context: {context}
    """
    human_prompt_tpl = """
    Question: {question}
    """
    prompt = ChatPromptTemplate([
        SystemMessagePromptTemplate.from_template(system_prompt_tpl),
        HumanMessagePromptTemplate.from_template(human_prompt_tpl),
    ])
    return prompt


def generator_node(state: AgentState) -> AgentState:
    prompt = prompt_generate()
    formatted_prompt = prompt.format_prompt(
        context=state["context"],
        question=state["query"],
    ).to_messages()
    # prompt_text = "\n".join([msg.content for msg in formatted_prompt])
    # print("Formatted Prompt:\n", prompt_text)
    response = llm.invoke(formatted_prompt)
    state["response"] = response.content
    return state


workflow = StateGraph(AgentState)
workflow.add_node("retriever", retriever_node)
workflow.add_node("generator", generator_node)
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)
workflow.set_entry_point("retriever")
agent = workflow.compile()


def invoke_agent(query: str) -> str:
    result = agent.invoke({
        "query": query,
        "context": "",
        "response": "",
    })
    return result["response"]


# model testing
if __name__ == '__main__':
    result = agent.invoke({
        "query": "hi",
        "context": "",
        "response": "",
    })
    print(result)










