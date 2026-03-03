from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, TypedDict
from qdrant_client import QdrantClient
import dotenv
import config

# from langchain_core.globals import set_debug
# set_debug(True)


dotenv.load_dotenv()


#
class AgentState(TypedDict):
    query: str
    context: List[str]
    response: str


def prompt_generate() -> str:
    """

    :return:
    """
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


def prompt_generate_2() -> str:
    """

    :return:
    """
    system_prompt_tpl = """
    Agent role: 
    You are a friendly and helpful expert about web development knowledge and responsible for answering any questions 
    about web development. 
    
    Answer users with the retrieved information and ensure that the information you provide is rich and accurate 
    and slightly shorter.
    
    Use the following context to answer the question, if the context is not contain associated information,
    you should provide the general answer based on your knowledge and leave the specific sign: [LLM] in the end.
    
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


class Agent:
    """
    The agent class provides methods to interact with the Local RAG system
    powered by local Qdrant database and OpenAI api.
    """

    def __init__(self, c_name: str):
        """
        Init the agent by given collection name of the vector database.
        :param c_name: collection name of the vector database
        """
        self.c_name = c_name

        # init resources
        embeddings = OpenAIEmbeddings(model=config.OPEN_AI_EMBEDDING_MODEL)
        qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=c_name,
            embedding=embeddings,
        )
        retriever = vector_store.as_retriever(
            # search_kwargs={"k": 3},
            # score_threshold=0.7,
        )
        llm = ChatOpenAI(
            model=config.OPEN_AI_MODEL,
            temperature=0,
        )
        self.agent = None
        self.llm = llm
        self.retriever = retriever

    def generator_node(self, state: AgentState) -> AgentState:
        # the two version of prompt
        # prompt = prompt_generate()
        prompt = prompt_generate_2()
        formatted_prompt = prompt.format_prompt(
            context=state["context"],
            question=state["query"],
        ).to_messages()
        # prompt_text = "\n".join([msg.content for msg in formatted_prompt])
        # print("Formatted Prompt:\n", prompt_text)
        response = self.llm.invoke(formatted_prompt)
        state["response"] = response.content
        return state

    def retriever_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        retrieved_chunks = self.retriever.invoke(query)
        state["context"] = [doc.page_content for doc in retrieved_chunks]
        return state

    def compile_agent(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)
        workflow.set_entry_point("retriever")
        self.agent = workflow.compile()

    def invoke(self, query: str) -> str:
        ret = self.agent.invoke({
            "query": query,
            "context": "",
            "response": "",
        })
        return ret["response"]


# model testing
if __name__ == '__main__':
    agent = Agent('rag_gpt')
    agent.compile_agent()
    # result = agent.invoke('What does LAMP stand for? ')
    result = agent.invoke('What is LLM in AI? ')
    print(result)










