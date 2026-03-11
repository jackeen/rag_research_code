from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, TypedDict
from qdrant_client import QdrantClient
import dotenv
import sys_config
from .config import AgentConfig, CustomerMetadata

# from langchain_core.globals import set_debug
# set_debug(True)


dotenv.load_dotenv()


class AgentState(TypedDict):
    query: str
    context: List[str]
    retrieved_contents: List[str]
    retrieved_contents_reference: List[str]
    response: str


def prompt_generate() -> ChatPromptTemplate:
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


def prompt_generate_v2() -> ChatPromptTemplate:
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
    powered by Qdrant database and LLMs.

    Init:
    """

    def __init__(self, config: AgentConfig):
        """
        Init the agent by given collection name of the vector database.
        :param c_name: collection name of the vector database
        """
        self.config = config

        # init resources
        embeddings = OpenAIEmbeddings(model=sys_config.OPEN_AI_EMBEDDING_MODEL)
        qdrant_client = QdrantClient(host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=config.collection_name,
            embedding=embeddings,
        )

        # the default settings of retrieving
        # the default configure not includes similarity, just return top 4 docs
        retriever = vector_store.as_retriever(
            # search_kwargs={"k": 4, "score_threshold": 0.0},
            # search_type="similarity_score_threshold"
        )
        llm = ChatOpenAI(
            model=sys_config.OPEN_AI_MODEL,
            temperature=0,
        )

        # This is the langgraph compiled agent
        self.agent = None
        self.llm = llm
        self.retriever = retriever
        self.db_client = qdrant_client
        self.docs_count = qdrant_client.count(
            collection_name=config.collection_name,
            exact=True
        )
        self.compile_agent()

    def generator_node(self, state: AgentState) -> AgentState:
        # the two version of prompt
        # prompt = prompt_generate()
        prompt = prompt_generate_v2()
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
        retrieved_docs = self.retriever.invoke(query)
        retrieved_contents = []
        retrieved_content_reference = []
        for doc in retrieved_docs:
            retrieved_contents.append(doc.page_content)
            retrieved_content_reference.append(doc.metadata.get(CustomerMetadata.CHUNK_REFERENCE_NAME.value))
        state["context"] = retrieved_contents
        state["retrieved_contents"] = retrieved_contents
        state["retrieved_contents_reference"] = retrieved_content_reference
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
            "retrieved_contents": [],
            "retrieved_contents_reference": [],
        })
        return ret["response"]

    def invoke_with_retrieved_contents(self, query: str) -> dict:
        ret = self.agent.invoke({
            "query": query,
            "context": "",
            "response": "",
            "retrieved_contents": [],
            "retrieved_contents_reference": [],
        })
        return {
            "response": ret["response"],
            "retrieved_contents": ret["retrieved_contents"],
            "retrieved_contents_reference": ret["retrieved_contents_reference"],
        }


# model testing
if __name__ == '__main__':
    agent_config = AgentConfig(collection_name='test')
    agent = Agent(agent_config)
    # result = agent.invoke('What does LAMP stand for? ')
    result = agent.invoke('What is LLM in AI? ')
    print(result)










