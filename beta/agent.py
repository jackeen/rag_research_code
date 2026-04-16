from typing import List, Tuple, TypedDict

import dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

import sys_config

from .config import (
    AgentConfig,
    AgentConfigChoseModel,
    CustomerMetadata,
    RetrieveType,
    VectorNames,
)

# from langchain_core.globals import set_debug
# set_debug(True)


class AgentState(TypedDict):
    query: str
    query_keywords: List[str]
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

    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_tpl),
            HumanMessagePromptTemplate.from_template(human_prompt_tpl),
        ]
    )
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

    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_tpl),
            HumanMessagePromptTemplate.from_template(human_prompt_tpl),
        ]
    )
    return prompt


def prompt_generate_without_llm_knowledge() -> ChatPromptTemplate:
    """
    Generate the prompt based on the retrieved documents to generate answer.

    Argus:
        None
    Return:
        ChatPromptTemplate
    """

    # this is work well on chatGPT 4.1 mini
    # system_prompt_tpl = """
    #     You are a strict Retrieval-Augmented Generation (RAG) system.

    #     You MUST follow these constraints:
    #     * Use ONLY the information provided in the [Context]
    #     * DO NOT use any external knowledge or prior training data
    #     * DO NOT infer, assume, or expand beyond what is explicitly stated
    #     * If the answer cannot be directly derived from the Context, respond with: I don't know
    #     * Keep the answer concise and do not add explanations

    #     [Contexts]
    #     {context}
    # """

    system_prompt_tpl_8b = """
    # You are a strict retrieval-based question answering system.

    ## Follow these rules EXACTLY:
    1. Answer ONLY using the provided context.
    2. The answer MUST be directly supported by the context.
    3. You may rephrase the context into natural and fluent language.
    4. Keep the answer concise and clear.
    5. DO NOT add any new information.
    6. DO NOT use any external knowledge.
    7. DO NOT guess or infer beyond the context.
    8. If the answer is not clearly supported by the context, reply EXACTLY: "I don't know."
    9. If the question is unrelated to the context, reply EXACTLY: "I don't know."

    ## Output rules:
    - Use simple and natural language.
    - Keep the answer short (1-2 sentences if possible).
    - **Do not copy the context word-for-word unless necessary**.
    - **Do not explain your reasoning**.
    - **Do not add extra commentary**.

    ## Examples:
    Context: "Paris is the capital of France."
    Question: "What is the capital of France?"
    Answer: "Paris"

    Context: "Paris is the capital of France."
    Question: "Why is Paris the capital of France?"
    Answer: "I don't know."

    Context: "The company was founded in 2010."
    Question: "When was the company founded?"
    Answer: "2010"

    ## Context:
    {context}
    """

    # for granite4:3b
    system_prompt_tpl_3b = """
    You are a strict retrieval-based QA system.

    You MUST follow these rules:
    - Answer ONLY using the provided context.
    - If the answer is NOT explicitly stated in the context, only reply EXACTLY: "I don't know."
    - DO NOT use any external knowledge.
    - DO NOT guess or infer.
    - DO NOT generate unsupported information.
    - If there is ANY uncertainty, only reply EXACTLY: "I don't know."

    Output rules:
    - Use simple and natural language.
    - Keep the answer short (1-2 sentences if possible).
    - Do not copy the context word-for-word unless necessary.
    - Do not explain your reasoning.
    - Do not add extra commentary.

    Context:
    {context}
    """

    human_prompt_tpl = """
        Question:
        {question}
    """

    prompt = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(system_prompt_tpl_3b),
            HumanMessagePromptTemplate.from_template(human_prompt_tpl),
        ]
    )
    return prompt


class Agent:
    """
    The agent class provides methods to interact with the Local RAG system
    powered by Qdrant database and LLMs.

    Init:
    """

    config: AgentConfig
    qdrant_client: QdrantClient
    llm: BaseChatModel
    embeddings: Embeddings
    sparse_embeddings: FastEmbedSparse
    vector_store: QdrantVectorStore
    retriever: BaseRetriever
    docs_count: int

    # It is the agent compiled by LangGraph
    graph_agent: CompiledStateGraph

    def __init__(self, config: AgentConfig):
        """
        Init the agent by given collection name of the vector database.
        :param config: the config of the agent shared with ingestor
        """
        self.config = config

        # This is for hybrid search
        sparse_embeddings = FastEmbedSparse(model=self.config.sparse_embedding_model)
        self.sparse_embeddings = sparse_embeddings

        # To connect the qdrant for langchain store,
        # the docs count for possible valuation
        qdrant_client = QdrantClient(
            host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT
        )
        self.docs_count = qdrant_client.count(
            collection_name=config.collection_name, exact=True
        ).count
        self.qdrant_client = qdrant_client

    def use_openai_llm(self):
        dotenv.load_dotenv()
        self.llm = ChatOpenAI(
            model=sys_config.OPEN_AI_MODEL,
            temperature=0,
        )

    def use_openai_embeddings(self):
        dotenv.load_dotenv()
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions,
        )

    def use_ollama_llm(self):
        self.llm = ChatOllama(
            base_url=sys_config.OLLAMA_URL_BASE,
            model=self.config.llm_model,
            temperature=0,
            reasoning=False,
        )

    def use_ollama_embeddings(self):
        self.embeddings = OllamaEmbeddings(
            base_url=sys_config.OLLAMA_URL_BASE,
            model=self.config.embedding_model,
        )

    def use_transformer_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
        )

    def generate_work_flow(self):
        # this part as same as the config of ingestion
        # the hybrid search will cause RRF(Reciprocal Rank Fusion)
        if self.config.is_hybrid_search:
            self.vector_store = QdrantVectorStore(
                retrieval_mode=RetrievalMode.HYBRID,
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                vector_name=VectorNames.DENSE.value,
                sparse_vector_name=VectorNames.SPARSE.value,
            )
        else:
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings,
            )

        # the default settings of retriever
        # the default configure not includes similarity, just return top 4 docs
        retriever = self.vector_store.as_retriever(
            # search_type=RetrieveType.SIMILARITY.value,
            # search_kwargs={"k": 4, "score_threshold": 0.0},
            # MMR: Maximal Marginal Relevance
            # If the chunks include same information, using it for more diversity
            # search_type=RetrieveType.MMR.value,
            # search_kwargs={
            #     "k": 4,
            #     "fetch_k": 20,
            #     "lambda_mult": 0.5
            # }
        )
        self.retriever = retriever

        workflow = StateGraph(AgentState)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("generator", END)
        workflow.set_entry_point("retriever")
        self.graph_agent = workflow.compile()

    def generator_node(self, state: AgentState) -> AgentState:
        # the two version of prompt
        # prompt = prompt_generate()
        prompt = prompt_generate_v2()

        if self.config.is_not_use_llm_knowledge:
            prompt = prompt_generate_without_llm_knowledge()

        formatted_prompt = prompt.format_prompt(
            context=state["context"],
            question=state["query"],
        ).to_messages()
        # prompt_text = "\n".join([msg.content for msg in formatted_prompt])
        # print("Formatted Prompt:\n", prompt_text)
        response = self.llm.invoke(formatted_prompt)
        state["response"] = str(response.content)
        return state

    def retriever_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        query_keywords = state["query_keywords"]

        # retrieved = self.retriever.invoke(query)

        # use this for debugging by trace scores
        keywords_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.keywords",
                    match=MatchAny(any=query_keywords),
                )
            ]
        )
        if len(query_keywords) == 0:
            keywords_filter = None

        retrieved = self.vector_store.similarity_search_with_score(
            query=query, k=self.config.retrieve_top_k, filter=keywords_filter
        )

        # print("---------------------------------------------------------")
        # print(query)
        # print(f"get query kws: {query_keywords}")
        # print(f"get: {len(retrieved)} chunks")

        # for llm to generate
        retrieved_contexts = []

        # for tracing docs, these will be listed in logs
        retrieved_contents = []
        retrieved_content_reference = []

        # collect info
        for doc, score in retrieved:
            keywords = doc.metadata.get("keywords")
            features = doc.metadata.get("features")
            KFS = doc.metadata.get("KFS")

            retrieved_contexts.append(doc.page_content)
            retrieved_contents.append(
                f"{doc.page_content}\nQCS:{score}\nKFS:{KFS}\nK:{keywords}\nF:{features}"
            )
            retrieved_content_reference.append(
                doc.metadata.get(CustomerMetadata.CHUNK_REFERENCE_NAME.value)
            )
        state["context"] = retrieved_contexts
        state["retrieved_contents"] = retrieved_contents
        state["retrieved_contents_reference"] = retrieved_content_reference

        # for test
        # ret = self.vector_store.similarity_search_with_score(query=query)
        # scores = [score for doc, score in ret]
        # print(query)
        # print(f"retrieve scores: {scores}")

        return state

    def invoke(self, query: str, query_keywords: list[str] = []) -> str:
        ret = self.graph_agent.invoke(
            {
                "query": query,
                "query_keywords": query_keywords,
                "context": "",
                "response": "",
                "retrieved_contents": [],
                "retrieved_contents_reference": [],
            }
        )
        return ret["response"]

    def invoke_with_retrieved_contents(
        self, query: str, query_keywords: list[str] = []
    ) -> Tuple[str, list[str], list[str]]:
        ret = self.graph_agent.invoke(
            {
                "query": query,
                "query_keywords": query_keywords,
                "context": "",
                "response": "",
                "retrieved_contents": [],
                "retrieved_contents_reference": [],
            }
        )
        return (
            ret["response"],
            ret["retrieved_contents"],
            ret["retrieved_contents_reference"],
        )


if __name__ == "__main__":
    pass
