"""
The anchor filter based on the given topic and candidate chunks to collect sections with specific knowledge.
The LLM power is based on Ollama.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Self, cast

import mistune
import ollama
from ollama import ChatResponse, Client, Message
from qdrant_client import QdrantClient

# from .checker_llm_information import is_usefull_information
# from ingestion.converters.code_block_inline import CodeTag
from .classifier_llm_core_point import is_transitional

# from .cleaner_llm_transitional_part import remove_transitional_part

# from .entailment_filter_cross_deberta import is_paragraph_supports_topic


@dataclass
class Topic:
    content: str = ""


@dataclass
class Segment:
    content: str = ""


class AnchorCollector:
    """
    The chunk collector based on LLM.
    The core of this is leveraging the power of LLM to match page chunks in a collection,
    the invoking LLM's time complexity is m*n.
    """

    _ollama_url: str
    _ollama_model: str
    _ollama_client: Client

    _topics: list[Topic]
    _segments: list[Segment]

    _collected_segements: dict[str, str]

    def __init__(self, ollama_url: str, ollama_model: str) -> None:
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._ollama_client = Client(host=ollama_url)
        self._collected_segements = dict()

    def push_data(self, topics: list[Topic], segments: list[Segment]) -> Self:
        self._topics = topics
        self._segments = segments
        return self

    def _generate_system_prompt(self) -> str:
        system_content = """
        # Does the following text discuss this topic?
        """
        return system_content

    #  in first line. Answer the refined content in the second line
    def _generate_user_prompt(self, topic: str, text: str) -> str:
        user_content = f"""
        ## Topic
        {topic}

        ## Text
        {text}

        ## Answer
        Answer yes or no without any other information.
        """
        return user_content

    def _invoke_llm(self, topic: str, segment: str) -> bool:
        # role: system, user, assistant, tool
        system_msg = Message(
            role="system",
            content=self._generate_system_prompt(),
        )
        user_msg = Message(
            role="user",
            content=self._generate_user_prompt(topic, segment),
        )

        res: ChatResponse = self._ollama_client.chat(
            model=self._ollama_model,
            messages=[system_msg, user_msg],
        )
        res_content = res.message.content
        if res_content is not None and res_content.lower() == "yes":
            return True
        else:
            return False

    def _gethash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def refine_segments_by_topics(self) -> Self:
        """Leverage LLM to refine topics"""
        for topic in self._topics:
            for segment in self._segments:
                if self._invoke_llm(topic.content, segment.content):
                    hash = self._gethash(segment.content)
                    # eliminate duplicates, 10 topics with about 300 section chunks, M*N times invloking LLM
                    self._collected_segements[hash] = segment.content
        return self

    def get_refined_segments_str_list(self, container: list[str]) -> Self:
        for v in self._collected_segements.values():
            container.append(v)
        return self


class ParagraphType(Enum):
    BLANK_LINE = "blank_line"
    THEMATIC_BREAK = "thematic_break"
    PARAGRAPH = "paragraph"
    BLOCK_CODE = "block_code"
    LIST = "list"
    TABLE = "table"


class SemanticModules(Enum):
    LLM = "llm"
    NLI = "nli"


class AnchorSelector:
    """
    The chunk selector based on NLI or LLM, to select and filter from page chunk to paragraph chunk.
    The core of this process is to pick up valuable chunks from the collection of database, so, this module name includs "selector".
    The module name also include "anchor", which means that the pre-setting topics are used as an anchor to compare with chunks.

    In the page searching level, The searching algorithm is HNSW for lowering time complexity, which is Olog(N),
    in practice, Qdrant is used for this process.

    In the paragrah level, ....
    """

    _ollama_url: str
    _ollama_model: str
    _ollama_client: Client

    _qdrant_host: str
    _qdrant_port: int
    _qdrant_client: QdrantClient
    _dense_embedding_model: str

    _all_topic_page_chunks: dict[str, list[str]] = {}

    _page_chunks: list[str] = []

    _paragraph_chunks: list[str] = []
    _refined_paragraph_chunks: list[str] = []

    _page_code_chunks: list[str] = []
    _page_list_chunks: list[str] = []

    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        dense_embedding_model: str,
        ollama_url: str,
        ollama_model: str,
    ) -> None:
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._dense_embedding_model = dense_embedding_model
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._ollama_client = Client(host=ollama_url)

    def _embedding(self, content: str) -> list[float]:
        res = ollama.embeddings(model=self._dense_embedding_model, prompt=content)
        return list(res.embedding)

    def connect_db(self) -> Self:
        client = QdrantClient(host=self._qdrant_host, port=self._qdrant_port)
        self._qdrant_client = client
        return self

    def _get_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _generate_system_prompt(self) -> str:
        system_content = """
        # Does the following text discuss the given topic?
        """
        return system_content

    def _generate_user_prompt(self, topic: str, text: str) -> str:
        user_content = f"""
        ## Topic
        {topic}

        ## Text
        {text}

        ## Answer
        Answer yes or no without any other information.
        """
        return user_content

    # this version of prompt will cause bad results
    # def _generate_system_prompt(self) -> str:
    #     system_content = """
    #     # Task: Determine whether the passage discusses the given topic.

    #     ## Definition of "discusses":
    #     - The passage's main subject or a substantial portion is about the given topic.
    #     - The passage provides factual information, context, or details related to
    #       the topic.

    #     ## Does NOT count as "discusses":
    #     - Brief mention of the topic in passing (e.g., one phrase) without elaboration
    #     - The passage only shares a keyword with the topic but addresses something else
    #     """
    #     return system_content

    # def _generate_user_prompt(self, topic: str, text: str) -> str:
    #     user_content = f"""
    #     ## Topic
    #     {topic}

    #     ## Passage
    #     {text}

    #     ## Answer
    #     Answer yes or no without any other information.
    #     """
    #     return user_content

    def _is_associated(self, topic: str, content: str) -> bool:
        # role: system, user, assistant, tool
        system_msg = Message(
            role="system",
            content=self._generate_system_prompt(),
        )
        user_msg = Message(
            role="user",
            content=self._generate_user_prompt(topic, content),
        )

        # different LLM may cause different result, carefully to chose and test
        res: ChatResponse = self._ollama_client.chat(
            model=self._ollama_model,
            messages=[system_msg, user_msg],
        )
        res_content = res.message.content

        if res_content is not None and res_content.lower() == "yes":
            return True
        else:
            return False

    def refine_page_chunks_from_queried_points(
        self,
        topics: list[Topic],
        collection_name: str,
        size: int = 8,
        threshold: float = 0.5,
    ) -> Self:
        print(f"Anchor Selector: get {len(topics)} topics")

        # collect all data
        topic_page_chunks: dict[str, list[str]] = {}

        # treat all topics by searching the database by the content of topc
        for topic in topics:
            res = self._qdrant_client.query_points(
                collection_name=collection_name,
                query=self._embedding(topic.content),
                limit=size,
                score_threshold=threshold,
            )

            # treat all results which are the page chunk
            current_page_chunks: list[str] = []
            for p in res.points:
                if p.payload:
                    content = p.payload.get("page_content")
                    if content is not None:
                        current_page_chunks.append(content)
                        self._page_chunks.append(content)
                    else:
                        continue
                else:
                    continue

            topic_page_chunks[topic.content] = current_page_chunks
        self._all_topic_page_chunks = topic_page_chunks

        return self

    def refine_paragrahp_chunks(
        self, semantic_module: SemanticModules, is_drop_code: bool = False
    ) -> Self:
        # the default renderer is HTML
        parse = mistune.create_markdown(renderer=None)

        for topic in self._all_topic_page_chunks.keys():
            page_chunks = self._all_topic_page_chunks.get(topic)
            page_chunks = cast(list[str], page_chunks)

            for page in page_chunks:
                ast = cast(list, parse(page))

                # collect children types from page chunk
                code_types = []
                nodes: list[dict] = []
                for node in ast:
                    node = cast(dict, node)
                    node_type = node.get("type", "")
                    code_types.append(node_type)
                    nodes.append(node)

                # pick up code and list page in different group
                if ParagraphType.LIST.value in code_types:
                    is_ok = self._is_associated(topic, page)
                    if is_ok:
                        self._page_list_chunks.append(page)
                    continue
                elif ParagraphType.BLOCK_CODE.value in code_types:
                    if is_drop_code:
                        continue
                    else:
                        is_ok = self._is_associated(topic, page)
                        if is_ok:
                            self._page_code_chunks.append(page)
                        continue

                # refine paragraph chunks in current page
                for node in nodes:
                    if node.get("type") == ParagraphType.PARAGRAPH.value:
                        node_content = self._render_children_content(node["children"])

                        if len(node_content) == 0:
                            continue
                        else:
                            self._paragraph_chunks.append(node_content)

                            is_associated_with_topic = False

                            # NLI method (future work)
                            # not used based on its lower effective and high complexity
                            # but in experiments, it provides some comparative values
                            # if semantic_module == SemanticModules.NLI:
                            #     is_associated_with_topic, _ = (
                            #         is_paragraph_supports_topic(topic, node_content)
                            #     )

                            # LLM method
                            if semantic_module == SemanticModules.LLM:
                                is_associated_with_topic = self._is_associated(
                                    topic, node_content
                                )
                                # is_associated_with_topic = is_usefull_information(
                                #     node_content, topic
                                # )

                            if is_associated_with_topic:
                                if not is_transitional(node_content):
                                    self._refined_paragraph_chunks.append(node_content)

                                # this part need more test, it not suitable for every books
                                # else:
                                #     cleaned_content = remove_transitional_part(
                                #         node_content
                                #     )
                                #     if len(cleaned_content) > 200:
                                #         self._refined_paragraph_chunks.append(
                                #             cleaned_content
                                #         )

        return self

    def _render_children_content(self, children) -> str:
        parts = []
        for c in children:
            ntype = c.get("type", "")
            if "raw" in c:
                parts.append(c["raw"])
            elif "children" in c:
                parts.append(self._render_children_content(c["children"]))
            elif ntype in ("softbreak", "linebreak"):
                # softbreak also can choose \n
                parts.append(" ")
        return "".join(parts)

    def get_selected_page_chunk_str_list(self, container: list[str]) -> Self:
        for c in self._page_chunks:
            container.append(c)
        return self

    def get_hybrid_chunks_str_list(self, container: list[str]) -> Self:
        for c in self._page_code_chunks:
            container.append(c)
        for c in self._page_list_chunks:
            container.append(c)
        for c in self._refined_paragraph_chunks:
            container.append(c)

        container = list(set(container))
        return self


if __name__ == "__main__":
    pass
