"""
The anchor filter based on the given topic and candidate chunks to collect sections with specific knowledge.
The LLM power is based on Ollama.
"""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Self, cast

import mistune
import ollama
from ollama import ChatResponse, Client, Message, chat
from qdrant_client import QdrantClient

from ingestion.converters.code_block_inline import CodeTag

# from .entailment_filter_cross_deberta import is_paragraph_supports_topic


def generate_system_prompt() -> str:
    system_content = """
    # Does the following text discuss this topic?
    """
    return system_content


def generate_user_prompt(topic: str, text: str) -> str:
    user_content = f"""
    ## Topic
    {topic}

    ## Text
    {text}

    ## Answer
    Answer yes or no without any other information.
    """
    return user_content


def is_associated(topic: str, content: str) -> bool:
    # role: system, user, assistant, tool
    system_msg = Message(
        role="system",
        content=generate_system_prompt(),
    )
    user_msg = Message(
        role="user",
        content=generate_user_prompt(topic, content),
    )

    res: ChatResponse = chat(
        model="granite4:3b-h",
        messages=[system_msg, user_msg],
    )
    res_content = res.message.content

    if res_content is not None and res_content.lower() == "yes":
        return True
    else:
        return False


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


class AnchorSelector:
    """
    The chunk selector based on NLI, to select and filter from page chunk to paragraph chunk.
    The core of this process is to pick up valuable chunks from the collection, so, this module name includs "selector".
    The module name also include "anchor", which means that the pre-setting topics are used as an anchor to compare with chunks.

    In the page searching level, The searching algorithm is HNSW for lowering time complexity, which is Olog(N),
    in practice, Qdrant is used for this process.

    In the paragrah level, ....
    """

    _qdrant_host: str
    _qdrant_port: int
    _qdrant_client: QdrantClient
    _dense_embedding_model: str
    _code_tag_llm_model: str

    _code_tag_converter: CodeTag

    # these two level chunks just for understanding easily,
    # it is not mean that is exact kind of chunks.

    _all_topic_page_chunks: dict[str, list[str]] = {}

    _page_chunks: list[str] = []

    _paragraph_chunks: list[str] = []
    _refined_paragraph_chunks: list[str] = []

    _page_code_chunks: list[str] = []
    _page_list_chunks: list[str] = []

    def __init__(
        self, host: str, port: int, dense_embedding_model: str, code_tag_model: str
    ) -> None:
        self._qdrant_host = host
        self._qdrant_port = port
        self._code_tag_llm_model = code_tag_model
        self._dense_embedding_model = dense_embedding_model
        self._code_tag_converter = CodeTag(code_tag_model)

    def _embedding(self, content: str) -> list[float]:
        res = ollama.embeddings(model=self._dense_embedding_model, prompt=content)
        return list(res.embedding)

    def connect_db(self) -> Self:
        client = QdrantClient(host=self._qdrant_host, port=self._qdrant_port)
        self._qdrant_client = client
        return self

    def _get_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _process_code_blocks(self, text: str) -> str:
        """process code blocks with code tags (not used)"""
        pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
        matches = list(pattern.finditer(text))

        tags = []
        for m in matches:
            raw_code = m.group(2).strip()
            code_id = self._get_hash(raw_code)
            tag = self._code_tag_converter.get_code_tag(raw_code)
            if tag is not None:
                tags.append(
                    f"[CODE:{code_id}|lang={tag.lang}|desc={tag.summary.strip('.')}]"
                )

        for m, tag in zip(reversed(matches), reversed(tags)):
            text = text[: m.start()] + tag + text[m.end() :]

        return text

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

    def refine_paragrahp_chunks(self) -> Self:
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
                    is_ok = is_associated(topic, page)
                    if is_ok:
                        self._page_list_chunks.append(page)
                    continue
                elif ParagraphType.BLOCK_CODE.value in code_types:
                    is_ok = is_associated(topic, page)
                    if is_ok:
                        self._page_code_chunks.append(page)
                    # self._page_code_chunks.append(page)
                    continue

                # refine paragraph chunks in current page
                for node in nodes:
                    if node.get("type") == ParagraphType.PARAGRAPH.value:
                        node_content = self._render_children_content(node["children"])

                        if len(node_content) == 0:
                            continue
                        else:
                            self._paragraph_chunks.append(node_content)
                            # NLI method
                            # is_ok, _ = is_paragraph_supports_topic(topic, node_content)

                            # LLM method
                            is_ok = is_associated(topic, node_content)
                            if is_ok:
                                self._refined_paragraph_chunks.append(node_content)

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

    def get_filtered_paragrahp_chunk_str_list(self, container: list[str]) -> Self:
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
    print(is_associated("USA is a country", "Good"))
