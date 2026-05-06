"""
This experiment based on beta RAG system for testing keywords filter.
"""

import dataclasses
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

# from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sklearn.feature_extraction.text import TfidfVectorizer

import sys_config
from beta.ingestor import AgentConfig, AgentConfigChoseModel, Ingestor
from ingestion.filters.anchor import (
    AnchorCollector,
    AnchorSelector,
    Segment,
    SemanticModules,
    Topic,
)
from ingestion.kg_builder.entity import (
    Entity,
    cluster_keys,
    combine_entities,
    concepts_extraction,
    gliner_extraction,
)
from tools.data_loader import (
    BookNames,
    PageGroupNames,
    PageGroupRanges,
    References,
    get_book_md_path,
    get_book_path,
    get_book_pdf_path,
    get_csv_data_path,
    get_csv_log_path,
    get_json_data_path,
    # get_txt_log_path,
)
from tools.similarity import calculate_cosine_similarity

from .exp_3_config import CollectionNames

# this file is for the process pass and cache entities
KEYWORD_JSON_FILENAME = "exp_3_keywords"

# for gliner extraction
GLINER_THRESHOLD = 0.3

# for entity similarity
ENTITY_INTERSECTION_THRESHOLD = 0.5


@dataclass
class KeyWords(DataClassJsonMixin):
    """
    This is keywords holder for three stages generated.
    They are cached as a json in process for lowering computing and time consuming.
    """

    spacy: list[str]
    cluster: list[str]
    gliner: list[str]


@dataclass
class Chunk:
    """This is the progressing record for score in several dimensions"""

    content: str = ""
    matched_keywords_number: int = 0
    matched_keywords: str = ""
    feature_words: str = ""
    content_keywords: str = ""
    header_1: str = ""
    header_2: str = ""
    header_3: str = ""
    header_4: str = ""


class ChunkPipeline:
    """
    Treat the chunks by some ways for filtered chunks,
    it also can lower the time by store the data after a computing consuming process.
    """

    # target pdf file path for loading data
    # pdf_file_path: str
    # md_file_path: str

    # the cached file contained scores and chunks
    cache_file_path: str

    # the loaded chunks from cached file or PDF file
    chunks: list[Chunk] = []

    # for comparing keywords
    gliner_global_keywords: list[str]
    gliner_labels: list[str]

    # the model for fluency score of content
    # llm: BaseChatModel

    # the splitter for keywords stored in db and inner of class
    KEYWORDS_SPLITTER: str = "|"

    def __init__(
        self,
        kws: list[str] = [],
        labels: list[str] = [],
    ):
        # self.pdf_file_path = pdf_file_path
        # self.md_file_path = md_file_path
        self.gliner_global_keywords = kws
        self.gliner_labels = labels
        # self.llm = ChatOllama(
        #     base_url=sys_config.OLLAMA_URL_BASE,
        #     model=llm_name,
        #     temperature=0,
        # )
        self.cache_file_path = str(get_csv_data_path("chunks"))

    def save_chunk_as_csv(self, target_file: str = "") -> Self:
        df = pd.DataFrame([dataclasses.asdict(c) for c in self.chunks])
        target_path = self.cache_file_path
        if target_file != "":
            target_path = str(get_csv_data_path(target_file))
        df.to_csv(target_path, index=False, encoding="utf-8")
        return self

    def load_chunks_from_cache_cvs(self, target_file: str = "") -> Self:
        target_path = self.cache_file_path
        if target_file != "":
            target_path = str(get_csv_data_path(target_file))
        df = pd.read_csv(target_path)
        self.chunks = [Chunk(**row) for row in df.to_dict(orient="records")]
        return self

    def load_chunks_from_pdf(self, pdf_file_name: str) -> Self:
        """
        Chunk the given PDF file into documents defined by langchain,
        it powered by PyMuPDFLoader which is not recommended for complex documents.
        """
        pdf_path = get_book_pdf_path(pdf_file_name)
        loader = PyMuPDFLoader(pdf_path)

        # loaded_docs[23:], after page 23 is first chapter for book 2
        loaded_docs = loader.load()

        # the GLiner support 512 tokens, one token about 3 to 4 characters
        # the largest size is set as 1200, and overlap is 200
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            is_separator_regex=True,
            separators=[r"(?<=[。！？\.!?])\s"],
            # separators=["\n\n", "\n", r"(?<=[.!?])\s", " "],
        )

        docs = text_splitter.split_documents(loaded_docs)
        for doc in docs:
            content = doc.page_content
            chunk = Chunk(
                content=content,
            )
            self.chunks.append(chunk)

        return self

    def load_chunks_from_md(self, md_file_name: str) -> Self:
        md_path = get_book_md_path(md_file_name)
        md_loader = TextLoader(md_path, encoding="utf-8")
        # this loader can read the dir with serveral files,
        # so its return is the list of documents
        md_data = md_loader.load()

        # the splitter based on markdown headers
        md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
            ],
            strip_headers=True,
        )

        raw_md_content = md_data[0].page_content

        # the splitter devided the input larger document into smaller documents
        header_docs = md_header_splitter.split_text(raw_md_content)
        for doc in header_docs:
            self.chunks.append(
                Chunk(
                    content=doc.page_content,
                    header_1=doc.metadata.get("header_1", ""),
                    header_2=doc.metadata.get("header_2", ""),
                    header_3=doc.metadata.get("header_3", ""),
                    header_4=doc.metadata.get("header_4", ""),
                )
            )

        return self

    def _get_intersection(
        self, doc_keywords: list[str], s_keywords: list[str], threshold: float = 0.7
    ) -> list[str]:
        intersection: list[str] = []
        for sk in s_keywords:
            for dk in doc_keywords:
                if calculate_cosine_similarity([sk], [dk])[0] > threshold:
                    intersection.append(dk)
            # if len(intersection) > 0:
            #     break
        return intersection

    def score_keywords(self) -> Self:
        """Extract the chunks' keywords based on provided labels, figure out the matched keywords"""
        for i, chunk in enumerate(self.chunks):
            content_entities = gliner_extraction(
                chunk.content, self.gliner_labels, GLINER_THRESHOLD
            )
            content_kws = [e.text.lower() for e in content_entities]
            matched_kws = self._get_intersection(
                content_kws, self.gliner_global_keywords, ENTITY_INTERSECTION_THRESHOLD
            )
            chunk.content_keywords = self.KEYWORDS_SPLITTER.join(content_kws)
            chunk.matched_keywords = self.KEYWORDS_SPLITTER.join(matched_kws)
            chunk.matched_keywords_number = len(matched_kws)
            print(f"score keywords in content {i}: {chunk.content_keywords}")
        return self

    def filter_keywords(self, limit: int = 1) -> Self:
        filtered_chunks: list[Chunk] = []
        for c in self.chunks:
            if c.matched_keywords_number >= limit:
                filtered_chunks.append(c)
        self.chunks = filtered_chunks
        return self

    def score_features(self, top_k: int = 5) -> Self:
        """This method need more step to normalization and test"""
        all_texts = [c.content for c in self.chunks]

        # use english words types, form 1 word to 3 word
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        all_feature_names = vectorizer.get_feature_names_out()

        # get top features
        for text_i in range(len(all_texts)):
            row = np.asarray(tfidf_matrix[text_i].todense()).flatten()
            top_indices = np.argsort(row)[::-1][:top_k]
            chunk_features = set()
            for i in top_indices:
                if float(row[i]) > 0:
                    chunk_features.add(all_feature_names[i])
            self.chunks[text_i].feature_words = self.KEYWORDS_SPLITTER.join(
                chunk_features
            ).lower()

        return self

    def refine_md_chunks_by_llm(self, std_answers: list[str], llm_name: str) -> Self:
        """Powered by given ollama llm"""
        # init the collector
        anchor_collector = AnchorCollector(
            ollama_url=sys_config.OLLAMA_URL_BASE,
            ollama_model=llm_name,
        )

        # generate segments
        segments: list[Segment] = []
        for c in self.chunks:
            seg = Segment(content=c.content)
            segments.append(seg)

        # generage topics
        topics: list[Topic] = []
        for answer in std_answers:
            topic = Topic(content=answer)
            topics.append(topic)

        refined_list: list[str] = []
        anchor_collector.push_data(
            topics, segments
        ).refine_segments_by_topics().get_refined_segments_str_list(refined_list)

        refined_chunks: list[Chunk] = []
        for chunk_str in refined_list:
            for c in self.chunks:
                if c.content == chunk_str:
                    refined_chunks.append(c)

        self.chunks = refined_chunks
        return self

    def refine_chunks_from_db(
        self,
        base_topics: list[str],
        level: int,
        is_skip_code_paragraph: bool,
        semantic_module: SemanticModules,
        collection_name: str,
        pages_per_topic: int,
        pages_threshold: float,
        llm_name: str,
        embedding_model: str,
    ) -> Self:
        """
        Refine the related chunks from database by cosine score.
        The implement is leveraging qdrant query method and algorithms.
        The embedding model is for retriving from the db, and the llm model is for comparing chunks.
        level: 0 - page level, 1 - paragraph level
        is_skip_code_paragraph: it is effective only when level is 1
        """

        anchor_selector = AnchorSelector(
            qdrant_host=sys_config.QDRANT_HOST,
            qdrant_port=sys_config.QDRANT_PORT,
            dense_embedding_model=embedding_model,
            ollama_url=sys_config.OLLAMA_URL_BASE,
            ollama_model=llm_name,
        )

        # wrap str topics as objects
        topics: list[Topic] = []
        for bt in base_topics:
            topic = Topic(content=bt)
            topics.append(topic)

        # refine the related page-level chunks from database
        refined_content_list: list[str] = []
        anchor_selector.connect_db().refine_page_chunks_from_queried_points(
            topics=topics,
            collection_name=collection_name,
            size=pages_per_topic,
            threshold=pages_threshold,
        )

        if level == 0:
            anchor_selector.get_selected_page_chunk_str_list(refined_content_list)
            print(f"layer one: get {len(refined_content_list)} candidate chunks")

        if level == 1:
            anchor_selector.refine_paragrahp_chunks(
                semantic_module, is_drop_code=is_skip_code_paragraph
            ).get_hybrid_chunks_str_list(refined_content_list)
            print(f"layer two: get {len(refined_content_list)} semanstic chunks")

        # remove duplicated
        refined_content_list = list(set(refined_content_list))

        # wrap str chunks as objects
        refined_chunks: list[Chunk] = []
        for rc in refined_content_list:
            refined_chunks.append(Chunk(content=rc))

        self.chunks = refined_chunks

        return self

    def ingest(
        self,
        embedding_name: str,
        embedding_dim: int,
        collection_name: str,
        is_hybrid=False,
        group_ref: str = "",
    ) -> Self:
        config = AgentConfig(
            collection_name=collection_name, is_hybrid_search=is_hybrid
        )
        AgentConfigChoseModel.chose_ollama_embedding(
            config=config,
            model_name=embedding_name,
            dimensions=embedding_dim,
        )

        ingestor = Ingestor(config)
        ingestor.use_ollama_embeddings()
        ingestor.force_create_collection()

        docs: list[Document] = []
        for c in self.chunks:
            doc = Document(
                page_content=c.content,
                metadata={
                    "keywords": str(c.content_keywords).split(self.KEYWORDS_SPLITTER),
                    "features": str(c.feature_words).split(self.KEYWORDS_SPLITTER),
                    "header_1": c.header_1,
                    "header_2": c.header_2,
                    "header_3": c.header_3,
                    "header_4": c.header_4,
                },
            )
            docs.append(doc)

        # save the chunks in vector database
        ingestor.save_docs(
            docs=docs,
            file_ref=group_ref,
        )

        # output ingestion logs
        log_df = pd.DataFrame(
            {
                "chunks": [c.content for c in self.chunks],
            }
        )
        log_path = get_csv_log_path("slim_exp_3_ingestion")
        log_df.to_csv(log_path, index=True, encoding="utf-8")

        return self


def load_standard_answers(file_name: str, start: int, end: int) -> list[str]:
    """
    Load the source of answers by given indexes,
    for example: 30,60 get book 2 anwers
    Arguments:
        start: the start index of answers list, from 1
        end: the end index of answers list
    return: the list of questions
    """
    df = pd.read_csv(str(get_csv_data_path(file_name)))
    source_df = df["standard_answers"]
    return source_df[start:end].dropna().tolist()


def load_group_2_standard_answers() -> list[str]:
    """
    Load the source of answers associated with pages of book 2
    """
    df = pd.read_csv(str(get_csv_data_path("questions_and_answers")))
    source_df = df["standard_answers"]
    return source_df[30:60].dropna().tolist()


def print_splitter_with_head(head: str):
    """print split line with given header"""
    print(f"================ {head} ================")


def extract_and_save_keywords():
    entity_source = load_group_2_standard_answers()

    concepts_list_spacy: list[str] = []
    concepts_list: list[Entity] = []
    concepts_str_list: list[str] = []

    for c in entity_source:
        concepts_list_spacy += concepts_extraction(c)
    concepts_list_spacy = list(set(concepts_list_spacy))

    print_splitter_with_head(f"SpaCy Keywords({len(concepts_list_spacy)})")
    print(concepts_list_spacy)

    concepts_categories_as_labels = cluster_keys(concepts_list_spacy)
    for c in entity_source:
        concepts_list += gliner_extraction(
            c, concepts_categories_as_labels, GLINER_THRESHOLD
        )
    print_splitter_with_head("clusters and llm result")
    print(concepts_categories_as_labels)

    print_splitter_with_head("Gliner")
    concepts_list = combine_entities(concepts_list)
    concepts_str_list = [c.text for c in concepts_list]
    print(concepts_str_list)

    kw = KeyWords(
        spacy=concepts_list_spacy,
        cluster=concepts_categories_as_labels,
        gliner=concepts_str_list,
    )

    with open(get_json_data_path(KEYWORD_JSON_FILENAME), "w", encoding="utf-8") as f:
        f.write(kw.to_json(indent=2))
    print("Keywords file writing is finished")


def load_extracted_entities() -> KeyWords:
    with open(get_json_data_path(KEYWORD_JSON_FILENAME), "r", encoding="utf-8") as f:
        kw = KeyWords.from_json(f.read())
        return kw


def exploring_ingestion_piplines():

    # this is working for extract keywords from qustions, chunks for comparing and evaluation
    # in current research is not used
    kws = load_extracted_entities()

    # the exp 3 includes three stages:
    # PDF no filter & keywords filter, semanstic filter by LLM, muiti-layer filter.
    # the pipline can hold all stage
    # the keywords is not used, but to keep them in the code for recording
    chunk_pipline = ChunkPipeline(
        kws=kws.gliner,
        labels=kws.cluster,
    )

    ### First stage: keywords match number filter
    # based on PDF and fixed size chunking

    FIXED_CHUNKS = "p_fixed_chunks"
    FIXED_CHUNKS_KW_SCORE = "p_fixed_chunks_kw_score"

    # fixed chunking without filtering
    chunk_pipline.load_chunks_from_pdf(
        PageGroupNames.RESPONSIVE_WEB_DESIGN_2.value
    ).score_keywords().save_chunk_as_csv(FIXED_CHUNKS).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.FIX_CHUNKING.value,
    )

    # keywords score for next filtering
    chunk_pipline.load_chunks_from_cache_cvs(
        FIXED_CHUNKS
    ).score_keywords().save_chunk_as_csv(FIXED_CHUNKS_KW_SCORE)

    chunk_pipline.load_chunks_from_cache_cvs(FIXED_CHUNKS_KW_SCORE).filter_keywords(
        limit=1
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.FIX_CHUNKING_1_KEYWORD.value,
    )

    chunk_pipline.load_chunks_from_cache_cvs(FIXED_CHUNKS_KW_SCORE).filter_keywords(
        limit=2
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.FIX_CHUNKING_2_KEYWORDS.value,
    )

    ### Second stage: semantic filter
    # based on header chunking and LLM semantic filter
    # computing consuming M*N

    # MD method without filter
    chunk_pipline.load_chunks_from_md(
        PageGroupNames.RESPONSIVE_WEB_DESIGN_2.value
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
    )

    # MD method with llm filter depend on low quality md file
    std_answers = load_group_2_standard_answers()

    chunk_pipline.load_chunks_from_md(
        PageGroupNames.RESPONSIVE_WEB_DESIGN_2.value
    ).refine_md_chunks_by_llm(
        std_answers, sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_LLM.value,
    )

    ### Third stage: multi layer filter
    # based on header chunking and cosine and LLM semantic filter
    # to lower the computing cost

    std_answers = load_group_2_standard_answers()

    # layers filter, the db source is from the result of md chunking without filter
    # so the query embedding should follow it
    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=0,
        is_skip_code_paragraph=False,
        semantic_module=SemanticModules.LLM,
        collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
        pages_per_topic=4,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
        embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PAGE.value,
    )

    # increase cosine output from 4 to 8
    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=0,
        is_skip_code_paragraph=False,
        semantic_module=SemanticModules.LLM,
        collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
        pages_per_topic=8,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
        embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PAGE_CONSINE_8_CHUNK.value,
    )

    # use smaller chunk
    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=1,
        is_skip_code_paragraph=False,
        semantic_module=SemanticModules.LLM,
        collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
        pages_per_topic=8,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
        embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH.value,
    )

    # use smaller chunk without code
    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=1,
        is_skip_code_paragraph=True,
        semantic_module=SemanticModules.LLM,
        collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
        pages_per_topic=8,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
        embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH_NO_CODE.value,
    )

    # use NLI as the semantic layer, future work
    # chunk_pipline.refine_chunks_from_db(
    #     base_topics=std_answers,
    #     level=1,
    #     is_skip_code_paragraph=False,
    #     semantic_module=SemanticModules.NLI,
    #     collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
    #     pages_per_topic=8,
    #     pages_threshold=0.5,
    #     llm_name=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
    #     embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    # ).ingest(
    #     embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    #     embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
    #     collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH_NLI.value,
    # )

    # use Gemma as the semantic layer
    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=1,
        is_skip_code_paragraph=False,
        semantic_module=SemanticModules.LLM,
        collection_name=CollectionNames.MD_HEAD_CHUNKING.value,
        pages_per_topic=8,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GEMMA_MODEL_4_E2B,
        embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH_GEMMA.value,
    )

    # use Gemma ingestion, which means the all process is depended on GEMMA
    # in this stage, the QA process should also use GEMMA

    # update the source db by GEMMA embedding
    chunk_pipline.load_chunks_from_md(
        PageGroupNames.RESPONSIVE_WEB_DESIGN_2.value
    ).ingest(
        embedding_name=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_2.value,
    )

    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=1,
        is_skip_code_paragraph=False,
        semantic_module=SemanticModules.LLM,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_2.value,
        pages_per_topic=8,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GEMMA_MODEL_4_E2B,
        embedding_model=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=CollectionNames.MD_HEAD_CHUNKING_ML_PARAGRAPH_GEMMA.value,
    )


def extended_ingestion(
    md_file_name: str,
    source_c_name: str,
    target_c_name: str,
    topics: list[str],
    group_ref: str,
):
    """other group sets"""
    chunk_pipline = ChunkPipeline()

    # prepare first layer db
    chunk_pipline.load_chunks_from_md(md_file_name).ingest(
        embedding_name=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=source_c_name,
    )

    # collect varified chunks from the temp db
    chunk_pipline.refine_chunks_from_db(
        base_topics=topics,
        level=1,
        is_skip_code_paragraph=False,
        semantic_module=SemanticModules.LLM,
        collection_name=source_c_name,
        pages_per_topic=8,
        pages_threshold=0.5,
        llm_name=sys_config.OLLAMA_GEMMA_MODEL_4_E2B,
        embedding_model=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
    ).ingest(
        embedding_name=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768,
        embedding_dim=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_DIMENSIONS,
        collection_name=target_c_name,
        group_ref=group_ref,
    )


if __name__ == "__main__":
    # in practice, the topic is manitained by manager for limiting the chat area
    # in experiment stage, the standard answers (groundtruth) is used as topic
    # to prepare keywords, just need to run once
    # the topic is also used to filter the chunks by semantic comparison in multi layer filter pipline
    # extract_and_save_keywords()

    # in exploring stage, only focus on one group (2) data
    # exploring_ingestion_piplines()

    # in extension stage, use the best options of filter to test other page groups
    topics_1 = load_standard_answers("questions_and_answers", 0, 30)
    extended_ingestion(
        md_file_name=PageGroupNames.INTRO_WEB_DEV_1.value,
        source_c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_1_TMP.value,
        target_c_name=CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_1.value,
        topics=topics_1,
        group_ref=References.INTRO_WEB_DEV_1.value,
    )

    # extended_ingestion(
    #     PageGroupNames.LEARNING_REACT_4.value,
    #     CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_4.value,
    # )
    # extended_ingestion(
    #     PageGroupNames.DESIGN_PATTERN_5.value,
    #     CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_5.value,
    # )
    # extended_ingestion(
    #     PageGroupNames.STRUCTURE_INTERPRETATION_6.value,
    #     CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_6.value,
    # )
    # extended_ingestion(
    #     PageGroupNames.SOCIAL_MARKETING_8.value,
    #     CollectionNames.MD_HEAD_CHUNKING_PAGES_GROUP_8.value,
    # )
