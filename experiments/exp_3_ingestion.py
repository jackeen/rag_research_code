"""
This experiment based on beta RAG system for testing keywords filter.
"""

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
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
from ingestion.filters.anchor import AnchorCollector, AnchorSelector, Segment, Topic
from ingestion.kg_builder.entity import (
    Entity,
    cluster_keys,
    combine_entities,
    concepts_extraction,
    gliner_extraction,
)
from tools.data_loader import (
    BookNames,
    References,
    get_book_md_path,
    get_book_path,
    get_book_pdf_path,
    get_csv_data_path,
    get_csv_log_path,
    get_json_data_path,
    get_txt_log_path,
)
from tools.similarity import calculate_cosine_similarity

# this file is for the process pass and cache entities
KEYWORD_JSON_FILENAME = "exp_3_keywords"

# this file is for caching the semanstic anchors (topic) depending on std_answers
ANCHOR_JSON_FILENAME = "exp_3_chunk_anchors"

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
class Anchors(DataClassJsonMixin):
    """"""

    topic_content_list: list[str] = field(default_factory=list)
    anchor_list: list[list[str]] = field(default_factory=list)


@dataclass
class Chunk:
    """This is the progressing record for score in several dimensions"""

    content: str = ""
    matched_keywords_number: int = 0
    matched_keywords: str = ""
    feature_words: str = ""
    content_keywords: str = ""
    KFS: float = 0.0
    summary: str = ""
    is_code: bool = False
    fluency_score: float = 0.0
    header_1: str = ""
    header_2: str = ""
    header_3: str = ""
    header_4: str = ""


class ChunkPipeline:
    """Treat the chunks by some ways for filtered chunks"""

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

    def __init__(self, kws: list[str], labels: list[str]):
        # self.pdf_file_path = pdf_file_path
        # self.md_file_path = md_file_path
        self.gliner_global_keywords = kws
        self.gliner_labels = labels
        self.llm = ChatOllama(
            base_url=sys_config.OLLAMA_URL_BASE,
            model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
            temperature=0,
        )
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

    def score_keywords(self) -> Self:
        """Extract the chunks' keywords based on provided labels, figure out the matched keywords"""
        for i, chunk in enumerate(self.chunks):
            content_entities = gliner_extraction(
                chunk.content, self.gliner_labels, GLINER_THRESHOLD
            )
            content_kws = [e.text.lower() for e in content_entities]
            matched_kws = get_intersection(
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

    def filter_features_with_keywords(self, min: float = 0.6, max: float = 1.0) -> Self:
        filtered_chunks: list[Chunk] = []
        for c in self.chunks:
            KFS = calculate_cosine_similarity([c.feature_words], [c.content_keywords])[
                0
            ]
            if min <= KFS <= max:
                c.KFS = KFS
                filtered_chunks.append(c)
        self.chunks = filtered_chunks
        return self

    def filter_keywords_statistics(self, threshold: float = 0.5) -> Self:
        filtered_chunks: list[Chunk] = []
        for chunk in self.chunks:
            kws = chunk.content_keywords
            kws_list = kws.split(self.KEYWORDS_SPLITTER)
            counter = Counter(kws_list)
            total_words_n = len(kws_list)
            if total_words_n < 4:
                continue
            for _, count in counter.most_common():
                ratio = count / total_words_n
                if ratio <= threshold:
                    filtered_chunks.append(chunk)
                    break

        self.chunks = filtered_chunks
        return self

    def refine_md_chunks_by_llm(self, std_answers: list[str]) -> Self:
        # init the collector
        anchor_collector = AnchorCollector(
            ollama_url=sys_config.OLLAMA_URL_BASE,
            ollama_model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
        )

        # generate segments
        segments: list[Segment] = []
        for chunk in self.chunks:
            seg = Segment(content=chunk.content)
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
            for chunk in self.chunks:
                if chunk.content == chunk_str:
                    refined_chunks.append(chunk)

        self.chunks = refined_chunks
        return self

    def refine_chunks_from_db(
        self,
        base_topics: list[str],
        level: int,
        collection_name: str,
        pages_per_topic: int,
        pages_threshold: float,
    ) -> Self:
        """
        Refine the related chunks from database by cosine score.
        The implement is leveraging qdrant query method and algorithms.
        level: 0 - page level, 1 - paragraph level
        """
        anchor_selector = AnchorSelector(
            host=sys_config.QDRANT_HOST,
            port=sys_config.QDRANT_PORT,
            dense_embedding_model=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
            code_tag_model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
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

        if level == 1:
            anchor_selector.refine_paragrahp_chunks().get_hybrid_chunks_str_list(
                refined_content_list
            )

        # wrap str chunks as objects
        refined_chunks: list[Chunk] = []
        for rc in refined_content_list:
            refined_chunks.append(Chunk(content=rc))

        self.chunks = refined_chunks

        return self

    # def refine_paragraph_chunks_from_page_chunks(self) -> Self:
    #     return self

    def ingest(
        self, collection_name: str = "exp_3_entity_filtered", is_hybrid=False
    ) -> Self:
        config = AgentConfig(
            collection_name=collection_name, is_hybrid_search=is_hybrid
        )
        AgentConfigChoseModel.chose_ollama_llm_model(
            config, sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
        )
        AgentConfigChoseModel.chose_ollama_embedding(
            config=config,
            model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
            dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
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
                    "KFS": c.KFS,
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
            file_ref=References.RESPONSIVE_WEB_DESIGN_2.value,
        )

        # output ingestion logs
        log_df = pd.DataFrame(
            {
                "chunks": [chunk.content for chunk in self.chunks],
            }
        )
        log_path = get_csv_log_path("exp_3_ingestion")
        log_df.to_csv(log_path, index=True, encoding="utf-8")

        return self


def get_intersection(
    doc_keywords: list[str], s_keywords: list[str], threshold: float = 0.7
) -> list[str]:
    intersection: list[str] = []
    for sk in s_keywords:
        for dk in doc_keywords:
            if calculate_cosine_similarity([sk], [dk])[0] > threshold:
                intersection.append(dk)
        # if len(intersection) > 0:
        #     break
    return intersection


def ingest_book_without_filter():
    config = AgentConfig(
        collection_name="exp_3_entity",
    )
    AgentConfigChoseModel.chose_ollama_llm_model(
        config, sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
    )
    AgentConfigChoseModel.chose_ollama_embedding(
        config=config,
        model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
    )

    ingestor = Ingestor(config)
    ingestor.use_ollama_embeddings()
    ingestor.force_create_collection()

    # pymupdf4llm try this later
    loader = PyMuPDFLoader(str(get_book_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)))
    loaded_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[r"(?<=[。！？\.!?])\s"],
        # separators=["\n\n", "\n", r"(?<=[.!?])\s", " "],
        is_separator_regex=True,
    )

    # loaded_docs[23:]
    docs = text_splitter.split_documents(loaded_docs)
    ingestor.save_docs(docs, References.RESPONSIVE_WEB_DESIGN_2.value)


def ingest_book(concepts: list[str], labels: list[str]):
    """ingestion with keywords filter (not used)"""
    config = AgentConfig(
        collection_name="exp_3_entity_filtered", is_hybrid_search=False
    )
    AgentConfigChoseModel.chose_ollama_llm_model(
        config, sys_config.OLLAMA_GRANITE_MODEL_4_3B_H
    )
    AgentConfigChoseModel.chose_ollama_embedding(
        config=config,
        model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS,
    )

    ingestor = Ingestor(config)
    ingestor.use_ollama_embeddings()
    ingestor.force_create_collection()

    # prepare data for chunking
    loader = PyMuPDFLoader(str(get_book_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)))
    loaded_docs = loader.load()

    # the GLiner support 512 tokens, one token about 3 to 4 characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[r"(?<=[。！？\.!?])\s"],
        # separators=["\n\n", "\n", r"(?<=[.!?])\s", " "],
        is_separator_regex=True,
    )

    # loaded_docs[23:]
    docs = text_splitter.split_documents(loaded_docs)

    # for chunk log
    control_group = ", ".join([c for c in concepts])
    contents = []
    control_keywords = []
    contents_keywords = []
    matched_numbers = []

    filtered_docs = []
    for _, doc in enumerate(docs):
        doc_entities = combine_entities(
            gliner_extraction(doc.page_content, labels, GLINER_THRESHOLD)
        )
        doc_keywords = [entity.text for entity in doc_entities]
        matched_words = get_intersection(
            doc_keywords, concepts, ENTITY_INTERSECTION_THRESHOLD
        )

        if len(matched_words) > 0:
            doc.metadata["keywords"] = doc_keywords
            doc.metadata["matched_keywords"] = matched_words
            filtered_docs.append(doc)

        # for chunk log
        contents.append(doc.page_content)
        control_keywords.append(control_group)
        contents_keywords.append(", ".join(doc_keywords))
        matched_numbers.append(len(matched_words))

    ingestor.save_docs(filtered_docs, References.RESPONSIVE_WEB_DESIGN_2.value)
    print(f"{len(filtered_docs)} filtered chunks saved")

    log_df = pd.DataFrame(
        {
            "chunks": contents,
            "matched_numbers": matched_numbers,
            "chunk_keywords": contents_keywords,
            "control_keywords": control_keywords,
        }
    )
    log_path = get_csv_log_path("exp_3_ingestion")
    log_df.to_csv(log_path, index=True, encoding="utf-8")


def load_standard_answers() -> list[str]:
    """
    Load the source of entities about Responsive Web Design
    :return: the list of questions
    """
    df = pd.read_csv(str(get_csv_data_path("questions_and_answers")))
    source_df = df["standard_answers"]
    return source_df[30:60].dropna().tolist()


def print_splitter_with_head(head: str):
    """print split line with given header"""
    print(f"================ {head} ================")


def extract_and_save_keywords():
    entity_source = load_standard_answers()

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


def extract_and_save_anchors():
    entity_source = load_standard_answers()
    anchors = Anchors()
    for c in entity_source:
        anchors.topic_content_list.append(c)
        concept_list = concepts_extraction(c)

        anchors.anchor_list.append(concept_list)
        # next part should to do more work for exceptions for better result
        # this part can also use other methods, not be only limited
        # if len(concept_list) <= 2:
        #     anchors.anchor_list.append(concept_list)
        #     continue

        # labels = cluster_keys(concept_list)
        # x = [concept.text for concept in gliner_extraction(c, labels, GLINER_THRESHOLD)]
        # anchors.anchor_list.append(x)

    with open(get_json_data_path(ANCHOR_JSON_FILENAME), "w", encoding="utf-8") as f:
        f.write(anchors.to_json(indent=2))

    print("Anchors file writing is finished")


def load_extracted_anchors() -> Anchors:
    with open(get_json_data_path(ANCHOR_JSON_FILENAME), "r", encoding="utf-8") as f:
        anchors = Anchors.from_json(f.read())
        return anchors


def entity_filter_task():
    """Ingest the documents by entity filter"""
    kw = load_extracted_entities()
    print(f"Ingestion task loads keywords: {kw}")

    # use the given keywords for ingestion
    ingest_book(kw.gliner, kw.cluster)


def score_chunks_before_ingestion():
    # pdf_file_path=str(get_book_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)),
    # md_file_path=str(get_book_md_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)),
    kws = load_extracted_entities()
    chunk_pipline = ChunkPipeline(
        kws=kws.gliner,
        labels=kws.cluster,
    )

    ### score filter based on keywords
    # chunk_pipline.load_chunks_from_cache_cvs().score_keywords().score_features().save_chunk_as_csv()
    # chunk_pipline.load_chunks_from_cache_cvs().filter_keywords().filter_features_with_keywords(
    #     0.5, 1
    # ).save_chunk_as_csv()

    # chunk_pipline.load_chunks_from_cache_cvs().filter_keywords_statistics().save_chunk_as_csv(
    #     "chunks_statistics"
    # )

    # chunk_pipline.load_chunks_from_cache_cvs(
    #     "chunks_backup"
    # ).score_features().score_keywords().save_chunk_as_csv("chunks_score")

    # chunk_pipline.load_chunks_from_cache_cvs(
    #     "chunks_score"
    # ).filter_features_with_keywords(0.3, 1.0).ingest()

    # chunk_pipline.load_chunks_from_cache_cvs("chunks_score").filter_keywords().ingest()

    ############################## the upper method is old experiments

    ### MD methods
    # chunk_pipline.load_chunks_from_md("2_slim").save_chunk_as_csv("chunks_md").ingest(
    #     "exp_3_md"
    # )
    # chunk_pipline.load_chunks_from_cache_cvs("chunks_md").ingest("exp_3_md")

    ### MD methods with llm filter (not used)
    # leverage llm to figure out each chunk that suitable for topics
    # this costs about one hour, lower 19% chunks
    # std_answers = load_standard_answers()
    # chunk_pipline.load_chunks_from_cache_cvs("chunks_md").refine_md_chunks_by_llm(
    #     std_answers
    # ).ingest("exp_3_md_refined")

    ### MD methods with label filter
    # refine chunks from db
    std_answers = load_standard_answers()
    chunk_pipline.refine_chunks_from_db(
        base_topics=std_answers,
        level=1,
        collection_name="exp_3_md",
        pages_per_topic=8,
        pages_threshold=0.5,
    ).save_chunk_as_csv("exp_3_md_refined_normal_paragraph").ingest(
        collection_name="exp_3_md_refined_normal_paragraph_chunks_2",
    )


def ingest_std_answers():
    std_answers = load_standard_answers()
    kws = load_extracted_entities()
    pipline = ChunkPipeline(kws.gliner, kws.cluster)
    pipline.chunks = [Chunk(content=answer) for answer in std_answers]
    pipline.ingest("exp_3_std_answer")


if __name__ == "__main__":
    # ingest_std_answers()

    # extract_and_save_keywords()
    # entity_filter_task()

    # extract_and_save_anchors()
    # print(load_extracted_anchors())

    # This part is for comparison
    # ingest_book_without_filter()

    score_chunks_before_ingestion()
