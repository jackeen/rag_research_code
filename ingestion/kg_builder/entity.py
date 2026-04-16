from collections import Counter, defaultdict
from dataclasses import dataclass

import dotenv
import hdbscan
import spacy
from gliner import GLiNER
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline

import sys_config

# load
dotenv.load_dotenv()

QUESTION_WORDS_SET = {"what", "where", "when", "which", "how", "who", "why"}
PRONOUN_AND_ARTICLE_WORDS_SET = {
    "this",
    "that",
    "these",
    "those",
    "its",
    "their",
    "our",
    "your",
    "a",
    "an",
    "the",
}
WEAK_HEADS_SET = {
    "context",
    "mechanism",
    "list",
    "core",
    "way",
    "thing",
    "type",
    "kind",
    "group",
    "set",
    "state",
    "purpose",
    "requirement",
    "approach",
    "technique",
    "application",
    "standpoint",
    "spectrum",
    "constraint",
    "shift",
    "aspect",
    "user",
    "developer",
    "device",
    "website",
    "webpage",
    "browser",
    "string",
    "code",
    "form",
    "element",
    "support",
    "version",
    "feature",
    "option",
}
LEMMA_FIX_DICT = {
    "medium": "media",
}


llm = ChatOllama(
    base_url=sys_config.OLLAMA_URL_BASE,
    model=sys_config.OLLAMA_GRANITE_MODEL_3_3_8B,
    temperature=0,
    # reasoning=False,
    format="json",
)

cluster_llm = ChatOllama(
    base_url=sys_config.OLLAMA_URL_BASE,
    model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
    temperature=0,
    # format='json',
)

# dotenv.load_dotenv()
# cluster_llm = ChatOpenAI(
#     model=sys_config.OPEN_AI_MODEL,
#     temperature=0,
#     format='json',
# )


def entities_extraction(content: str) -> list[tuple[str, str]]:
    # fine-tuned for extracting ORG, PEO, LOC or MISC
    model = "elastic/distilbert-base-uncased-finetuned-conll03-english"
    # model = 'elastic/distilbert-base-cased-finetuned-conll03-english'
    # model = "dbmdz/bert-large-cased-finetuned-conll03-english"

    # NER (Named Entity Recognition) task
    # token-classification
    ner = pipeline(
        task="token-classification",
        model=model,
        aggregation_strategy="first",
    )
    entities = ner(inputs=content)
    # for entity in entities:
    #      print(entity)

    return [(entity["word"], entity["entity_group"]) for entity in entities]


def verbs_extraction(content: str) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    counter = Counter()
    doc = nlp(content)
    sentences = [sent.text for sent in doc.sents]
    for sentence in sentences:
        sent_doc = nlp(sentence)
        for token in sent_doc:
            if token.pos_ == "VERB" and not token.is_stop:
                counter[token.lemma_] += 1

    return [verb for verb, _ in counter.most_common(10)]


# def concepts_extraction(content: str) -> list[str]:
#     nlp = spacy.load('en_core_web_sm')
#     doc = nlp(content)
#     concepts = []
#     for chunk in doc.noun_chunks:
#         if chunk.root.pos_ != 'PRON' and not chunk.root.is_stop:
#             concepts.append(chunk.text.lower())
#
#     return concepts


def is_clear_concept(text: str) -> bool:
    tokens = text.lower().split()
    if not tokens:
        return False

    first_token = tokens[0]
    last_token = tokens[-1]

    if first_token in QUESTION_WORDS_SET:
        return False
    if first_token in PRONOUN_AND_ARTICLE_WORDS_SET:
        return False
    if len(first_token) == 1 and first_token in WEAK_HEADS_SET:
        return False
    if last_token in WEAK_HEADS_SET:
        return False

    return True


# if init this many times which will cause break of python
con_nlp = spacy.load("en_core_web_trf")

def concepts_extraction(content: str) -> list[str]:
    """to extract concepts from a given content by spacy"""
    # nlp = spacy.load("en_core_web_trf")
    # nlp = spacy.load('en_core_web_sm')

    # combine noun chunks
    if "merge_noun_chunks" not in con_nlp.pipe_names:
        con_nlp.add_pipe("merge_noun_chunks")

    doc = con_nlp(content)
    refined_concepts = set()
    for token in doc:
        # filter pronoun, punctuation
        if token.pos_ in ["PRON", "PUNCT"] or token.is_stop:
            continue
        # leave noun, proper noun
        if token.pos_ in ["NOUN", "PROPN"]:
            # token.text is the original word
            # token.lemma is the restored version of the word
            concept = token.lemma_.lower().strip()

            # fix lemma bug
            # concept = LEMMA_FIX_DICT.get(concept, concept)

            if len(concept) > 1 and is_clear_concept(concept):
                refined_concepts.add(concept)

    return list(refined_concepts)


def llm_extraction(content: str):
    prompt = """
    # Role
    You are an expert Knowledge Graph Extractor specializing in Computer Science and Software Engineering.

    # Task
    Extract a list of Entities and their Relationships from the provided technical text to build a GraphRAG index.

    # Schema Definitions
    1. **Entity**: Technical terms, tools, frameworks, languages, architectural patterns, or data structures.
    2. **Relationship**: A directed link between two entities (e.g., "implements", "uses", "optimizes", "deploys", "encapsulates").

    # Extraction Rules (Strict)
    - **Granularity**: Keep entity names concise (e.g., "Rust" instead of "The Rust Programming Language").
    - **Exclusion**: Do not extract generic pronouns or non-technical concepts.
    - **No Hallucination**: Only extract information explicitly stated in the text.
    - **Output**: You MUST respond ONLY with a valid JSON object. No conversational filler.

    # JSON Structure
    {
      "entities": [
        {"name": "Entity Name", "type": "Category", "description": "One sentence context"}
      ],
      "relationships": [
        {"source": "Entity A", "target": "Entity B", "relation": "Verb/Action"}
      ]
    }

    # Example
    Input: "The Actix Web framework leverages the Tokio runtime for asynchronous I/O handling."
    Output: {
      "entities": [
        {"name": "Actix Web", "type": "Framework", "description": "A powerful, pragmatic, and extremely fast web framework for Rust."},
        {"name": "Tokio", "type": "Runtime", "description": "An event-driven, non-blocking I/O platform for writing asynchronous applications."}
      ],
      "relationships": [
        {"source": "Actix Web", "target": "Tokio", "relation": "leverages"}
      ]
    }

    """

    prompt += f"""
    # Source Text
    {content}
    """

    result = llm.invoke(input=prompt)
    print(result.content)


@dataclass
class Entity:
    text: str = ""
    start_index: int = 0
    end_index: int = 0
    label: str = ""
    context: str = ""


# Init GLiner Model for NER
gliner_model = GLiNER.from_pretrained(
    model_id="gliner-community/gliner_large-v2.5",
    local_files_only=True,
    force_download=False
)
gliner_model.data_processor.transformer_tokenizer.model_max_length = 384

# Semantic Competition, too many labels at same time extraction will cause this,
# which means, one label might cause another loss weight.
# to avoid this, there are two solutions, one way is to extract several times for different labels,
# another way is to set the label name more specific, for example, set the library to programming library
gliner_labels_general = [
    "Programming language",
    "Programming framework",
    "Programming library",
    "Database",
    "Architectural pattern",
    "Tool",
    "General concept",
]
gliner_labels_responsive_web = [
    "UI Component",
    "CSS Strategy",  # Flexbox, Grid, Floats, Box Model
    "Responsive Technique",  # Media Queries, Container Queries, Fluid Grids
    "Design Token",  # Breakpoints, Viewport Units (vw/vh), Rem/Em
    "Web Image Strategy",  # Srcset, Picture Element, Lazy Loading, SVG
    "Front-end Framework",  # Bootstrap, Tailwind CSS, Foundation
    "Device/Screen Type",  # Mobile, Tablet, Desktop, Retinal Display
    "Performance Metric",  # Core Web Vitals, LCP, CLS
    "Interaction Pattern",  # Touch Gestures, Hover States, Hamburger Menu
    "HTML Structure",  # Semantic Tags, DOM Tree, Meta Viewport
]


def gliner_extraction(
    content: str, labels: list[str], threshold: float = 0.3
) -> list[Entity]:
    # gliner-community/gliner_medium-v2.5
    # gliner-community/gliner_large-v2.5
    entities = gliner_model.predict_entities(
        text=content,
        labels=labels,
        threshold=threshold,
        use_fast=False,
    )
    entities_list: list[Entity] = []

    # convert the dict of list into Entity of list
    for entity in entities:
        entities_list.append(
            Entity(
                text=entity["text"],
                start_index=entity["start"],
                end_index=entity["end"],
                label=entity["label"],
            )
        )
    return entities_list


def gliner_extraction_str_keywords(input: str, labels: list[str]) -> list[str]:
    entities: list[Entity] = gliner_extraction(input, labels)
    kws = set()
    for e in entities:
        kws.add(e.text)
    return list(kws)


def combine_entities(entities_list: list[Entity]) -> list[Entity]:
    unique_entities: list[Entity] = []
    check_dict: dict[str, Entity] = {}
    for entity in entities_list:
        if entity.text not in check_dict.keys():
            check_dict[entity.text] = entity
            unique_entities.append(entity)

    return unique_entities


cluster_name_prompt = """
**System Role:**
You are a linguistic expert specializing in Taxonomy and Named Entity Recognition (NER).
Your task is to analyze a cluster of semantically related terms and provide a single, concise,
professional category term.

**User Prompt:**
### Task:
Analyze the following list of terms belonging to the same semantic cluster.
Based on their common characteristics, generate one unique, high-level word or short sentence.
The output must without any format mark and data structure, just output the raw string.

### Examples:
- Input: Python, Rust, C++, Java; Output: computing programming language
- Input: iPhone 15, Galaxy S23, Pixel 8; Output: mobile device
- Input: CSS, transform; Output: CSS transform

### Input Cluster:

"""


def cluster_keys(keys: list[str]) -> list[str]:
    embedding_model = SentenceTransformer(
        sys_config.TRANSFORMER_MINI_L6_V2_EMBEDDING_MODEL_384
    )
    embeddings = embedding_model.encode(keys)
    cluster = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster.fit_predict(embeddings)
    cluster_dict = defaultdict(list)
    for term, label in zip(keys, cluster_labels):
        if label != -1:
            cluster_dict[label].append(term)

    cluster_group_names: list[str] = []
    for cluster_group in cluster_dict.values():
        ret = cluster_llm.invoke(input=cluster_name_prompt + "\n".join(cluster_group))
        cluster_group_names.append(str(ret.content))
        print(cluster_group)
        print(ret.content)

    return cluster_group_names


def test_entity_recognition():
    s = """
    In the context of web development, the acronym "LAMP" stands for Linux, Apache, MySQL, and PHP.
    It represents a popular combination of open-source software used together to run dynamic websites and web applications.
    Bill Gates said it's good in USA for Microsoft staffs.
    """

    s2 = """
    By default, most browsers aid user input by autocompleting the value of form fields
    where possible. Whilst the user can turn this preference on and off within the
    browser, we can now also indicate to the browser when we don't want a form or field
    to allow auto-completion. This is useful not just for sensitive data (for example bank
    account numbers) but also if you want to ensure users pay attention and enter
    something by hand. For example, for many forms I complete, if a telephone number
    is required, I enter a 'spoof' telephone number. I know I'm not the only one that does
    that (doesn't everyone?) but I can ensure that users don't enter an autocompleted
    spoof number by setting the autocomplete attribute to off on the relevant input field.
    """

    s3 = "What does the acronym “LAMP” stand for in the context of web development?"
    s33 = "I'm studying server-side web technologies for backend development in Linux environments. I’ve come across the term “LAMP stack” multiple times. What does LAMP stand for, and could you briefly explain the role of each component in the stack?"
    s333 = "In web development, the LAMP stack is often mentioned alongside other stacks like MEAN and MERN. What does LAMP stand for, and how does it compare to these other stacks in terms of technology, performance, and common use cases?"

    #
    # print('entities: ', entities_extraction(s))
    # print('verbs', verbs_extraction(s))
    # print('concepts', concepts_extraction(s))
    gliner_cluster = [
        "CSS transition properties",
        "CSS property",
        "document type",
        "dimensional layout types",
    ]
    e = gliner_extraction(
        "What are the three fundamental tenets that constitute the core of responsive web design?",
        gliner_labels_responsive_web,
    )
    print(e)


if __name__ == "__main__":
    test_entity_recognition()
