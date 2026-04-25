"""
The label
"""

from typing import TypedDict, cast

from transformers import pipeline


class LabelResult(TypedDict):
    sequence: str
    labels: list[str]
    scores: list[float]


# MoritzLaurer/deberta-v3-base-zeroshot-v2.0
# MoritzLaurer/deberta-v3-large-zeroshot-v2.0
# feature-extraction
content_classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    # device="cpu",
)

candidate_labels = [
    "a meta-commentary about what was covered or will be covered",
    "substantive content that explains ideas in depth",
]

hypothesis_template = "This text is {}"

label_to_category = {
    candidate_labels[0]: "NO",
    candidate_labels[1]: "YES",
}


def classify_knowledge_chunk(content: str) -> tuple[str, float]:
    result = cast(
        LabelResult,
        content_classifier(
            content,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        ),
    )
    top_label = result["labels"][0]
    top_confidence = result["scores"][0]

    return (label_to_category[top_label], top_confidence)


paragraphs = """"""


if __name__ == "__main__":
    segments = paragraphs.split("\n\n")
    for seg in segments:
        print(classify_knowledge_chunk(seg))
