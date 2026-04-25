"""
The classifier powered by MoritzLaurer/deberta-v3-base-zeroshot-v2.0 model,
which support list, code, prose. (not used)
"""

from enum import StrEnum
from typing import TypedDict, cast

from transformers import pipeline


class LabelResult(TypedDict):
    sequence: str
    labels: list[str]
    scores: list[float]


class TextLabel(StrEnum):
    LIST = "list"
    CODE = "code"
    PROSE = "prose"


candidate_labels = [
    "a table of contents or an alphabetical index, consisting of structured entries with titles, keywords, or page number references",
    "computer program source code or a code snippet",
    "continuous prose, paragraphs, or narrative text with flowing sentences and discussion",
]

hypothesis_template = "This content is {}."

label_to_category: dict[str, TextLabel] = {
    candidate_labels[0]: TextLabel.LIST,
    candidate_labels[1]: TextLabel.CODE,
    candidate_labels[2]: TextLabel.PROSE,
}

# MoritzLaurer/deberta-v3-base-zeroshot-v2.0
# MoritzLaurer/deberta-v3-large-zeroshot-v2.0

#
content_classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device="cpu",
)


def classfiy_text(content: str) -> tuple[TextLabel, float]:
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
    confidence = result["scores"][0]
    return (label_to_category[top_label], confidence)
