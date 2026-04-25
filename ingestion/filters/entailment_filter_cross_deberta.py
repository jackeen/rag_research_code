"""
The entailment verifier for paragrahp.

"""

import re

import torch

# import torch.nn.functional as TF
from sentence_transformers import CrossEncoder

LABELS = ["contradiction", "entailment", "neutral"]
ENTAILMENT_INDEX = 1

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

# cross-encoder/nli-deberta-v3-base
# cross-encoder/nli-deberta-v3-xsmall
model = CrossEncoder("cross-encoder/nli-deberta-v3-large", device=device)


def _split_sentences(paragraph: str) -> list[str]:
    # split sentences based on . ! ? marks
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())

    # some whole sentences may include \n, such as treated code block
    return [s.replace("\n", " ").strip() for s in sentences if len(s.strip()) > 10]


def is_paragraph_supports_topic(
    topic: str, paragraph: str
) -> tuple[bool, list[dict[str, str]]]:
    sentences = _split_sentences(paragraph)
    if not sentences:
        return (False, [])

    pairs = [(sentence, topic) for sentence in sentences]

    # shape: (n_sentences, 3)
    scores = model.predict(pairs)

    # model.predict() returns raw logits (unbounded values) for each label.
    # the softmax to convert them into probabilities that sum to 1,
    # allowing us to interpret each score as a confidence level.
    # dim=-1 normalizes across the 3 labels for each sentence independently.
    probs = torch.softmax(torch.tensor(scores), dim=-1)

    results: list[dict[str, str]] = []
    is_supported = False
    for i, sentence in enumerate(sentences):
        pred_index = probs[i].argmax().item()
        pred_label = LABELS[int(pred_index)]
        entailment_score = probs[i][ENTAILMENT_INDEX].item()

        if pred_label == LABELS[ENTAILMENT_INDEX]:
            is_supported = True

        results.append(
            {
                "sentence": sentence,
                "label": pred_label,
                "entailment_score": str(round(entailment_score, 4)),
            }
        )

    return (is_supported, results)


# test
if __name__ == "__main__":
    topic = "Three tenets of responsive web design are media queries, flexible layouts, and flexible media"

    s = """
    The web design is good work.
    """

    s2 = """
    The core of responsive design includes flexible layouts, it also depends on media queries.
    """

    ret = is_paragraph_supports_topic(topic, s2)
    print(ret)
