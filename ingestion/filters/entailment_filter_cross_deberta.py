"""
The entailment verifier for paragrahp.

"""

import re

import torch

# import torch.nn.functional as TF
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

LABELS = ["contradiction", "entailment", "neutral"]
ENTAILMENT_INDEX = 1

device = "cpu"
# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"


CROSS_NLI_XSMALL_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
CROSS_NLI_BASE_MODEL = "cross-encoder/nli-deberta-v3-base"
CROSS_NLI_LARGE_MODEL = "cross-encoder/nli-deberta-v3-large"
model = CrossEncoder(CROSS_NLI_LARGE_MODEL, device=device)

# for counting input
# tokenizer = AutoTokenizer.from_pretrained(CROSS_NLI_LARGE_MODEL)


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

    topic_2 = "It’s a cliche at this point to talk about JavaScript Fatigue, but the source of this fake illness can be traced back to the building process."

    s3 = """
    t's a cliche at this point to talk about [JavaScript Fatigue,](http://bit.ly/2pSiuE4) but the source of this fake illness can be traced back to the building process. In the past, you just added Java‐ Script files to your page. Now the JavaScript file has to be built, usually with an auto‐ mated continuous delivery process. There's emerging syntax that has to be transpiled to work in all browsers. There's JSX that has to be converted to JavaScript. There's SCSS that you might want to preprocess. These components need to be tested, and
    they have to pass. You might love React, but now you also need to be a webpack expert, handling code splitting, compression, testing, and on and on.
    """

    topic_3 = """
The Strategy Pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
    """

    s4 = """
    Congratulations on your first pattern!
    You just applied your first design pattern—the **STRATEGY** Pattern. That's right, you used the Strategy Pattern to rework the SimUDuck app.
    Thanks to this pattern, the simulator is ready for any changes those execs might cook up on their next business trip to Maui.
    Now that we've made you take the long road to learn it, here's the formal definition of this pattern:
    **The Strategy Pattern** defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
    Use THIS definition when you need to impress friends and influence key executives.
    Below you'll find a mess of classes and interfaces for an action adventure game. You'll find classes for game characters along with classes for weapon behaviors the characters can use in the game. Each character can make use of one weapon at a time, but can change weapons at any time during the game. Your job is to sort it all out...
    (Answers are at the end of the chapter.)

    """

    s5 = """
    **The Observer Pattern** defines a one-to-many dependency between objects so that when one object changes state, all of its dependents are notified and updated automatically.
    """

    topic_4 = """
The key point of difference to all other branches of marketing, is that the social marketer’s goals relate to the wellbeing of the community, whereas for all others, the marketer’s goals relate to the wellbeing of the marketer.
    """

    s6 = """
    - What are the major differences between commercial marketing and social marketing?
    - How does social marketing differ from cause marketing?
    """

    # tokens = tokenizer.encode(topic_3 + s4, add_special_tokens=False)
    # print(len(tokens))

    ret, _ = is_paragraph_supports_topic(topic_4, s6)
    print(ret)
