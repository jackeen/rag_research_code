"""
https://huggingface.co/datasets/sentence-transformers/stsb
https://huggingface.co/cross-encoder/nli-deberta-v3-base
"""

import numpy as np
from sentence_transformers import CrossEncoder

# 512 max token input including two sentences
model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

# This is not suitable for long sentence, for simple sentence is ok
if __name__ == "__main__":
    # p1 = ("A woman is peeling an apple.", "A woman is peeling a potato.")
    # p2 = ("A boy is eating an apple.", "A boy is eating some meet.")
    # p3 = ("LAMP, standing for Linux, Apache, MySQL, and PHP, named in the same order.", "no")

    # p4 = ('Firewood Box', 'Flexible Box Layout')

    compared_sentence = """"""
    segments = [
        (
            compared_sentence,
            """""",
        ),
        (
            compared_sentence,
            """""",
        ),
    ]

    scores = model.predict(segments)
    print(scores)
    label_mapping = ["contradiction", "entailment", "neutral"]
    result = label_mapping[np.argmax(scores)]
    print(result)
    # [[0.49756497 - 0.94587356  0.3559419]]
    # contradiction
