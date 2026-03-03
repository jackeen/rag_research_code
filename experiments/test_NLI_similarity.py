"""
https://huggingface.co/datasets/sentence-transformers/stsb
https://huggingface.co/cross-encoder/nli-deberta-v3-base
"""

from sentence_transformers import CrossEncoder
import numpy as np

model = CrossEncoder('cross-encoder/nli-deberta-v3-base')


if __name__ == '__main__':
    p1 = ("A woman is peeling an apple.", "A woman is peeling a potato.")
    p2 = ("A boy is eating an apple.", "A boy is eating some meet.")
    p3 = ("LAMP, standing for Linux, Apache, MySQL, and PHP, named in the same order.", "no")
    scores = model.predict([p2])
    print(scores)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    result = label_mapping[np.argmax(scores)]
    print(result)
    # [[0.49756497 - 0.94587356  0.3559419]]
    # contradiction

