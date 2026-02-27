from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import config


embedding_model = SentenceTransformer(config.TRANSFORMER_EMBEDDING_MODEL_384)


# this function convert the NaN values into empty string for cosine calculation
def normal_str_list(values: list) -> list[str]:
    r = []
    for v in values:
        # NaN is not equal itself
        if v != v:
            r.append('')
        else:
            r.append(str(v))
    return r


def calculate_cosine_similarity(s1_list: list[str], s2_list: list[str]) -> list[float]:
    s1_list = normal_str_list(s1_list)
    s2_list = normal_str_list(s2_list)
    s1_vectors = embedding_model.encode(s1_list)
    s2_vectors = embedding_model.encode(s2_list)

    # zip converts the two given list into a list including value pairs
    # the result of similarity is a 2d list
    return [
        round(float(cosine_similarity([s1], [s2])[0][0]), 4) for s1, s2 in zip(s1_vectors, s2_vectors)
    ]

