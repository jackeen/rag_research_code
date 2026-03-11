from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import dotenv
import sys_config


dotenv.load_dotenv()


embedding_model = SentenceTransformer(sys_config.TRANSFORMER_EMBEDDING_MODEL_384)
open_ai_embedding_model = OpenAIEmbeddings(model=sys_config.OPEN_AI_EMBEDDING_MODEL)


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


def calculate_max_cosine_similarities(chunks: list[str], answers: list[str]) -> list[float]:
    """
    For compare the M * N strings similarities
    :param chunks:
    :param answers:
    :return:
    """
    chunks_vectors = open_ai_embedding_model.embed_documents(chunks)
    answers_vectors = open_ai_embedding_model.embed_documents(answers)
    similarity_matrix = cosine_similarity(chunks_vectors, answers_vectors)
    max_cosine_similarities = similarity_matrix.max(axis=1)
    return [round(float(v), 4) for v in max_cosine_similarities]

