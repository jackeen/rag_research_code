"""
Cosine similarity calculation
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import dotenv
import sys_config


dotenv.load_dotenv()


embedding_model = SentenceTransformer(sys_config.TRANSFORMER_MINI_L6_V2_EMBEDDING_MODEL_384)
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


def calculate_cosine_similarity_openai_embedding(s1_list: list[str], s2_list: list[str]) -> list[float]:
    s1_list = normal_str_list(s1_list)
    s2_list = normal_str_list(s2_list)
    s1_vectors = open_ai_embedding_model.embed_documents(s1_list)
    s2_vectors = open_ai_embedding_model.embed_documents(s2_list)

    # zip converts the two given list into a list including value pairs
    # the result of similarity is a 2d list
    return [
        round(float(cosine_similarity([s1], [s2])[0][0]), 4) for s1, s2 in zip(s1_vectors, s2_vectors)
    ]


def calculate_cosine_similarity_transformer_embedding(
        model_name: str,
        s1_list: list[str],
        s2_list: list[str]
) -> list[float]:
    s1_list = normal_str_list(s1_list)
    s2_list = normal_str_list(s2_list)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    s1_vectors = embeddings.embed_documents(s1_list)
    s2_vectors = embeddings.embed_documents(s2_list)

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


if __name__ == '__main__':
    # s1 = ['apple', 'meet']
    # s2 = ['apple', 'FQQ']

    # mini[1.0, 0.2322]
    # granite small[1.0, 0.7451]
    # granite[1.0, 0.7199]
    # openai mini[1.0, 0.2231]

    # new log
    s1 = ['LAMP, standing for Linux, Apache, MySQL, and PHP, named in the same order.']
    s2 = ['In the context of web development, the acronym "LAMP" stands for Linux, Apache, MySQL, and PHP. It represents a popular combination of open-source software used to create a web server environment: Linux as the operating system, Apache as the web server, MySQL as the database, and PHP as the scripting language. This stack is widely used for hosting dynamic websites and web applications.']
    # mini[0.7824]
    # granite small[0.9564]
    # granite[0.9589]
    # openai mini[0.8561]

    # old log
    # s1 = ['LAMP, standing for Linux, Apache, MySQL, and PHP, named in the same order.']
    # s2 = ['In the context of web development, the acronym "LAMP" stands for Linux, Apache, MySQL, and PHP. It represents a popular combination of open-source software used together to run dynamic websites and web applications.']
    # mini[0.7765]
    # granite small[0.9544]
    # granite[0.9598]
    # Qwen[0.8375]
    # openai mini[0.8544]

    # s1 = ['Tom like eat meet in the evening at friday']
    # s2 = ['Tom do not like eat meet in the evening at friday']
    # mini[0.9332]
    # granite small[0.9723]
    # granite[0.9821]
    # Qwen[0.8644]
    # openai mini[0.8534]

    print('mini', calculate_cosine_similarity(s1, s2))
    print('granite small', calculate_cosine_similarity_transformer_embedding(
        sys_config.TRANSFORMER_GRANITE_SMALL_R2_EMBEDDING_MODEL_384,
        s1,
        s2
    ))
    print('granite', calculate_cosine_similarity_transformer_embedding(
        sys_config.TRANSFORMER_GRANITE_R2_EMBEDDING_MODEL_768,
        s1,
        s2
    ))
    print('Qwen', calculate_cosine_similarity_transformer_embedding(
        sys_config.TRANSFORMER_QWEN3_06B_EMBEDDING_MODEL_1024,
        s1,
        s2
    ))
    print('openai mini', calculate_cosine_similarity_openai_embedding(
        s1,
        s2
    ))


