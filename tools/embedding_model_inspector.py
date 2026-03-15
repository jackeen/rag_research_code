from sentence_transformers import SentenceTransformer
import sys_config


if __name__ == '__main__':
    model = SentenceTransformer(sys_config.TRANSFORMER_MINI_L6_V2_EMBEDDING_MODEL_384)
    print(f'all-MiniLM-L6-v2, max token: {model.max_seq_length}, dim: {model.get_sentence_embedding_dimension()}')

