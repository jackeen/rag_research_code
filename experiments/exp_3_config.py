"""The configurations about exp 3"""

from enum import Enum


class CollectionNames(Enum):
    # proved the keyword filter is not useful, and fixed chunk break semantic structure
    FIX_CHUNKING = "p_exp_3_entity"
    FIX_CHUNKING_1_KEYWORD = "p_exp_3_entity_filtered"
    FIX_CHUNKING_3_KEYWORDS = "p_exp_3_entity_filtered_3"
    FIX_CHUNKING_6_KEYWORDS = "p_exp_3_entity_filtered_6"

    # keep the semantic structure by md sections and LLM semantic filter
    MD_HEAD_CHUNKING = "p_exp_3_md"
    MD_HEAD_CHUNKING_2 = "p_exp_3_md_2"
    MD_HEAD_CHUNKING_LLM = "p_exp_3_md_llm_refined"

    # lower the cost of semantic filter
    MD_HEAD_CHUNKING_ML_PAGE = "p_exp_3_md_multi_layer_page_refined"
    MD_HEAD_CHUNKING_ML_PAGE_CONSINE_8_CHUNK = (
        "p_exp_3_md_multi_layer_page_cosine_8_chunk_refined"
    )
    MD_HEAD_CHUNKING_ML_PARAGRAPH = "p_exp_3_md_multi_layer_paragraph_refined"
    MD_HEAD_CHUNKING_ML_PARAGRAPH_NO_CODE = (
        "p_exp_3_md_multi_layer_paragraph_no_code_refined"
    )
    MD_HEAD_CHUNKING_ML_PARAGRAPH_NLI = "p_exp_3_md_multi_layer_paragraph_nli_refined"
    MD_HEAD_CHUNKING_ML_PARAGRAPH_GEMMA = (
        "p_exp_3_md_multi_layer_paragraph_gemma_refined"
    )

    MD_HEAD_CHUNKING_PAGES_GROUP_1 = "p_exp_3_md_multi_layer_pg_1"
    MD_HEAD_CHUNKING_PAGES_GROUP_1_TMP = "p_exp_3_md_multi_layer_pg_1_tmp"
    MD_HEAD_CHUNKING_PAGES_GROUP_2 = "p_exp_3_md_multi_layer_pg_2"
    MD_HEAD_CHUNKING_PAGES_GROUP_2_TMP = "p_exp_3_md_multi_layer_pg_2_tmp"
    MD_HEAD_CHUNKING_PAGES_GROUP_4 = "p_exp_3_md_multi_layer_pg_4"
    MD_HEAD_CHUNKING_PAGES_GROUP_4_TMP = "p_exp_3_md_multi_layer_pg_4_tmp"
    MD_HEAD_CHUNKING_PAGES_GROUP_5 = "p_exp_3_md_multi_layer_pg_5"
    MD_HEAD_CHUNKING_PAGES_GROUP_5_TMP = "p_exp_3_md_multi_layer_pg_5_tmp"
    MD_HEAD_CHUNKING_PAGES_GROUP_6 = "p_exp_3_md_multi_layer_pg_6"
    MD_HEAD_CHUNKING_PAGES_GROUP_6_TMP = "p_exp_3_md_multi_layer_pg_6_tmp"
    MD_HEAD_CHUNKING_PAGES_GROUP_8 = "p_exp_3_md_multi_layer_pg_8"
    MD_HEAD_CHUNKING_PAGES_GROUP_8_TMP = "p_exp_3_md_multi_layer_pg_8_tmp"

    MD_HEAD_CHUNKING_FULL_1 = "p_exp_3_md_multi_layer_1"
    MD_HEAD_CHUNKING_FULL_1_TMP = "p_exp_3_md_multi_layer_1_tmp"
    MD_HEAD_CHUNKING_FULL_6 = "p_exp_3_md_multi_layer_6"
    MD_HEAD_CHUNKING_FULL_6_TMP = "p_exp_3_md_multi_layer_6_tmp"
