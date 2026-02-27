"""
This module depends on marker-pdf.
"""

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from pathlib import Path

# import config

# optional llm service
# from marker.services.ollama import OllamaService
# llm_service = OllamaService(
#     model=config.OLLAMA_GRANITE_MODEL,
#     base_url=config.OLLAMA_URL_BASE
# )

artifact_dict = create_model_dict()
config_dict = {
    "output_format": "markdown",
}
marker_converter = PdfConverter(
    artifact_dict=artifact_dict,
    config=config_dict,
    # llm_service=llm_service
)


def clean_and_save_md(file_path: str, content: str):
    # clean
    file_path_obj = Path(file_path)
    md_file_path = file_path_obj.with_suffix('.md')
    md_file_path.write_text(content, encoding='utf-8')
    return md_file_path


def convert(pdf_file_path):
    rendered = marker_converter(pdf_file_path)
    full_md_text, images, out_meta = text_from_rendered(rendered)
    return clean_and_save_md(pdf_file_path, full_md_text)



