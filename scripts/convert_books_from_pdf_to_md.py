import re
from pathlib import Path

import fitz

# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling_core.types.doc.labels import DocItemLabel
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from tools.data_loader import BookNames, get_book_path, get_book_pdf_path


def collect_pages_as_new_pdf(
    target_file_path: str, new_file_path: str, pages: list[tuple[int, int]]
):
    """pick the pages"""
    source_file = fitz.open(target_file_path)
    dist_file = fitz.open()
    for start, end in pages:
        dist_file.insert_pdf(source_file, from_page=start, to_page=end)
    dist_file.save(new_file_path, garbage=4, deflate=True, clean=True)
    source_file.close()
    dist_file.close()


# def convert_pdf_to_md(file_path: Path, md_path: Path):
#     pipeline_options = PdfPipelineOptions(
#         do_table_structure=False,
#         images_scale=0,
#         do_ocr=False,
#         generate_page_images=False,
#     )
#     pdf_format_options = PdfFormatOption(pipeline_options=pipeline_options)
#     converter = DocumentConverter(format_options={InputFormat.PDF: pdf_format_options})

#     INCLUDE_LABELS = {
#         DocItemLabel.TITLE,
#         DocItemLabel.SECTION_HEADER,
#         DocItemLabel.TEXT,
#         DocItemLabel.LIST_ITEM,
#         DocItemLabel.PARAGRAPH,
#     }

#     result = converter.convert(source=file_path)
#     doc = result.document
#     md_content = doc.export_to_markdown(labels=INCLUDE_LABELS)
#     md_path.write_text(md_content, encoding="utf-8")


def convert_pdf_by_marker(file_path: Path, md_path: Path):
    """
    Leverage marker as the cnverter for pdf which needs massive computing resources.
    This part should use CPU to accelerate for lower time consuming and higher quality of the document.
    """
    config = {
        "disable_image_extraction": True,
        "output_format": "markdown",
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(file_path))
    md_path.write_text(rendered.markdown, encoding="utf-8")


def clean_markdown(file_path: Path):
    """depend on the situation to do"""
    pass


def generate_slim_book(book_file_name: str, page_ranges: list[tuple[int, int]]):
    book_path = get_book_path(book_file_name)
    slim_book_path = get_book_pdf_path(book_path.stem + "_slim")
    collect_pages_as_new_pdf(str(book_path), str(slim_book_path), page_ranges)


if __name__ == "__main__":
    book_2_selected_pages = [(23, 340)]
    generate_slim_book(BookNames.RESPONSIVE_WEB_DESIGN_2.value, book_2_selected_pages)

    # test
    # book_2_path = get_book_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)
    # book_2_slim_path = get_book_pdf_path(book_2_path.stem + "_slim")
    # book_2_slim_md_path = book_2_slim_path.with_suffix(".md")

    ### select pages
    # collect_pages_as_new_pdf(
    #     str(book_2_path), str(book_2_slim_path), book_2_selected_pages
    # )

    ### convert document
    # convert_pdf_by_marker(book_2_slim_path, book_2_slim_md_path)

    ### clean document
    # clean_markdown(book_2_slim_md_path)
