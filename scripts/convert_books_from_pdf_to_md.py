from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from tools.data_loader import get_book_path, BookNames


if __name__ == "__main__":
    book_2_path = get_book_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)
    target_md_path = book_2_path.with_suffix(".md")

    pipeline_options = PdfPipelineOptions(
        do_table_structure=False,
        images_scale=0,
        do_ocr=False,
        generate_page_images=False,
        # do_chart_extraction=False,
        # do_formula_enrichment=False,
        # do_picture_classification=False,
        # do_picture_description=False,
    )
    pdf_format_options = PdfFormatOption(
        pipeline_options=pipeline_options
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: pdf_format_options
        }
    )

    result = converter.convert(source=book_2_path)
    print(result)

    # md_content = result.document.export_to_markdown()
    # target_md_path.write_text(md_content, encoding="utf-8")
