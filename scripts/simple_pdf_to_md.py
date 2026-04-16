import fitz
from pathlib import Path
from tools.data_loader import get_book_path, BookNames


if __name__ == "__main__":
    book_2_path = get_book_path(BookNames.RESPONSIVE_WEB_DESIGN_2.value)
    target_md_path = book_2_path.with_suffix(".md")

    doc = fitz.open(str(book_2_path))
    md_content = ""
    for page in doc:
        md_content += str(page.get_text("text"))

    target_md_path.write_text(md_content, encoding="utf-8")
