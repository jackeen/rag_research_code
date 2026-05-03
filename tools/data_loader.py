"""
The model for files' IO
"""

from datetime import datetime
from enum import Enum
from pathlib import Path


class BookNames(Enum):
    """The enum book names and their files, which can provide an easy way to access them."""

    INTRO_WEB_DEV_1 = "1.pdf"
    RESPONSIVE_WEB_DESIGN_2 = "2.pdf"
    LEARNING_REACT_4 = "4.pdf"
    DESIGN_PATTERN_5 = "5.pdf"
    STRUCTURE_INTERPRETATION_6 = "6.pdf"
    SOCIAL_MARKETING_8 = "8.pdf"


class References(Enum):
    """The references for every book, which is used to track the chunk where is from."""

    INTRO_WEB_DEV_1 = "Mendez, M. (2014). The Missing Link: An Introduction to Web Development and Programming. Open SUNY Textbooks."
    RESPONSIVE_WEB_DESIGN_2 = (
        "Frain, B. (2012). Responsive web design with HTML5 and CSS3. Packt Publishing."
    )
    LEARNING_REACT_4 = "Banks, A., & Porcello, E. (2017). Learning React: functional web development with React and Redux. ' O'Reilly Media, Inc.'."
    DESIGN_PATTERN_5 = (
        "Freeman, E., & Robson, E. (2020). Head first design patterns. O'Reilly Media."
    )
    STRUCTURE_INTERPRETATION_6 = "Abelson, H., & Sussman, G. J. (1996). Structure and interpretation of computer programs (p. 688). The MIT Press."
    SOCIAL_MARKETING_8 = "Donovan, R., & Henley, N. (2010). Principles and practice of social marketing: an international perspective. Cambridge University Press."


def get_book_path(book_file_name: str) -> Path:
    """
    Get the experiment original book path as the data.
    :param book_file_name:
    :return: the absolute path of the book
    """
    return Path(__file__).parent.parent.absolute().joinpath(f"books/{book_file_name}")


def get_book_pdf_path(book_file_name: str) -> Path:
    """
    Get the pdf book path from books dir.
    :param book_file_name:
    :return: the absolute path of the book
    """
    return (
        Path(__file__).parent.parent.absolute().joinpath(f"books/{book_file_name}.pdf")
    )


def get_book_md_path(book_file_name: str) -> Path:
    """
    Get the markdown file path from books dir.
    :param book_file_name: the book name
    :return: the absolute path of the md file
    """
    return (
        Path(__file__).parent.parent.absolute().joinpath(f"books/{book_file_name}.md")
    )


def get_excel_data_path(excel_name: str) -> Path:
    """
    Get the experiment result Excel data path by given file name.
    :param excel_name: the Excel file name
    :return: the absolute path of the Excel file
    """
    return Path(__file__).parent.parent.absolute().joinpath(f"data/{excel_name}.xlsx")


def get_csv_data_path(csv_name: str) -> Path:
    return Path(__file__).parent.parent.absolute().joinpath(f"data/csv/{csv_name}.csv")


def get_json_data_path(json_name: str) -> Path:
    return (
        Path(__file__).parent.parent.absolute().joinpath(f"data/json/{json_name}.json")
    )


def get_time_stamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_csv_log_path(csv_name: str) -> Path:
    file_name = f"logs/{csv_name}_{get_time_stamp_string()}.csv"
    return Path(__file__).parent.parent.absolute().joinpath(file_name)


def get_no_tail_csv_log_path(csv_name: str) -> Path:
    file_name = f"logs/{csv_name}.csv"
    return Path(__file__).parent.parent.absolute().joinpath(file_name)


def get_txt_log_path(txt_name: str) -> Path:
    file_name = f"log/{txt_name}_{get_time_stamp_string()}.txt"
    return Path(__file__).parent.parent.absolute().joinpath(file_name)


if __name__ == "__main__":
    # for test
    print(get_csv_log_path("test_log"))
